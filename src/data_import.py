"""Download ERCOT DAM and RTM settlement point prices for all hubs, one CSV per month.

Calls the ERCOT Public API directly and reads the response's named fields. The `ercot` package's
er.dam/er.rtm guess column meanings from the data and mislabel RTM hour/interval columns, so we only
use its ErcotAuth (handles the 1-hour token refresh).

Credentials: ERCOT_USERNAME, ERCOT_PASSWORD, ERCOT_SUBSCRIPTION_KEY. Re-running skips finished months.
The API only has data from mid-December 2023 onwards.
"""
import os
import time

import pandas as pd
import requests
from ercot.auth import ErcotAuth
from ercot.models import BASE_URL

from config import DOWNLOAD_START, END, HUBS, RAW_DAM, RAW_RTM

DAM_PATH = "/np4-190-cd/dam_stlmnt_pnt_prices"
RTM_PATH = "/np6-905-cd/spp_node_zone_hub"


def month_ranges(start, end):
    """[(first_day, last_day), ...] per calendar month, clipped to [start, end], as YYYY-MM-DD."""
    out = []
    for m in pd.period_range(start, end, freq="M"):
        a = max(m.start_time, pd.Timestamp(start))
        b = min(m.end_time.normalize(), pd.Timestamp(end))
        out.append((a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
    return out


def with_retry(call, tries=6, wait=60):
    """Run call(); on HTTP 429 (rate limit) wait 60s, 120s, ... (or the server's Retry-After) and try again."""
    for i in range(tries):
        try:
            return call()
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 429 or i == tries - 1:
                raise
            retry_after = e.response.headers.get("Retry-After", "")
            time.sleep(int(retry_after) if retry_after.isdigit() else wait * (i + 1))


def fetch(auth, path, params):
    """All pages of one API query as a DataFrame with the API's own field names."""
    rows, page = [], 1
    while True:
        r = requests.get(BASE_URL + path, headers=auth.get_headers(),
                         params={**params, "size": 100_000, "page": page}, timeout=120)
        r.raise_for_status()
        payload = r.json()
        fields = [f["name"] for f in payload["fields"]]
        rows += payload.get("data") or []
        if page >= payload["_meta"]["totalPages"]:
            return pd.DataFrame(rows, columns=fields)
        page += 1


def _interval_ending(dates, begin_offset, length, dst_flag):
    """UTC end of each interval. Localize the interval *start* (always a real wall-clock time);
    DSTFlag 'Y' marks the repeated fall-back hour, which is standard time."""
    repeated = dst_flag.astype(str).str.upper().isin(["Y", "TRUE"]).to_numpy()
    begin = (pd.to_datetime(dates) + begin_offset).dt.tz_localize("America/Chicago", ambiguous=~repeated)
    return (begin + length).dt.tz_convert("UTC")


def to_long_dam(raw):
    hour = raw["hourEnding"].astype(str).str.split(":").str[0].astype(int)
    return pd.DataFrame({
        "date": raw["deliveryDate"],
        "settlement_point": raw["settlementPoint"],
        "hour": hour,
        "interval_ending": _interval_ending(raw["deliveryDate"], pd.to_timedelta(hour - 1, unit="h"),
                                            pd.Timedelta(hours=1), raw["DSTFlag"]),
        "price": pd.to_numeric(raw["settlementPointPrice"]),
    })


def to_long_rtm(raw):
    hour = raw["deliveryHour"].astype(int)
    interval = raw["deliveryInterval"].astype(int)
    offset = pd.to_timedelta(hour - 1, unit="h") + pd.to_timedelta((interval - 1) * 15, unit="min")
    return pd.DataFrame({
        "date": raw["deliveryDate"],
        "settlement_point": raw["settlementPoint"],
        "interval": (hour - 1) * 4 + interval,
        "interval_ending": _interval_ending(raw["deliveryDate"], offset, pd.Timedelta(minutes=15), raw["DSTFlag"]),
        "price": pd.to_numeric(raw["settlementPointPrice"]),
    })


def fetch_month(fetch_hub, first, last):
    """One request per hub: the API's settlementPoint filter takes a single value, not a comma list."""
    parts = []
    for hub in HUBS:
        parts.append(with_retry(lambda: fetch_hub(hub, first, last)))
        time.sleep(1)  # be gentle with ERCOT's rate limit
    df = pd.concat(parts, ignore_index=True)
    got = set(df["settlement_point"]) if "settlement_point" in df else set()
    missing = [h for h in HUBS if h not in got]
    if missing:
        # Never save a partial month: a saved file is treated as finished and skipped on rerun.
        raise RuntimeError(f"no data for {missing} in {first}..{last}")
    return df


def download_all():
    auth = ErcotAuth(os.environ["ERCOT_USERNAME"], os.environ["ERCOT_PASSWORD"], os.environ["ERCOT_SUBSCRIPTION_KEY"])
    markets = {
        "DAM": (RAW_DAM, DAM_PATH, to_long_dam),
        "RTM": (RAW_RTM, RTM_PATH, to_long_rtm),
    }
    for folder, _, _ in markets.values():
        folder.mkdir(parents=True, exist_ok=True)
    for first, last in month_ranges(DOWNLOAD_START, END):
        for market, (folder, path, to_long) in markets.items():
            out = folder / f"{market}_hubs_{first[:7]}.csv"
            if out.exists():
                continue

            def fetch_hub(hub, a, b, path=path, to_long=to_long):
                params = {"deliveryDateFrom": a, "deliveryDateTo": b, "settlementPoint": hub}
                return to_long(fetch(auth, path, params))

            fetch_month(fetch_hub, first, last).to_csv(out, index=False)
            print(f"saved {out.name}")


if __name__ == "__main__":
    download_all()
