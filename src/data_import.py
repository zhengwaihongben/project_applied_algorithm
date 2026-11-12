"""Download ERCOT DAM and RTM settlement point prices for all hubs, one CSV per month.

Credentials: ERCOT_USERNAME, ERCOT_PASSWORD, ERCOT_SUBSCRIPTION_KEY. Re-running skips finished months.
"""
import os
import time

import ercot as er
import pandas as pd
import requests

from config import DOWNLOAD_START, END, HUBS, RAW_DAM, RAW_RTM


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


def fetch_month(fetch, first, last):
    """One request per hub: the API's settlementPoint filter takes a single value, not a comma list."""
    parts = [with_retry(lambda p=p: fetch(p, first, last, format="long")) for p in HUBS]
    df = pd.concat(parts, ignore_index=True)
    got = set(df["settlement_point"]) if "settlement_point" in df else set()
    missing = [h for h in HUBS if h not in got]
    if missing:
        # Never save a partial month: a saved file is treated as finished and skipped on rerun.
        raise RuntimeError(f"no data for {missing} in {first}..{last}")
    return df


def download_all():
    # save=False: don't write credentials to ~/.ercot/credentials.json in plain text
    er.configure(os.environ["ERCOT_USERNAME"], os.environ["ERCOT_PASSWORD"],
                 os.environ["ERCOT_SUBSCRIPTION_KEY"], save=False)
    RAW_DAM.mkdir(parents=True, exist_ok=True)
    RAW_RTM.mkdir(parents=True, exist_ok=True)
    for first, last in month_ranges(DOWNLOAD_START, END):
        for market, fetch, folder in (("DAM", er.dam, RAW_DAM), ("RTM", er.rtm, RAW_RTM)):
            path = folder / f"{market}_hubs_{first[:7]}.csv"
            if path.exists():
                continue
            fetch_month(fetch, first, last).to_csv(path, index=False)
            print(f"saved {path.name}")


if __name__ == "__main__":
    download_all()
