"""Wind, solar and load forecasts as they stood at 10:00 the day before delivery.

The `ercot` package doesn't expose the publication-time filter, so we call the API directly
(data_import.fetch), reusing ErcotAuth for the 1-hour token refresh.

    uv run python src/covariates.py probe 2024-06-02   # check field names on one day
    uv run python src/covariates.py                    # full download, resumable
"""
import os
import sys
import time

import pandas as pd
from ercot.auth import ErcotAuth

from config import DOWNLOAD_START, END, RAW_COV
from data_import import fetch, month_ranges, with_retry

ENDPOINTS = {
    "WIND": "/np4-742-cd/wpp_hrly_actual_fcast_geo",
    "SOLAR": "/np4-745-cd/spp_hrly_actual_fcast_geo",
    "LOAD": "/np3-565-cd/lf_by_model_weather_zone",
}
CUTOFF = pd.Timedelta(hours=14)    # delivery day 00:00 minus 14h = 10:00 on D-1
LOOKBACK = pd.Timedelta(hours=48)


def _auth():
    return ErcotAuth(os.environ["ERCOT_USERNAME"], os.environ["ERCOT_PASSWORD"], os.environ["ERCOT_SUBSCRIPTION_KEY"])


def pick_vintage(df, day):
    """Rows of the latest forecast published before 10:00 on the day before `day` (local time)."""
    posted = pd.to_datetime(df["postedDatetime"])
    ok = posted < day - CUTOFF
    if not ok.any():
        return df.iloc[0:0]
    return df[posted == posted[ok].max()]


def _in_use(df):
    return df["inUseFlag"].astype(str).str.upper().isin(["TRUE", "Y", "1"])


def _params(day):
    fmt = "%Y-%m-%dT%H:%M:%S"
    return {
        "deliveryDateFrom": day.strftime("%Y-%m-%d"),
        "deliveryDateTo": day.strftime("%Y-%m-%d"),
        "postedDatetimeFrom": (day - CUTOFF - LOOKBACK).strftime(fmt),
        "postedDatetimeTo": (day - CUTOFF).strftime(fmt),
    }


def download_covariates():
    auth = _auth()
    RAW_COV.mkdir(parents=True, exist_ok=True)
    for first, last in month_ranges(DOWNLOAD_START, END):
        for name, path in ENDPOINTS.items():
            out = RAW_COV / f"{name}_{first[:7]}.csv"
            if out.exists():
                continue
            parts = []
            for day in pd.date_range(first, last):
                raw = with_retry(lambda: fetch(auth, path, _params(day)))
                if name == "LOAD" and not raw.empty:
                    raw = raw[_in_use(raw)]
                parts.append(pick_vintage(raw, day) if not raw.empty else raw)
                time.sleep(0.3)
            df = pd.concat(parts, ignore_index=True)
            if df.empty:
                # Never save an empty month: a saved file is treated as finished and skipped on rerun.
                print(f"no {name} data for {first[:7]}, skipped")
                continue
            df.to_csv(out, index=False)
            print(f"saved {out.name} ({df['deliveryDate'].nunique()} days)")


def probe(day):
    """Print what the API returns for one delivery day, to check field names and formats."""
    auth = _auth()
    for name, path in ENDPOINTS.items():
        raw = fetch(auth, path, _params(pd.Timestamp(day)))
        print(f"== {name}: {len(raw)} rows, columns: {list(raw.columns)}")
        if not raw.empty:
            print("posted:", sorted(raw["postedDatetime"].unique())[-3:])
            if name == "LOAD":
                print(raw.groupby(["model", "inUseFlag"]).size())
            print(raw.head(2).to_string())


if __name__ == "__main__":
    probe(sys.argv[2]) if sys.argv[1:2] == ["probe"] else download_covariates()
