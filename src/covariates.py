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
from pandas.tseries.holiday import USFederalHolidayCalendar

from config import DOWNLOAD_START, END, LOAD_FIELDS, PROCESSED, RAW_COV, SOLAR_FIELDS, WIND_FIELDS
from data_import import fetch, month_ranges, with_retry

COVARIATE_FILE = PROCESSED / "covariates.csv"

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


def to_hour_ending(df):
    """ERCOT deliveryDate + hourEnding (1-24, '1' or '01:00') + DSTFlag -> UTC hour ending.

    Localize the hour *beginning* (always a real wall-clock time), then add 1h.
    DSTFlag true marks the repeated fall-back hour, which is standard time.
    """
    he = df["hourEnding"].astype(str).str.split(":").str[0].astype(int)
    begin = pd.to_datetime(df["deliveryDate"]) + pd.to_timedelta(he - 1, unit="h")
    repeated = df["DSTFlag"].astype(str).str.upper().isin(["Y", "TRUE"]).to_numpy()
    # The solar report labels the spring-forward gap as HE2 instead of HE3; at 2am solar is 0, so shift it.
    local = begin.dt.tz_localize("America/Chicago", ambiguous=~repeated, nonexistent="shift_forward")
    return (local + pd.Timedelta(hours=1)).dt.tz_convert("UTC")


def calendar(hour_ending):
    start = hour_ending.dt.tz_convert("America/Chicago") - pd.Timedelta(hours=1)
    day = start.dt.tz_localize(None).dt.normalize()
    holidays = USFederalHolidayCalendar().holidays(day.min(), day.max())
    return pd.DataFrame({
        "hour": start.dt.hour, "dow": start.dt.dayofweek, "month": start.dt.month,
        "holiday": day.isin(holidays).astype(int),
    }, index=hour_ending.index)


def _hourly(kind, fields):
    raw = pd.concat([pd.read_csv(f) for f in sorted(RAW_COV.glob(f"{kind}_*.csv"))], ignore_index=True)
    out = raw.assign(hour_ending=to_hour_ending(raw)).rename(columns=fields)
    cols = list(fields.values())
    out[cols] = out[cols].apply(pd.to_numeric, errors="coerce")
    return out.groupby("hour_ending")[cols].mean()


def build_covariates():
    """One row per UTC hour: each forecast as it stood at 10:00 the day before delivery, plus derived inputs."""
    cov = pd.concat([_hourly("WIND", WIND_FIELDS), _hourly("SOLAR", SOLAR_FIELDS), _hourly("LOAD", LOAD_FIELDS)],
                    axis=1).sort_index()
    cov["net_load"] = cov["load_sys"] - cov["wind_sys"] - cov["solar_sys"]
    # shortcut: proxy for power pushing out of the West (weather-zone sums, not ERCOT's load-zone mapping)
    cov["west_surplus"] = (cov["wind_west"] + cov["wind_pan"] + cov["solar_farwest"] + cov["solar_cwest"]
                           - cov["load_farwest"] - cov["load_west"])
    cov = cov.rename_axis("hour_ending").reset_index()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cov.to_csv(COVARIATE_FILE, index=False)
    print(cov.describe().T[["count", "mean", "min", "max"]].to_string())
    print("first/last hour:", cov["hour_ending"].min(), cov["hour_ending"].max())
    print("hours with any missing value:", int(cov.isna().any(axis=1).sum()), "of", len(cov))


def load_covariates():
    cov = pd.read_csv(COVARIATE_FILE)
    cov["hour_ending"] = pd.to_datetime(cov["hour_ending"], utc=True)
    return cov.set_index("hour_ending")


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
    cmd = sys.argv[1:2]
    if cmd == ["probe"]:
        probe(sys.argv[2])
    elif cmd == ["build"]:
        build_covariates()
    else:
        download_covariates()
