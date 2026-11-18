"""Extra inputs for the LightGBM model, all known at 10:00 on T-1 when used as in lgbm.py.

| kind     | what                                    | used for day T as             |
|----------|-----------------------------------------|-------------------------------|
| WINDACT  | actual wind output by region            | forecast miss on T-2          |
| SOLARACT | actual solar output by region           | forecast miss on T-2          |
| LOADACT  | actual load by weather zone             | forecast miss on T-2          |
| OUTAGE   | generator outage MW by zone (forecast)  | value for T, 10:00 T-1 vintage |
| SHADOW   | DAM shadow prices of binding constraints| T-1's auction (public on T-2)  |

    uv run python src/extra_data.py probe 2024-06-02   # check what the API returns for one day
    uv run python src/extra_data.py                    # full download, resumable, one CSV per month and kind
    uv run python src/extra_data.py build              # -> data/processed/extra.csv
"""
import sys
import time

import pandas as pd

from config import DOWNLOAD_START, END, PROCESSED, RAW_COV
from covariates import CUTOFF, LOOKBACK, _auth, pick_vintage, to_hour_ending
from data_import import fetch, month_ranges, with_retry

FMT = "%Y-%m-%dT%H:%M:%S"


def _settled(day):
    """Actuals for `day`: the last posting before 10:00 the next day (when actuals for all hours exist)."""
    nxt = day + pd.Timedelta(days=1)
    return {"deliveryDateFrom": f"{day:%Y-%m-%d}", "deliveryDateTo": f"{day:%Y-%m-%d}",
            "postedDatetimeFrom": f"{nxt:{FMT}}", "postedDatetimeTo": f"{nxt + pd.Timedelta(hours=10):{FMT}}"}


def _vintage(day, prefix="operatingDate"):
    return {f"{prefix}From": f"{day:%Y-%m-%d}", f"{prefix}To": f"{day:%Y-%m-%d}",
            "postedDatetimeFrom": f"{day - CUTOFF - LOOKBACK:{FMT}}", "postedDatetimeTo": f"{day - CUTOFF:{FMT}}"}


def _latest(df, day):
    posted = pd.to_datetime(df["postedDatetime"])
    return df[posted == posted.max()]


SOURCES = {   # kind -> (endpoint, request params for one day, which rows to keep)
    "WINDACT": ("/np4-742-cd/wpp_hrly_actual_fcast_geo", _settled, _latest),
    "SOLARACT": ("/np4-745-cd/spp_hrly_actual_fcast_geo", _settled, _latest),
    "OUTAGE": ("/np3-233-cd/hourly_res_outage_cap", _vintage, pick_vintage),
}
MONTHLY = {   # no publication-time choice needed, so one request per month (ERCOT rate-limits per request)
    "LOADACT": ("/np6-345-cd/act_sys_load_by_wzn", "operatingDay"),
    "SHADOW": ("/np4-191-cd/dam_shadow_prices", "deliveryDate"),
}


def download():
    auth = _auth()
    RAW_COV.mkdir(parents=True, exist_ok=True)
    for first, last in month_ranges(DOWNLOAD_START, END):
        for kind, (path, field) in MONTHLY.items():
            out = RAW_COV / f"{kind}_{first[:7]}.csv"
            if not out.exists():
                df = with_retry(lambda: fetch(auth, path, {f"{field}From": first, f"{field}To": last}))
                if df.empty:
                    print(f"no {kind} data for {first[:7]}, skipped")
                else:
                    df.to_csv(out, index=False)
                    print(f"saved {out.name} ({len(df):,} rows)", flush=True)
        for kind, (path, params, keep) in SOURCES.items():
            out = RAW_COV / f"{kind}_{first[:7]}.csv"
            if out.exists():
                continue
            parts = []
            for day in pd.date_range(first, last):
                raw = with_retry(lambda: fetch(auth, path, params(day)))
                parts.append(keep(raw, day) if keep and not raw.empty else raw)
                time.sleep(0.3)
            df = pd.concat(parts, ignore_index=True)
            if df.empty:   # never save an empty month: a saved file counts as finished
                print(f"no {kind} data for {first[:7]}, skipped")
                continue
            df.to_csv(out, index=False)
            print(f"saved {out.name} ({len(df):,} rows)", flush=True)


EXTRA_FILE = PROCESSED / "extra.csv"
ACTUALS = {   # kind -> {API field: our name}; compared with covariates.py's day-ahead forecast of the same thing
    "WINDACT": {"genWest": "wind_west", "genPanhandle": "wind_pan", "genSystemWide": "wind_sys"},
    "SOLARACT": {"genSystemWide": "solar_sys"},
    "LOADACT": {"total": "load_sys", "northC": "load_ncent", "farWest": "load_farwest"},
}
OUTAGES = {"totalResourceMWZoneWest": "out_west", "totalResourceMWZoneNorth": "out_north",
           "totalResourceMWZoneSouth": "out_south", "totalResourceMWZoneHouston": "out_houston",
           "totalIRRMWZoneWest": "out_irr_west", "totalIRRMWZoneNorth": "out_irr_north"}
GTCS = ["WESTEX", "PNHNDL"]   # ERCOT's West Texas and Panhandle export limits: the West->North corridor


def _read(kind, rename=None):
    files = sorted(RAW_COV.glob(f"{kind}_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).rename(columns=rename or {})
    if "DSTFlag" not in df:
        df["DSTFlag"] = False
    return df.assign(hour_ending=to_hour_ending(df))


def build():
    """One row per UTC hour: actuals (for forecast misses), outage forecast, DAM shadow prices."""
    parts = []
    for kind, fields in ACTUALS.items():
        df = _read(kind, {"operatingDay": "deliveryDate"})
        parts.append(df.groupby("hour_ending")[list(fields)].mean().rename(columns=lambda c: fields[c] + "_act"))
    out = _read("OUTAGE", {"operatingDate": "deliveryDate"})
    parts.append(out.groupby("hour_ending")[list(OUTAGES)].mean().rename(columns=OUTAGES))
    sp = _read("SHADOW")
    sp["constraintName"] = sp["constraintName"].astype(str).str.strip()   # the API pads text fields with spaces
    hours = pd.Index(sp["hour_ending"].unique())   # hours with DAM data; a constraint absent there = not binding
    gtc = sp.pivot_table(index="hour_ending", columns="constraintName", values="shadowPrice", aggfunc="sum")
    shadow = pd.DataFrame({
        **{f"sp_{c.lower()}": gtc[c] if c in gtc else 0.0 for c in GTCS},
        "sp_total": sp.groupby("hour_ending")["shadowPrice"].sum(),
        "sp_max": sp.groupby("hour_ending")["shadowPrice"].max(),
        "sp_count": sp.groupby("hour_ending").size(),
    }).reindex(hours).fillna(0.0)
    parts.append(shadow)
    extra = pd.concat(parts, axis=1).sort_index().rename_axis("hour_ending").reset_index()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    extra.to_csv(EXTRA_FILE, index=False)
    print(extra.describe().T[["count", "mean", "min", "max"]].round(1).to_string())


def load_extra():
    df = pd.read_csv(EXTRA_FILE)
    df["hour_ending"] = pd.to_datetime(df["hour_ending"], utc=True)
    return df.set_index("hour_ending")


def probe(day):
    auth, day = _auth(), pd.Timestamp(day)
    for kind, (path, field) in MONTHLY.items():
        raw = fetch(auth, path, {f"{field}From": f"{day:%Y-%m-%d}", f"{field}To": f"{day:%Y-%m-%d}"})
        print(f"\n== {kind}: {len(raw)} rows, columns: {list(raw.columns)}")
    for kind, (path, params, keep) in SOURCES.items():
        raw = fetch(auth, path, params(day))
        print(f"\n== {kind}: {len(raw)} rows, columns: {list(raw.columns)}")
        if not raw.empty:
            kept = keep(raw, day) if keep else raw
            print(f"kept {len(kept)} rows")
            print(kept.head(3).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["probe"]:
        probe(sys.argv[2])
    elif cmd == ["build"]:
        build()
    else:
        download()
