# PTP Strategy v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `docs/superpowers/specs/2026-11-13-ptp-strategy-design.md`: forecast the day-ahead auction's error on ERCOT hub pairs with Chronos-2 (with and without wind/load forecasts) against an hour × month baseline, trade it under the deck's rules, and report honest metrics.

**Architecture:** Small pandas modules in `src/`, one responsibility each, glued together by scripts. The slow steps (downloads, Chronos forecasts) cache results to CSV and resume. All timestamps are UTC internally; `delivery_date` and `hour` are ERCOT local time (America/Chicago). Every forecast model emits the same frame: `pair, hour_ending, delivery_date, q10 … q90`. The final deliverable is one notebook that reads the cached outputs.

**Tech Stack:** Python 3.13, uv, pandas 3, `ercot` 0.1.2, `chronos-forecasting` (Chronos-2, CPU), matplotlib, pytest, Jupyter.

**The spec is the source of truth for every rule.** This plan says how to build it. Read spec §2 (timing) before any task.

---

## File structure

| File | Responsibility |
|---|---|
| `src/config.py` | All constants |
| `src/data_import.py` | Monthly DAM/RTM price download (all hubs) |
| `src/processing.py` | Raw prices → `data/processed/hourly_prices.csv` |
| `src/spreads.py` | Hourly prices → per-pair `da_gap`, `rt_gap`, `error` |
| `src/screen.py` | Rank the 10 hub pairs on 2024 data |
| `src/covariates.py` | Wind/load forecast download (version as of 10:00 on D−1), parsing, calendar features |
| `src/baselines.py` | Hour × month quantile model, zero model |
| `src/chronos_model.py` | Chronos-2 forecasts per delivery day, cached per month |
| `src/strategy.py` | Gate, size, clearing, daily MW cap |
| `src/risk.py`, `src/backtest.py` | Risk multiplier, day-by-day loop |
| `src/metrics.py` | Trading metrics, block bootstrap, forecast quality |
| `src/run_backtest.py` | Every model × period, α tuning, cost sensitivity → `output/` |
| `notebooks/ptp_strategy.ipynb` | Deliverable: the story, charts and tables from `output/` |
| `tests/test_*.py` | One per module |

---

### Task 0: Tooling and config

Branch `backtest-v1` already exists and is checked out.

**Files:** Modify `pyproject.toml`, `.gitignore`. Create `src/config.py`, `tests/__init__.py` (empty).

- [ ] **Step 1: Add dependencies**

```bash
uv add matplotlib chronos-forecasting
uv add --dev pytest jupyter
```

- [ ] **Step 2: Configure pytest.** Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: Append to `.gitignore`**

```
.idea/
data/raw/*/*_hubs_*.csv
data/raw/COV/
data/processed/
output/forecasts/
.ipynb_checkpoints/
```

- [ ] **Step 4: Write `src/config.py`**

```python
"""All project constants. Rules come from docs/superpowers/specs/2026-11-13-ptp-strategy-design.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DAM = ROOT / "data" / "raw" / "DAM"
RAW_RTM = ROOT / "data" / "raw" / "RTM"
RAW_COV = ROOT / "data" / "raw" / "COV"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
FORECASTS = OUTPUT / "forecasts"

HUBS = ["HB_WEST", "HB_NORTH", "HB_SOUTH", "HB_HOUSTON", "HB_PAN"]  # HB_BUSAVG excluded: average of the others

DOWNLOAD_START = "2023-01-01"   # 2023 = model context only
TRAIN_START, TRAIN_END = "2024-01-01", "2024-12-31"
PERIODS = {                      # name -> (start, end), delivery dates inclusive
    "train": (TRAIN_START, TRAIN_END),
    "test_a": ("2025-01-01", "2025-12-04"),
    "test_b": ("2025-12-05", "2026-10-31"),  # after RTC+B went live on 2025-12-05
}
END = PERIODS["test_b"][1]

# Forecast inputs (API field name -> our column)
WIND_FIELDS = {"STWPFWest": "wind_west", "STWPFPanhandle": "wind_pan"}
LOAD_FIELDS = {
    "north": "load_north", "northCentral": "load_ncent", "farWest": "load_farwest",
    "coast": "load_coast", "southCentral": "load_scent", "southern": "load_south",
}

# Models
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{round(q * 100)}" for q in QUANTILES]
CLIP = 500.0             # $/MWh, clip history fed to Chronos-2
CONTEXT_HOURS = 8192     # Chronos-2 max; lower it if Task 8 timing is too slow

# Trading rules
CAPITAL = 1_000_000.0
EDGE_THRESHOLD = 5.0
ALPHAS = [0.6, 0.7, 0.8]
BID_CAP = 25.0
MAX_MW_HOUR = 10.0
MAX_MW_DAY = 40.0
SIGMA_FLOOR = 1.0
COST_PER_MWH = 0.5       # assumed broker fee
COSTS = [0.0, 0.5, 1.0, 2.0]
DAILY_HALF, DAILY_STOP, MAX_DRAWDOWN = 0.02, 0.05, 0.15
SCARCITY_PRICE = 1000.0

# (source, sink). Set after Task 4's pair screen.
PAIRS = [("HB_WEST", "HB_NORTH")]
```

- [ ] **Step 5: Check pytest runs**

Run: `uv run pytest -v`
Expected: `no tests ran` (exit code 5)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/config.py tests/__init__.py CLAUDE.md docs/ src/data_import.py
git commit -m "chore: config, dependencies, test tooling, ERCOT API spec"
```

---

### Task 1: Price download, 2023-01 → 2026-10

**Files:** Rewrite `src/data_import.py`. Test: `tests/test_data_import.py`.

- [ ] **Step 1: Failing test**

```python
from data_import import month_ranges


def test_month_ranges_clips_to_start_and_end():
    assert month_ranges("2024-01-15", "2024-03-10") == [
        ("2024-01-15", "2024-01-31"),
        ("2024-02-01", "2024-02-29"),
        ("2024-03-01", "2024-03-10"),
    ]
```

Run: `uv run pytest tests/test_data_import.py -v` → FAIL (`cannot import name 'month_ranges'`)

- [ ] **Step 2: Rewrite `src/data_import.py`**

```python
"""Download ERCOT DAM and RTM settlement point prices for all hubs, one CSV per month.

Credentials: ERCOT_USERNAME, ERCOT_PASSWORD, ERCOT_SUBSCRIPTION_KEY. Re-running skips finished months.
"""
import os

import ercot as er
import pandas as pd

from config import DOWNLOAD_START, END, HUBS, RAW_DAM, RAW_RTM


def month_ranges(start, end):
    """[(first_day, last_day), ...] per calendar month, clipped to [start, end], as YYYY-MM-DD."""
    out = []
    for m in pd.period_range(start, end, freq="M"):
        a = max(m.start_time, pd.Timestamp(start))
        b = min(m.end_time.normalize(), pd.Timestamp(end))
        out.append((a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
    return out


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
            fetch(HUBS, first, last, format="long").to_csv(path, index=False)
            print(f"saved {path.name}")


if __name__ == "__main__":
    download_all()
```

- [ ] **Step 3:** `uv run pytest tests/test_data_import.py -v` → PASS

- [ ] **Step 4: Run the download** (long; rerun if it fails partway)

Run: `uv run python src/data_import.py`
Expected: 92 `saved` lines (46 months × 2 markets).

- [ ] **Step 5: Verify the format before building on it**

```bash
head -3 data/raw/DAM/DAM_hubs_2024-01.csv data/raw/RTM/RTM_hubs_2024-01.csv
```
Expected:
- DAM columns `date,settlement_point,hour,interval_ending,price`;
- RTM columns `date,settlement_point,interval,interval_ending,price`;
- `interval_ending` **has a UTC offset** (for example `-06:00`).

**If there's no offset, stop:** Task 2 would shift every price by 5–6 hours.

- [ ] **Step 6: Commit**

```bash
git add src/data_import.py tests/test_data_import.py
git commit -m "feat: monthly all-hub DAM/RTM download from 2023"
```

---

### Task 2: Hourly prices per hub

**Files:** Create `src/processing.py`. Test: `tests/test_processing.py`.

- [ ] **Step 1: Failing tests**

```python
import pandas as pd

from processing import build_hourly, rtm_hourly


def _rtm(start_local, periods, prices, point="HB_WEST"):
    ends = pd.date_range(start_local, periods=periods, freq="15min", tz="America/Chicago")
    return pd.DataFrame({"settlement_point": point, "interval_ending": ends.tz_convert("UTC"), "price": prices})


def test_rtm_hourly_averages_four_intervals_into_hour_ending():
    out = rtm_hourly(_rtm("2024-01-01 00:15", 4, [1.0, 2.0, 3.0, 4.0]))
    assert len(out) == 1
    assert out.loc[0, "hour_ending"] == pd.Timestamp("2024-01-01 01:00", tz="America/Chicago")
    assert out.loc[0, "rtm"] == 2.5


def test_dst_fall_back_day_has_25_hours_on_one_delivery_date():
    rtm = _rtm("2025-11-02 00:15", 100, [10.0] * 100)
    hours = rtm["interval_ending"].dt.ceil("h").drop_duplicates().reset_index(drop=True)
    dam = pd.DataFrame({"settlement_point": "HB_WEST", "interval_ending": hours, "price": 20.0})
    out = build_hourly(dam, rtm)
    assert len(out) == 25
    assert (out["delivery_date"] == pd.Timestamp("2025-11-02")).all()
    assert out["hour"].iloc[-1] == 23
    assert (out["hour"] == 1).sum() == 2
    assert (out["dam"] == 20.0).all() and (out["rtm"] == 10.0).all()
```

Run: `uv run pytest tests/test_processing.py -v` → FAIL (`No module named 'processing'`)

- [ ] **Step 2: Write `src/processing.py`**

```python
"""Raw monthly ERCOT price CSVs -> one row per (hour_ending, hub) with DAM and hourly RTM price."""
import pandas as pd

from config import PROCESSED, RAW_DAM, RAW_RTM

HOURLY_FILE = PROCESSED / "hourly_prices.csv"


def local_day_and_hour(hour_ending):
    """ERCOT delivery date and hour (0-23) from a UTC hour-ending series; HE24 belongs to the same day."""
    start = hour_ending.dt.tz_convert("America/Chicago") - pd.Timedelta(hours=1)
    return start.dt.tz_localize(None).dt.normalize(), start.dt.hour


def _read(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["interval_ending"] = pd.to_datetime(df["interval_ending"], utc=True)
    return df


def rtm_hourly(rtm):
    """Average the four 15-minute RT prices into the hour they end in."""
    rtm = rtm.assign(hour_ending=rtm["interval_ending"].dt.ceil("h"))
    out = rtm.groupby(["hour_ending", "settlement_point"], as_index=False)["price"].mean()
    return out.rename(columns={"settlement_point": "point", "price": "rtm"})


def build_hourly(dam, rtm):
    dam = dam.rename(columns={"interval_ending": "hour_ending", "settlement_point": "point", "price": "dam"})
    df = dam[["hour_ending", "point", "dam"]].merge(rtm_hourly(rtm), on=["hour_ending", "point"], how="inner")
    df["delivery_date"], df["hour"] = local_day_and_hour(df["hour_ending"])
    return df.sort_values(["hour_ending", "point"], ignore_index=True)


def load_hourly():
    df = pd.read_csv(HOURLY_FILE, parse_dates=["delivery_date"])
    df["hour_ending"] = pd.to_datetime(df["hour_ending"], utc=True)
    return df


def main():
    hourly = build_hourly(_read(sorted(RAW_DAM.glob("DAM_hubs_*.csv"))), _read(sorted(RAW_RTM.glob("RTM_hubs_*.csv"))))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(HOURLY_FILE, index=False)
    print(hourly.groupby("point").size())
    print(hourly[["dam", "rtm"]].describe())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3:** `uv run pytest tests/test_processing.py -v` → 2 passed

- [ ] **Step 4: Build and sanity-check**

Run: `uv run python src/processing.py`
Expected: about 33,600 rows per hub (1,400 days × 24). More than about 1% missing means a month didn't download fully: delete that month's file and rerun Task 1. The `rtm` max should be in the thousands.

- [ ] **Step 5: Commit**

```bash
git add src/processing.py tests/test_processing.py
git commit -m "feat: hourly DAM/RTM prices per hub, DST-safe"
```

---

### Task 3: Per-pair gaps and the auction error

**Files:** Create `src/spreads.py`. Test: `tests/test_spreads.py`.

- [ ] **Step 1: Failing test**

```python
import pandas as pd

from spreads import build_spreads


def test_gaps_and_error():
    he = pd.Timestamp("2024-01-01 07:00", tz="UTC")
    hourly = pd.DataFrame({
        "hour_ending": [he, he], "delivery_date": [pd.Timestamp("2024-01-01")] * 2, "hour": [0, 0],
        "point": ["HB_WEST", "HB_NORTH"], "dam": [10.0, 30.0], "rtm": [5.0, 50.0],
    })
    row = build_spreads(hourly, [("HB_WEST", "HB_NORTH")]).iloc[0]
    assert row["pair"] == "HB_WEST->HB_NORTH"
    assert (row["da_gap"], row["rt_gap"], row["error"]) == (20.0, 45.0, 25.0)
```

Run: `uv run pytest tests/test_spreads.py -v` → FAIL

- [ ] **Step 2: Write `src/spreads.py`**

```python
"""Hourly hub prices -> per-pair DA gap, RT gap and the auction's error (= PnL per MW, source -> sink)."""
import pandas as pd


def build_spreads(hourly, pairs):
    wide = hourly.pivot_table(index=["hour_ending", "delivery_date", "hour"], columns="point", values=["dam", "rtm"])
    parts = []
    for source, sink in pairs:
        part = pd.DataFrame({
            "da_gap": wide["dam"][sink] - wide["dam"][source],
            "rt_gap": wide["rtm"][sink] - wide["rtm"][source],
        }).reset_index()
        part["pair"] = f"{source}->{sink}"
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).dropna(subset=["da_gap", "rt_gap"])
    out["error"] = out["rt_gap"] - out["da_gap"]
    return out.reset_index(drop=True)
```

- [ ] **Step 3:** `uv run pytest tests/test_spreads.py -v` → PASS

- [ ] **Step 4: Commit**

```bash
git add src/spreads.py tests/test_spreads.py
git commit -m "feat: per-pair gaps and auction error"
```

---

### Task 4: Pair screen, then choose pairs

**Files:** Create `src/screen.py`. Test: `tests/test_screen.py`. Modify `src/config.py` (`PAIRS`).

- [ ] **Step 1: Failing tests**

```python
import pandas as pd
import pytest

from screen import all_pairs, screen


def test_all_pairs():
    assert len(all_pairs(["A", "B", "C", "D", "E"])) == 10


def test_screen_stats():
    s = pd.DataFrame({"pair": ["X"] * 3, "error": [10.0, -10.0, 2.0], "rt_gap": [1.0, -3.0, 2.0]})
    row = screen(s).loc["X"]
    assert row["hours"] == 3
    assert row["mean_abs_error"] == pytest.approx(22 / 3)
    assert row["share_abs_error_gt_5"] == pytest.approx(2 / 3)
    assert row["mean_abs_rt_gap"] == pytest.approx(2.0)
```

Run: `uv run pytest tests/test_screen.py -v` → FAIL

- [ ] **Step 2: Write `src/screen.py`**

```python
"""Rank every hub pair on 2024: how big and how frequent is the auction's error?"""
from itertools import combinations

import pandas as pd

from config import HUBS, OUTPUT, TRAIN_END, TRAIN_START
from processing import load_hourly
from spreads import build_spreads


def all_pairs(hubs):
    return list(combinations(hubs, 2))  # unordered: the strategy picks the direction


def screen(spreads):
    g = spreads.groupby("pair")
    table = pd.DataFrame({
        "hours": g.size(),
        "mean_error": g["error"].mean(),
        "std_error": g["error"].std(),
        "mean_abs_error": g["error"].apply(lambda x: x.abs().mean()),
        "share_abs_error_gt_5": g["error"].apply(lambda x: (x.abs() > 5).mean()),
        "mean_abs_rt_gap": g["rt_gap"].apply(lambda x: x.abs().mean()),
    })
    return table.sort_values("share_abs_error_gt_5", ascending=False)


def main():
    s = build_spreads(load_hourly(), all_pairs(HUBS))
    table = screen(s[s["delivery_date"].between(pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END))])
    OUTPUT.mkdir(exist_ok=True)
    table.to_csv(OUTPUT / "pair_screen.csv")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3:** `uv run pytest tests/test_screen.py -v` → 2 passed

- [ ] **Step 4: Run it.** `uv run python src/screen.py` writes a 10-row table to `output/pair_screen.csv`.

- [ ] **Step 5: Team decision.** Pick 1–3 pairs, at most one pair per hub (spec §3), preferring a high `share_abs_error_gt_5`. Keep West→North unless it ranks clearly worse. Use 2024 only. Edit `PAIRS` in `src/config.py`, for example:

```python
PAIRS = [("HB_WEST", "HB_NORTH"), ("HB_PAN", "HB_SOUTH")]
```

- [ ] **Step 6: Commit**

```bash
git add src/screen.py tests/test_screen.py src/config.py output/pair_screen.csv
git commit -m "feat: pair screen on 2024; choose traded pairs"
```

---

### Task 5: Wind and load forecast download

**Files:** Create `src/covariates.py` (download part). Test: `tests/test_covariates.py`.

- [ ] **Step 1: Failing test for the version rule**

```python
import pandas as pd

from covariates import pick_vintage


def _raw(posted):
    return pd.DataFrame({"postedDatetime": posted, "hourEnding": ["01:00"] * len(posted), "v": range(len(posted))})


def test_picks_latest_version_before_10am_day_before():
    raw = _raw(["2024-06-01T08:15:00", "2024-06-01T09:15:00", "2024-06-01T10:15:00"])
    out = pick_vintage(raw, "2024-06-02")
    assert out["v"].tolist() == [1]          # 09:15 on D-1; the 10:15 version is too late


def test_falls_back_to_older_version_and_never_to_a_later_one():
    assert pick_vintage(_raw(["2024-05-31T23:00:00"]), "2024-06-02")["v"].tolist() == [0]
    assert pick_vintage(_raw(["2024-06-01T10:00:00"]), "2024-06-02").empty
```

Run: `uv run pytest tests/test_covariates.py -v` → FAIL

- [ ] **Step 2: Write the download part of `src/covariates.py`**

```python
"""Wind and load forecasts as they stood at 10:00 the day before delivery, plus calendar features.

The `ercot` package doesn't expose the publication-time filter, so we call the API directly,
reusing its ErcotAuth (handles the 1-hour token refresh).
"""
import os
import sys
import time

import pandas as pd
import requests
from ercot.auth import ErcotAuth
from ercot.models import BASE_URL

from config import DOWNLOAD_START, END, RAW_COV
from data_import import month_ranges

ENDPOINTS = {
    "WIND": "/np4-742-cd/wpp_hrly_actual_fcast_geo",
    "LOAD": "/np3-565-cd/lf_by_model_weather_zone",
}
CUTOFF = pd.Timedelta(hours=14)    # delivery day 00:00 minus 14h = 10:00 on D-1
LOOKBACK = pd.Timedelta(hours=48)


def _auth():
    return ErcotAuth(os.environ["ERCOT_USERNAME"], os.environ["ERCOT_PASSWORD"], os.environ["ERCOT_SUBSCRIPTION_KEY"])


def fetch(auth, path, params):
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


def pick_vintage(df, delivery_date):
    """Rows of the latest forecast published before 10:00 on the day before delivery_date (local time)."""
    cutoff = pd.Timestamp(delivery_date) - CUTOFF
    posted = pd.to_datetime(df["postedDatetime"])
    ok = posted < cutoff
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
                raw = fetch(auth, path, _params(day))
                if name == "LOAD" and not raw.empty:
                    raw = raw[_in_use(raw)]
                parts.append(pick_vintage(raw, day))
                time.sleep(0.3)
            pd.concat(parts, ignore_index=True).to_csv(out, index=False)
            print(f"saved {out.name}")


def probe(day):
    """Print what the API returns for one delivery day, to check field names and formats."""
    auth = _auth()
    for name, path in ENDPOINTS.items():
        raw = fetch(auth, path, _params(pd.Timestamp(day)))
        print(f"== {name}: {len(raw)} rows\n{raw.dtypes}\n{raw.head(3).to_string()}")


if __name__ == "__main__":
    probe(sys.argv[2]) if sys.argv[1:2] == ["probe"] else download_covariates()
```

- [ ] **Step 3:** `uv run pytest tests/test_covariates.py -v` → 2 passed

- [ ] **Step 4: Probe the real API before the long download**

Run: `uv run python src/covariates.py probe 2024-06-02`
Check every item below. If one is wrong, fix `covariates.py` and `config.py` before continuing:
- WIND has columns `postedDatetime`, `deliveryDate`, `hourEnding`, `DSTFlag`, `STWPFWest`, `STWPFPanhandle` (these names are inferred from the API's filter names).
- LOAD has `postedDatetime`, `deliveryDate`, `hourEnding`, `DSTFlag`, `inUseFlag`, `model`, and the six zones in `LOAD_FIELDS`.
- `postedDatetime` is local time without an offset, and every value is before `2024-06-01T10:00:00`.
- `inUseFlag` values match `_in_use`, and exactly one `model` has it set.
- `hourEnding` looks like `01:00`…`24:00`.

- [ ] **Step 5: Run the download** (about 2,800 requests, a few hours; resumable)

Run: `uv run python src/covariates.py`
Expected: 92 `saved` lines in `data/raw/COV/`.

- [ ] **Step 6: Commit**

```bash
git add src/covariates.py tests/test_covariates.py
git commit -m "feat: wind/load forecast download, version as of 10:00 the day before"
```

---

### Task 6: Covariate table and calendar features

**Files:** Modify `src/covariates.py`. Modify `tests/test_covariates.py`.

- [ ] **Step 1: Failing tests** (append to `tests/test_covariates.py`)

```python
from covariates import calendar, to_hour_ending


def test_hour_ending_handles_he24_and_repeated_dst_hour():
    df = pd.DataFrame({
        "deliveryDate": ["2024-01-01", "2025-11-02", "2025-11-02"],
        "hourEnding": ["24:00", "02:00", "02:00"],
        "DSTFlag": ["N", "N", "Y"],
    })
    out = to_hour_ending(df).tolist()
    assert out[0] == pd.Timestamp("2024-01-02 06:00", tz="UTC")
    assert out[1] == pd.Timestamp("2025-11-02 07:00", tz="UTC")   # first 01:00-02:00 (CDT)
    assert out[2] == pd.Timestamp("2025-11-02 08:00", tz="UTC")   # repeated hour (CST)


def test_calendar_marks_local_hour_and_holiday():
    he = pd.Series(pd.to_datetime(["2024-07-04 06:00", "2024-07-05 06:00"], utc=True))  # HE1 local
    cal = calendar(he)
    assert cal["hour"].tolist() == [0, 0]
    assert cal["holiday"].tolist() == [1, 0]
    assert cal["dow"].tolist() == [3, 4]
```

Run: `uv run pytest tests/test_covariates.py -v` → FAIL (`cannot import name 'calendar'`)

- [ ] **Step 2: Add to `src/covariates.py`** (new imports at the top, functions above `download_covariates`)

```python
from pandas.tseries.holiday import USFederalHolidayCalendar

from config import LOAD_FIELDS, PROCESSED, WIND_FIELDS

COVARIATE_FILE = PROCESSED / "covariates.csv"


def to_hour_ending(df):
    """ERCOT deliveryDate + hourEnding ('HH:00', 1-24) + DSTFlag -> UTC hour ending.

    Localize the hour *beginning* (always a real wall-clock time), then add 1h.
    DSTFlag 'Y' marks the repeated fall-back hour, which is standard time.
    """
    he = df["hourEnding"].astype(str).str.split(":").str[0].astype(int)
    begin = pd.to_datetime(df["deliveryDate"]) + pd.to_timedelta(he - 1, unit="h")
    repeated = df["DSTFlag"].astype(str).str.upper().isin(["Y", "TRUE"]).to_numpy()
    local = begin.dt.tz_localize("America/Chicago", ambiguous=~repeated)
    return (local + pd.Timedelta(hours=1)).dt.tz_convert("UTC")


def calendar(hour_ending):
    start = hour_ending.dt.tz_convert("America/Chicago") - pd.Timedelta(hours=1)
    day = start.dt.tz_localize(None).dt.normalize()
    holidays = USFederalHolidayCalendar().holidays(day.min(), day.max())
    return pd.DataFrame({
        "hour": start.dt.hour, "dow": start.dt.dayofweek, "month": start.dt.month,
        "holiday": day.isin(holidays).astype(int),
    }, index=hour_ending.index)


def _hourly(raw, fields):
    out = raw.assign(hour_ending=to_hour_ending(raw)).rename(columns=fields)
    cols = list(fields.values())
    out[cols] = out[cols].apply(pd.to_numeric, errors="coerce")
    return out.groupby("hour_ending", as_index=False)[cols].mean()


def build_covariates():
    wind = pd.concat([pd.read_csv(f) for f in sorted(RAW_COV.glob("WIND_*.csv"))], ignore_index=True)
    load = pd.concat([pd.read_csv(f) for f in sorted(RAW_COV.glob("LOAD_*.csv"))], ignore_index=True)
    cov = _hourly(wind, WIND_FIELDS).merge(_hourly(load, LOAD_FIELDS), on="hour_ending", how="outer")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cov.sort_values("hour_ending").to_csv(COVARIATE_FILE, index=False)
    print(cov.describe().to_string())
    print("hours with any missing value:", int(cov.isna().any(axis=1).sum()))


def load_covariates():
    cov = pd.read_csv(COVARIATE_FILE)
    cov["hour_ending"] = pd.to_datetime(cov["hour_ending"], utc=True)
    return cov
```

Change the `__main__` block to:

```python
if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["probe"]:
        probe(sys.argv[2])
    elif cmd == ["build"]:
        build_covariates()
    else:
        download_covariates()
```

- [ ] **Step 3:** `uv run pytest tests/test_covariates.py -v` → 4 passed

- [ ] **Step 4: Build.** `uv run python src/covariates.py build`. Expect wind values in MW (thousands for West) and load values in MW. If more than ~2% of hours have missing values, look at which days are empty before going on.

- [ ] **Step 5: Commit**

```bash
git add src/covariates.py tests/test_covariates.py
git commit -m "feat: hourly covariate table and calendar features"
```

---

### Task 7: Baseline models

**Files:** Create `src/baselines.py`. Test: `tests/test_baselines.py`.

- [ ] **Step 1: Failing test**

```python
import pandas as pd

from baselines import hour_month_forecast, hour_month_table, zero_forecast
from config import QCOLS


def _err(dates, errors):
    d = pd.to_datetime(dates)
    return pd.DataFrame({"pair": "A->B", "hour": 0, "delivery_date": d,
                         "hour_ending": d.tz_localize("UTC") + pd.Timedelta(hours=6), "error": errors})


def test_hour_month_uses_training_rows_only_and_maps_by_month():
    train = _err(["2024-01-01", "2024-01-02", "2024-01-03", "2025-01-01"], [0.0, 10.0, 20.0, 999.0])
    table = hour_month_table(train, "2024-01-01", "2024-12-31")
    fc = hour_month_forecast(_err(["2025-01-15"], [0.0]), table)
    assert fc.loc[0, "q50"] == 10.0
    assert list(fc.columns) == ["pair", "hour_ending", "delivery_date", *QCOLS]


def test_zero_forecast_is_all_zero():
    fc = zero_forecast(_err(["2025-01-15"], [3.0]))
    assert (fc[QCOLS] == 0.0).all().all()
```

Run: `uv run pytest tests/test_baselines.py -v` → FAIL

- [ ] **Step 2: Write `src/baselines.py`**

```python
"""Reference models with the same output shape as Chronos-2: pair, hour_ending, delivery_date, q10..q90."""
import pandas as pd

from config import QCOLS, QUANTILES

KEYS = ["pair", "hour_ending", "delivery_date"]


def hour_month_table(err, start, end):
    t = err[err["delivery_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    t = t.assign(month=t["delivery_date"].dt.month)
    q = t.groupby(["pair", "hour", "month"])["error"].quantile(QUANTILES).unstack()
    q.columns = QCOLS
    return q.reset_index()


def hour_month_forecast(err, table):
    fc = err.assign(month=err["delivery_date"].dt.month).merge(table, on=["pair", "hour", "month"], how="left")
    return fc[KEYS + QCOLS]


def zero_forecast(err):
    """'The auction is right.' Reference for forecast accuracy only."""
    return err[KEYS].assign(**{c: 0.0 for c in QCOLS})
```

- [ ] **Step 3:** `uv run pytest tests/test_baselines.py -v` → 2 passed

- [ ] **Step 4: Commit**

```bash
git add src/baselines.py tests/test_baselines.py
git commit -m "feat: hour x month and zero baseline models"
```

---

### Task 8: Chronos-2 forecasts

**Files:** Create `src/chronos_model.py`. Test: `tests/test_chronos_model.py` (uses a fake pipeline, so no model download in tests).

- [ ] **Step 1: Failing tests**

```python
import pandas as pd

from chronos_model import build_frame, forecast_day, origin
from config import QCOLS, QUANTILES


class FakePipeline:
    def predict_df(self, ctx, future_df, prediction_length, quantile_levels, id_column, timestamp_column, target):
        self.ctx, self.fut, self.h = ctx, future_df, prediction_length
        out = future_df[[id_column, timestamp_column]].copy()
        for q in quantile_levels:
            out[str(q)] = 1.0
        return out


def _err(days=10):
    he = pd.date_range("2024-03-01 07:00", periods=24 * days, freq="h", tz="UTC")  # HE1 local on 1 March
    start = he.tz_convert("America/Chicago") - pd.Timedelta(hours=1)
    return pd.DataFrame({"pair": "A->B", "hour_ending": he, "delivery_date": start.tz_localize(None).normalize(),
                         "error": range(len(he))})


def _cov(err, drop_day=None):
    cov = pd.DataFrame({"hour_ending": err["hour_ending"], "wind_west": 1.0})
    if drop_day is not None:
        cov.loc[err["delivery_date"] == pd.Timestamp(drop_day), "wind_west"] = None
    return cov


def test_context_stops_at_end_of_t_minus_2_and_forecast_covers_day_t():
    err = _err()
    p = FakePipeline()
    fc = forecast_day(p, build_frame(err, _cov(err)), pd.Timestamp("2024-03-08"), use_covariates=True)
    assert p.ctx["timestamp"].max() <= origin(pd.Timestamp("2024-03-08")).tz_localize(None)
    assert p.h == 48
    assert (fc["delivery_date"] == pd.Timestamp("2024-03-08")).all() and len(fc) == 24
    assert list(fc.columns) == ["pair", "hour_ending", "delivery_date", *QCOLS]


def test_missing_covariates_for_day_t_means_no_forecast():
    err = _err()
    assert forecast_day(FakePipeline(), build_frame(err, _cov(err, "2024-03-08")),
                        pd.Timestamp("2024-03-08"), use_covariates=True) is None


def test_history_is_clipped():
    err = _err().assign(error=1e6)
    frame = build_frame(err, _cov(err))
    assert frame["target"].max() == 500.0
```

Run: `uv run pytest tests/test_chronos_model.py -v` → FAIL

- [ ] **Step 2: Write `src/chronos_model.py`**

```python
"""Chronos-2 zero-shot forecasts of the auction error, one call per delivery day T, cached per month.

Context: error history through the end of T-2 (last fully settled real-time day at 10:00 on T-1).
Future covariates: calendar always; wind/load as published before 10:00 the day before each hour's day.
"""
import sys
import time

import pandas as pd

from config import CLIP, CONTEXT_HOURS, END, FORECASTS, PAIRS, QCOLS, QUANTILES, TRAIN_START
from covariates import calendar, load_covariates
from data_import import month_ranges
from processing import load_hourly, local_day_and_hour
from spreads import build_spreads

VARIANTS = {"chronos_cov": True, "chronos_nocov": False}
QMAP = {str(q): c for q, c in zip(QUANTILES, QCOLS)}


def origin(T):
    """UTC end of delivery day T-2 = local midnight starting T-1."""
    return (T - pd.Timedelta(days=1)).tz_localize("America/Chicago").tz_convert("UTC")


def build_frame(err, cov, use_covariates=True):
    """Regular hourly UTC grid per pair: clipped target, calendar, optional wind/load."""
    grid = pd.date_range(err["hour_ending"].min(), err["hour_ending"].max(), freq="h")
    parts = []
    for pair, g in err.groupby("pair"):
        s = g.set_index("hour_ending")["error"].reindex(grid)
        parts.append(pd.DataFrame({"id": pair, "timestamp": grid,
                                   "target": s.clip(-CLIP, CLIP).ffill().fillna(0.0).to_numpy()}))
    f = pd.concat(parts, ignore_index=True)
    f = pd.concat([f, calendar(f["timestamp"])], axis=1)
    if use_covariates:
        f = f.merge(cov, left_on="timestamp", right_on="hour_ending", how="left").drop(columns="hour_ending")
    return f


def forecast_day(pipeline, frame, T, use_covariates):
    o = origin(T)
    end_of_t = (T + pd.Timedelta(days=1)).tz_localize("America/Chicago").tz_convert("UTC")
    horizon = int((end_of_t - o) / pd.Timedelta(hours=1))   # 47, 48 or 49 on DST days
    cov_cols = [c for c in frame.columns if c not in ("id", "timestamp", "target")]
    ctx = frame[frame["timestamp"] <= o].groupby("id").tail(CONTEXT_HOURS).copy()
    fut = frame[(frame["timestamp"] > o) & (frame["timestamp"] <= end_of_t)].drop(columns="target")
    if use_covariates and fut[cov_cols].isna().any().any():
        return None  # spec 4.2: missing forecasts -> no trades that day
    ctx[cov_cols] = ctx.groupby("id")[cov_cols].ffill().bfill()
    for df in (ctx, fut):
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    pred = pipeline.predict_df(ctx, future_df=fut, prediction_length=horizon, quantile_levels=QUANTILES,
                               id_column="id", timestamp_column="timestamp", target="target")
    pred = pred.rename(columns={"id": "pair", **QMAP})
    pred["hour_ending"] = pd.to_datetime(pred["timestamp"]).dt.tz_localize("UTC")
    pred["delivery_date"], _ = local_day_and_hour(pred["hour_ending"])
    return pred.loc[pred["delivery_date"] == T, ["pair", "hour_ending", "delivery_date", *QCOLS]].reset_index(drop=True)


def run(variant):
    from chronos import Chronos2Pipeline
    use_cov = VARIANTS[variant]
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    err = build_spreads(load_hourly(), PAIRS)
    frame = build_frame(err, load_covariates() if use_cov else None, use_cov)
    FORECASTS.mkdir(parents=True, exist_ok=True)
    for first, last in month_ranges(TRAIN_START, END):
        out = FORECASTS / f"{variant}_{first[:7]}.csv"
        if out.exists():
            continue
        t0 = time.time()
        days = [forecast_day(pipeline, frame, T, use_cov) for T in pd.date_range(first, last)]
        pd.concat([d for d in days if d is not None], ignore_index=True).to_csv(out, index=False)
        print(f"saved {out.name} ({time.time() - t0:.0f}s, {sum(d is None for d in days)} days skipped)")


def load_forecasts(variant):
    fc = pd.concat([pd.read_csv(f, parse_dates=["delivery_date"]) for f in sorted(FORECASTS.glob(f"{variant}_*.csv"))],
                   ignore_index=True)
    fc["hour_ending"] = pd.to_datetime(fc["hour_ending"], utc=True)
    return fc


if __name__ == "__main__":
    run(sys.argv[1])
```

- [ ] **Step 3:** `uv run pytest tests/test_chronos_model.py -v` → 3 passed

- [ ] **Step 4: Time one month first**

Run: `uv run python src/chronos_model.py chronos_cov` and stop it (Ctrl-C) after the first `saved` line.
Expected: `saved chronos_cov_2024-01.csv (Ns, ...)`.
- If N is over ~600 s (20 s/day), set `CONTEXT_HOURS = 2048` in `config.py`, delete the file and retry.
- Open the CSV: q10 ≤ q50 ≤ q90 on almost every row, and values in a plausible $ range.

- [ ] **Step 5: Run both variants fully** (resumable; about 34 months each)

```bash
uv run python src/chronos_model.py chronos_cov
uv run python src/chronos_model.py chronos_nocov
```

- [ ] **Step 6: Commit** (the forecast CSVs are gitignored)

```bash
git add src/chronos_model.py tests/test_chronos_model.py src/config.py
git commit -m "feat: Chronos-2 zero-shot forecasts per delivery day, cached"
```

---

### Task 9: Gate, size, clearing, daily cap

**Files:** Create `src/strategy.py`. Test: `tests/test_strategy.py`.

- [ ] **Step 1: Failing tests**

```python
import pandas as pd

from config import QCOLS
from strategy import apply_daily_cap, clear, gate_and_size

ARGS = dict(capital=1_000_000.0, edge_threshold=5.0, max_mw_hour=10.0, sigma_floor=1.0)


def _fc(qs, da_gap=0.0):
    row = dict(zip(QCOLS, qs))
    return pd.DataFrame([{"pair": "A->B", "delivery_date": pd.Timestamp("2025-01-01"), "da_gap": da_gap, **row}])


def test_long_when_q30_above_threshold():
    b = gate_and_size(_fc([0, 3, 6, 8, 10, 12, 14, 16, 20]), alpha=0.7, **ARGS).iloc[0]
    assert b["direction"] == 1 and b["mw_wanted"] == 10.0


def test_short_when_q70_below_minus_threshold():
    b = gate_and_size(_fc([-20, -16, -14, -12, -10, -8, -6, -3, 0]), alpha=0.7, **ARGS).iloc[0]
    assert b["direction"] == -1


def test_no_trade_when_range_straddles_threshold():
    assert gate_and_size(_fc([-10, -6, -3, 0, 2, 4, 6, 8, 12]), alpha=0.7, **ARGS).iloc[0]["direction"] == 0


def test_alpha_selects_quantile():
    qs = [0, 3, 4, 6, 10, 12, 14, 16, 20]          # q30=4 fails, q40=6 passes
    assert gate_and_size(_fc(qs), alpha=0.7, **ARGS).iloc[0]["direction"] == 0
    assert gate_and_size(_fc(qs), alpha=0.6, **ARGS).iloc[0]["direction"] == 1


def test_half_kelly_with_sigma_floor_and_wide_range():
    b = gate_and_size(_fc([6] * 9), alpha=0.7, **ARGS).iloc[0]                       # sigma -> floor $1
    assert b["mw_wanted"] == 10.0
    wide = gate_and_size(_fc([-1e5, 6, 6, 6, 10, 6, 6, 6, 1e5]), alpha=0.7, **ARGS).iloc[0]
    sigma = 2e5 / 2.563
    assert abs(wide["mw_wanted"] - 0.5 * 1e6 * 10 / sigma**2) < 1e-9


def test_clearing_respects_bid_cap_on_our_path():
    b = gate_and_size(_fc([0, 3, 6, 8, 10, 12, 14, 16, 20], da_gap=24.0), alpha=0.7, **ARGS)
    assert clear(b, 25.0).iloc[0]["cleared"]
    b = gate_and_size(_fc([0, 3, 6, 8, 10, 12, 14, 16, 20], da_gap=26.0), alpha=0.7, **ARGS)
    assert not clear(b, 25.0).iloc[0]["cleared"]


def test_daily_cap_spends_budget_on_largest_rank():
    bids = pd.DataFrame({"delivery_date": pd.Timestamp("2025-01-01"),
                         "rank": [6.0, 11.0, 7.0, 10.0, 8.0, 9.0], "mw_wanted": 10.0})
    out = apply_daily_cap(bids, 40.0)
    assert out["mw"].sum() == 40.0
    assert list(out.loc[out["mw"] == 0, "rank"]) == [6.0, 7.0]
```

Run: `uv run pytest tests/test_strategy.py -v` → FAIL

- [ ] **Step 2: Write `src/strategy.py`**

```python
"""Spec section 6: forecast quantiles -> direction, size, clearing, daily MW budget."""
import numpy as np


def gate_and_size(fc, alpha, capital, edge_threshold, max_mw_hour, sigma_floor):
    lo, hi = f"q{round((1 - alpha) * 100)}", f"q{round(alpha * 100)}"
    out = fc.copy()
    out["direction"] = np.select([out[lo] > edge_threshold, out[hi] < -edge_threshold], [1, -1], 0)
    sigma = np.maximum((out["q90"] - out["q10"]) / 2.563, sigma_floor)
    half_kelly = 0.5 * capital * out["q50"].abs() / sigma**2
    out["mw_wanted"] = np.where(out["direction"] != 0, np.minimum(max_mw_hour, half_kelly), 0.0)
    out["rank"] = out["q50"].abs()
    return out


def clear(bids, bid_cap):
    """The auction clears our bid if the price on our path is at most the cap."""
    out = bids.copy()
    out["cleared"] = (out["direction"] != 0) & (out["direction"] * out["da_gap"] <= bid_cap)
    return out


def apply_daily_cap(bids, max_mw_day):
    """Spend each delivery day's MW budget on the highest-ranked bids first (decided before the auction)."""
    b = bids.sort_values(["delivery_date", "rank"], ascending=[True, False])
    used_before = b.groupby("delivery_date")["mw_wanted"].cumsum() - b["mw_wanted"]
    b["mw"] = np.minimum(b["mw_wanted"], (max_mw_day - used_before).clip(lower=0))
    return b.sort_index()
```

- [ ] **Step 3:** `uv run pytest tests/test_strategy.py -v` → 7 passed

- [ ] **Step 4: Commit**

```bash
git add src/strategy.py tests/test_strategy.py
git commit -m "feat: quantile gate, half-Kelly size, bid cap clearing, daily MW cap"
```

---

### Task 10: Risk rules and backtest loop

**Files:** Create `src/risk.py`, `src/backtest.py`. Tests: `tests/test_risk.py`, `tests/test_backtest.py`.

- [ ] **Step 1: Failing tests**

`tests/test_risk.py`:

```python
from risk import risk_multiplier

ARGS = dict(daily_half=0.02, daily_stop=0.05)


def test_no_loss_full_size():
    assert risk_multiplier(1000.0, 1e6, scarcity=False, stopped=False, **ARGS) == 1.0


def test_daily_stop_levels():
    assert risk_multiplier(-30_000.0, 1e6, scarcity=False, stopped=False, **ARGS) == 0.5
    assert risk_multiplier(-60_000.0, 1e6, scarcity=False, stopped=False, **ARGS) == 0.0


def test_scarcity_stacks():
    assert risk_multiplier(0.0, 1e6, scarcity=True, stopped=False, **ARGS) == 0.5
    assert risk_multiplier(-30_000.0, 1e6, scarcity=True, stopped=False, **ARGS) == 0.25


def test_strategy_stop_wins():
    assert risk_multiplier(5000.0, 1e6, scarcity=False, stopped=True, **ARGS) == 0.0
```

`tests/test_backtest.py`:

```python
import pandas as pd
import pytest

from backtest import run_backtest

ARGS = dict(capital=1_000_000.0, cost_per_mwh=0.5, daily_half=0.02, daily_stop=0.05, max_drawdown=0.15)


def _bids(errors):
    return pd.DataFrame({
        "delivery_date": pd.date_range("2025-01-01", periods=len(errors), freq="D"),
        "hour": 0, "pair": "A->B", "direction": 1, "cleared": True, "mw": 10.0,
        "da_gap": 0.0, "rt_gap": errors, "error": errors,
    })


def test_daily_stop_reacts_to_t_minus_2():
    daily, log = run_backtest(_bids([-2999.5] * 3), scarcity_dates=set(), **ARGS)
    assert list(daily["multiplier"]) == [1.0, 1.0, 0.5]
    assert list(daily["pnl"]) == [-30_000.0, -30_000.0, -15_000.0]
    assert log["mw"].tolist() == [10.0, 10.0, 5.0]


def test_drawdown_stop_is_permanent():
    daily, _ = run_backtest(_bids([-19_999.5, 1.5, 1.5, 1.5]), scarcity_dates=set(), **ARGS)
    assert list(daily["multiplier"]) == [1.0, 1.0, 0.0, 0.0]
    assert daily["drawdown"].max() == pytest.approx(0.2)


def test_scarcity_on_t_minus_2_halves_size():
    daily, _ = run_backtest(_bids([10.5] * 3), scarcity_dates={pd.Timestamp("2025-01-01")}, **ARGS)
    assert list(daily["multiplier"]) == [1.0, 1.0, 0.5]
```

Run: `uv run pytest tests/test_risk.py tests/test_backtest.py -v` → FAIL

- [ ] **Step 2: Write `src/risk.py`**

```python
"""Spec section 7: size multiplier from information settled by bid time."""


def risk_multiplier(last_settled_pnl, equity, scarcity, stopped, daily_half, daily_stop):
    if stopped:
        return 0.0
    loss = -last_settled_pnl / equity
    m = 0.0 if loss > daily_stop else 0.5 if loss > daily_half else 1.0
    return m * 0.5 if scarcity else m
```

- [ ] **Step 3: Write `src/backtest.py`**

```python
"""Day-by-day backtest: risk rules only see PnL settled by bid time (delivery day T-2)."""
import numpy as np
import pandas as pd

from risk import risk_multiplier

TWO_DAYS = pd.Timedelta(days=2)


def run_backtest(bids, capital, cost_per_mwh, scarcity_dates, daily_half, daily_stop, max_drawdown):
    b = bids.copy()
    b["unit_pnl"] = b["direction"] * b["error"] - cost_per_mwh
    b["pnl_full"] = np.where(b["cleared"], b["mw"] * b["unit_pnl"], 0.0)
    full = b.groupby("delivery_date")["pnl_full"].sum()
    days = pd.date_range(full.index.min(), full.index.max(), freq="D")
    full = full.reindex(days, fill_value=0.0)

    pnl, mult = {}, {}
    equity = peak = capital
    stopped = False
    for d in days:
        settled = d - TWO_DAYS
        last = pnl.get(settled, 0.0)
        if settled in pnl:
            equity += last
            peak = max(peak, equity)
        stopped = stopped or (peak - equity) / peak > max_drawdown
        mult[d] = risk_multiplier(last, equity, settled in scarcity_dates, stopped, daily_half, daily_stop)
        pnl[d] = mult[d] * full[d]

    daily = pd.DataFrame({"pnl": pd.Series(pnl), "multiplier": pd.Series(mult)})
    daily["equity"] = capital + daily["pnl"].cumsum()
    daily["drawdown"] = 1 - daily["equity"] / daily["equity"].cummax().clip(lower=capital)

    b["mw"] = b["mw"] * b["delivery_date"].map(mult)
    log = b[b["cleared"] & (b["mw"] > 0)].copy()
    log["price_paid"] = log["direction"] * log["da_gap"]
    log["price_received"] = log["direction"] * log["rt_gap"]
    log["pnl"] = log["mw"] * log["unit_pnl"]
    log = log[["delivery_date", "hour", "pair", "direction", "mw", "price_paid", "price_received", "pnl"]]
    return daily, log.sort_values(["delivery_date", "hour", "pair"], ignore_index=True)
```

- [ ] **Step 4:** `uv run pytest tests/test_risk.py tests/test_backtest.py -v` → 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/risk.py src/backtest.py tests/test_risk.py tests/test_backtest.py
git commit -m "feat: daily stop, strategy stop, scarcity filter, backtest loop"
```

---

### Task 11: Metrics, bootstrap, forecast quality

**Files:** Create `src/metrics.py`. Test: `tests/test_metrics.py`.

- [ ] **Step 1: Failing tests**

```python
import math

import pandas as pd
import pytest

from config import QCOLS
from metrics import block_bootstrap, forecast_quality, summarize


def test_summary_numbers():
    daily = pd.DataFrame({"pnl": [100.0, -50.0, 50.0], "drawdown": [0.0, 0.05, 0.0]})
    log = pd.DataFrame({"pnl": [100.0, -50.0, 30.0, 20.0]})
    s = summarize(daily, log, capital=1000.0)
    assert (s["total_pnl"], s["return_pct"], s["trades"], s["win_rate"]) == (100.0, 10.0, 4, 0.75)
    assert (s["avg_win"], s["avg_loss"], s["win_loss_ratio"]) == (50.0, -50.0, 1.0)
    assert s["max_drawdown_pct"] == pytest.approx(5.0)
    assert s["sharpe"] == pytest.approx(daily["pnl"].mean() / daily["pnl"].std() * math.sqrt(365))


def test_no_trades_gives_nan():
    s = summarize(pd.DataFrame({"pnl": [0.0, 0.0], "drawdown": [0.0, 0.0]}), pd.DataFrame({"pnl": []}), 1000.0)
    assert s["trades"] == 0 and math.isnan(s["win_rate"]) and math.isnan(s["sharpe"])


def test_bootstrap_ci_contains_zero_for_noise_and_not_for_constant_gain():
    noise = block_bootstrap([1.0, -1.0] * 50)
    assert noise["mean_daily_pnl_ci"][0] < 0 < noise["mean_daily_pnl_ci"][1]
    assert block_bootstrap([1.0] * 100)["mean_daily_pnl_ci"] == [1.0, 1.0]


def test_forecast_quality():
    he = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    fc = pd.DataFrame({"pair": "A->B", "hour_ending": he, **{c: [10.0, 0.0] for c in QCOLS}})
    fc["q10"], fc["q90"] = [0.0, -1.0], [20.0, 1.0]
    err = pd.DataFrame({"pair": "A->B", "hour_ending": he, "error": [12.0, 5.0]})
    q = forecast_quality(fc, err, alpha=0.7, edge_threshold=5.0)
    assert q["mae"] == 3.5
    assert q["coverage_10_90"] == 0.5
    assert (q["hours_traded"], q["hit_rate_traded"]) == (1, 1.0)
```

Run: `uv run pytest tests/test_metrics.py -v` → FAIL

- [ ] **Step 2: Write `src/metrics.py`**

```python
"""Spec section 8: trading metrics, block-bootstrap confidence intervals, forecast quality."""
import math

import numpy as np


def summarize(daily, log, capital):
    wins = log.loc[log["pnl"] > 0, "pnl"]
    losses = log.loc[log["pnl"] < 0, "pnl"]
    std = daily["pnl"].std()
    return {
        "total_pnl": float(daily["pnl"].sum()),
        "return_pct": float(daily["pnl"].sum() / capital * 100),
        "trades": int(len(log)),
        "win_rate": float((log["pnl"] > 0).mean()) if len(log) else math.nan,
        "avg_win": float(wins.mean()) if len(wins) else math.nan,
        "avg_loss": float(losses.mean()) if len(losses) else math.nan,
        "win_loss_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else math.nan,
        "max_drawdown_pct": float(daily["drawdown"].max() * 100),
        "sharpe": float(daily["pnl"].mean() / std * math.sqrt(365)) if std > 0 else math.nan,
    }


def block_bootstrap(daily_pnl, block=7, n=2000, seed=0):
    """95% CIs for mean daily PnL and Sharpe; 7-day blocks keep the day-to-day correlation."""
    x = np.asarray(daily_pnl, dtype=float)
    m = len(x)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, m - block + 1, size=(n, math.ceil(m / block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :m]
    s = x[idx]
    mean, sd = s.mean(axis=1), s.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, mean / np.where(sd > 0, sd, 1) * math.sqrt(365), np.nan)
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [math.nan, math.nan]
    return {"mean_daily_pnl_ci": ci(mean), "sharpe_ci": ci(sharpe)}


def forecast_quality(fc, err, alpha, edge_threshold):
    lo, hi = f"q{round((1 - alpha) * 100)}", f"q{round(alpha * 100)}"
    d = fc.merge(err[["pair", "hour_ending", "error"]], on=["pair", "hour_ending"])
    traded = (d[lo] > edge_threshold) | (d[hi] < -edge_threshold)
    hits = np.sign(d.loc[traded, "q50"]) == np.sign(d.loc[traded, "error"])
    return {
        "hours": int(len(d)),
        "mae": float((d["q50"] - d["error"]).abs().mean()),
        "coverage_10_90": float(((d["error"] >= d["q10"]) & (d["error"] <= d["q90"])).mean()),
        "hours_traded": int(traded.sum()),
        "hit_rate_traded": float(hits.mean()) if traded.any() else math.nan,
    }
```

- [ ] **Step 3:** `uv run pytest tests/test_metrics.py -v` → 4 passed

- [ ] **Step 4: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: trading metrics, block bootstrap, forecast quality"
```

---

### Task 12: End-to-end run

**Files:** Create `src/run_backtest.py`.

- [ ] **Step 1: Write `src/run_backtest.py`**

```python
"""Every model x period: tune alpha on 2024, report train/test_a/test_b, cost sensitivity -> output/."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as c
from backtest import run_backtest
from baselines import hour_month_forecast, hour_month_table, zero_forecast
from chronos_model import load_forecasts
from metrics import block_bootstrap, forecast_quality, summarize
from processing import load_hourly
from spreads import build_spreads
from strategy import apply_daily_cap, clear, gate_and_size

ERR_COLS = ["pair", "hour_ending", "hour", "da_gap", "rt_gap", "error"]


def scarcity_dates(hourly, price):
    """Weather-filter proxy: days where any hub's hourly RT price exceeded `price`."""
    day_max = hourly.groupby("delivery_date")["rtm"].max()
    return set(day_max[day_max > price].index)


def in_period(df, name):
    start, end = c.PERIODS[name]
    return df[df["delivery_date"].between(pd.Timestamp(start), pd.Timestamp(end))]


def simulate(fc, err, alpha, cost, period, scarce):
    bids = gate_and_size(in_period(fc, period), alpha, c.CAPITAL, c.EDGE_THRESHOLD, c.MAX_MW_HOUR, c.SIGMA_FLOOR)
    bids = bids.merge(err[ERR_COLS], on=["pair", "hour_ending"])
    bids = apply_daily_cap(clear(bids, c.BID_CAP), c.MAX_MW_DAY)
    return run_backtest(bids, c.CAPITAL, cost, scarce, c.DAILY_HALF, c.DAILY_STOP, c.MAX_DRAWDOWN)


def tune_alpha(fc, err, scarce):
    """Highest 2024 Sharpe; ties -> fewest trades (spec 6.1)."""
    scores = []
    for a in c.ALPHAS:
        s = summarize(*simulate(fc, err, a, c.COST_PER_MWH, "train", scarce), c.CAPITAL)
        sharpe = s["sharpe"] if s["sharpe"] == s["sharpe"] else float("-inf")
        scores.append((sharpe, -s["trades"], a))
    return max(scores)[2]


def breakeven(costs, pnls):
    """Cost where PnL crosses zero, by linear interpolation on the grid; None if it never crosses."""
    for (c0, p0), (c1, p1) in zip(zip(costs, pnls), zip(costs[1:], pnls[1:])):
        if p0 >= 0 > p1:
            return c0 + (c1 - c0) * p0 / (p0 - p1)
    return None


def plot(daily, name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    daily["equity"].plot(ax=ax1, title=f"Equity: {name} ($)")
    (-daily["drawdown"] * 100).plot(ax=ax2, title="Drawdown (%)")
    fig.tight_layout()
    fig.savefig(c.OUTPUT / f"equity_{name}.png", dpi=150)
    plt.close(fig)


def main():
    hourly = load_hourly()
    err = build_spreads(hourly, c.PAIRS)
    scarce = scarcity_dates(hourly, c.SCARCITY_PRICE)
    models = {
        "chronos_cov": load_forecasts("chronos_cov"),
        "chronos_nocov": load_forecasts("chronos_nocov"),
        "hour_month": hour_month_forecast(err, hour_month_table(err, c.TRAIN_START, c.TRAIN_END)),
    }
    results = {"pairs": [f"{a}->{b}" for a, b in c.PAIRS], "models": {}}
    for name, fc in models.items():
        alpha = tune_alpha(fc, err, scarce)
        r = {"alpha": alpha}
        for period in c.PERIODS:
            daily, log = simulate(fc, err, alpha, c.COST_PER_MWH, period, scarce)
            r[period] = {
                **summarize(daily, log, c.CAPITAL),
                **block_bootstrap(daily["pnl"]),
                "forecast": forecast_quality(in_period(fc, period), err, alpha, c.EDGE_THRESHOLD),
            }
            daily.to_csv(c.OUTPUT / f"daily_{name}_{period}.csv", index_label="delivery_date")
            log.to_csv(c.OUTPUT / f"trade_log_{name}_{period}.csv", index=False)
            plot(daily, f"{name}_{period}")
        results["models"][name] = r

    results["zero_mae"] = {p: forecast_quality(in_period(zero_forecast(err), p), err, 0.7, c.EDGE_THRESHOLD)["mae"]
                           for p in c.PERIODS}

    fc, alpha = models["chronos_cov"], results["models"]["chronos_cov"]["alpha"]
    pnls = [sum(summarize(*simulate(fc, err, alpha, cost, p, scarce), c.CAPITAL)["total_pnl"]
                for p in ("test_a", "test_b")) for cost in c.COSTS]
    results["cost_sensitivity"] = {"costs": c.COSTS, "test_pnl": pnls, "breakeven_cost": breakeven(c.COSTS, pnls)}

    (c.OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Full test suite.** `uv run pytest -v` → all pass (33)

- [ ] **Step 3: Run.** `uv run python src/run_backtest.py` writes `output/metrics.json`, and daily CSVs, trade logs and charts per model and period.

- [ ] **Step 4: Sanity checks** (must hold before anyone sees numbers)

```bash
uv run python -c "
import pandas as pd, glob
for f in glob.glob('output/trade_log_*_test_*.csv'):
    log = pd.read_csv(f)
    assert (log.groupby('delivery_date').mw.sum() <= 40 + 1e-9).all(), f + ': daily cap'
    assert (log.mw <= 10 + 1e-9).all(), f + ': hourly cap'
    assert (log.price_paid <= 25 + 1e-9).all(), f + ': bid cap'
    print(f, len(log), 'trades')
"
```
Eyeball:
- Near-zero trades means the gate is too strict: inspect the 2024 quantiles, don't loosen a rule silently.
- 2024 much better than test means overfitting: report it.
- A test Sharpe above ~3 is suspicious: re-check look-ahead in `chronos_model.origin` and `covariates.pick_vintage`.

- [ ] **Step 5: Commit**

```bash
git add src/run_backtest.py output/
git commit -m "feat: end-to-end evaluation of all models and periods"
```

---

### Task 13 (stretch): Alternative signals S1 and S2

Do this only once Task 12 runs end to end. Spec §5.1. (S3 and S4 are not planned; add them later only if time allows.)

**Files:** Modify `src/strategy.py`, `src/run_backtest.py`, `tests/test_strategy.py`.

- [ ] **Step 1: Failing test** (append to `tests/test_strategy.py`)

```python
from strategy import flat_signal, persistence


def test_persistence_uses_same_hour_two_days_back():
    d = pd.date_range("2025-01-01", periods=4, freq="D")
    err = pd.DataFrame({"pair": "A->B", "hour": 0, "delivery_date": d,
                        "hour_ending": d.tz_localize("UTC") + pd.Timedelta(hours=6), "error": [7.0, -9.0, 1.0, 2.0]})
    sig = persistence(err)
    assert sig["signal"].tolist()[2:] == [7.0, -9.0]
    out = flat_signal(sig, edge_threshold=5.0, max_mw_hour=10.0)
    assert out["direction"].tolist()[2:] == [1, -1] and out["mw_wanted"].tolist()[2:] == [10.0, 10.0]
```

- [ ] **Step 2: Add to `src/strategy.py`**

```python
import pandas as pd


def persistence(err):
    """S1: the auction's error for the same pair and hour on T-2 (last settled day at bid time)."""
    shifted = err[["pair", "hour", "delivery_date", "error"]].assign(delivery_date=lambda d: d["delivery_date"] + pd.Timedelta(days=2))
    out = err[["pair", "hour_ending", "delivery_date", "hour"]].merge(shifted, on=["pair", "hour", "delivery_date"], how="left")
    return out.groupby(["pair", "hour_ending", "delivery_date"], as_index=False)["error"].mean().rename(columns={"error": "signal"})


def flat_signal(sig, edge_threshold, max_mw_hour):
    """S1/S2: trade the sign of `signal` beyond the threshold at a flat size."""
    out = sig.copy()
    out["direction"] = np.select([out["signal"] > edge_threshold, out["signal"] < -edge_threshold], [1, -1], 0)
    out["mw_wanted"] = np.where(out["direction"] != 0, max_mw_hour, 0.0)
    out["rank"] = out["signal"].abs()
    return out
```

- [ ] **Step 3: Add to `src/run_backtest.py`.** Before `(c.OUTPUT / "metrics.json")…`, insert:

```python
    from strategy import flat_signal, persistence
    hm = models["hour_month"].rename(columns={"q50": "signal"})[["pair", "hour_ending", "delivery_date", "signal"]]
    for name, sig in {"S1_persistence": persistence(err), "S2_calendar": hm}.items():
        results["models"][name] = {}
        for period in c.PERIODS:
            bids = flat_signal(in_period(sig, period), c.EDGE_THRESHOLD, c.MAX_MW_HOUR).merge(err[ERR_COLS], on=["pair", "hour_ending"])
            daily, log = run_backtest(apply_daily_cap(clear(bids, c.BID_CAP), c.MAX_MW_DAY), c.CAPITAL, c.COST_PER_MWH,
                                      scarce, c.DAILY_HALF, c.DAILY_STOP, c.MAX_DRAWDOWN)
            results["models"][name][period] = {**summarize(daily, log, c.CAPITAL), **block_bootstrap(daily["pnl"])}
```

- [ ] **Step 4:** `uv run pytest -v` → all pass. Then `uv run python src/run_backtest.py` → `metrics.json` now includes `S1_persistence` and `S2_calendar`.

- [ ] **Step 5: Commit**

```bash
git add src/strategy.py src/run_backtest.py tests/test_strategy.py output/
git commit -m "feat: persistence and calendar-premium alternative signals"
```

---

### Task 14: Deliverable notebook

**Files:** Create `notebooks/ptp_strategy.ipynb`.

It reads only `output/` and `data/processed/`, never downloads or forecasts, so it runs in about a minute. Build it with `uv run jupyter lab`, one markdown cell then one code cell per section:

| # | Section (markdown) | Code cell |
|---|---|---|
| 1 | The trade: PTP Obligation, `error = rt_gap − da_gap`, the 10:00 timing rule | `import sys; sys.path.insert(0, "../src"); import json, pandas as pd, config as c` |
| 2 | Data: hubs, period, RTC+B break | `from processing import load_hourly; h = load_hourly(); h.groupby("point").rtm.describe()` |
| 3 | Pair choice | `pd.read_csv("../output/pair_screen.csv", index_col=0).round(2)` |
| 4 | How big is the auction's error? | `from spreads import build_spreads; e = build_spreads(h, c.PAIRS); e.groupby(["pair", "hour"]).error.mean().unstack(0).plot(title="Mean auction error by hour")` |
| 5 | Models: Chronos-2 ± covariates vs hour × month vs zero | `m = json.load(open("../output/metrics.json")); pd.DataFrame({k: {p: v[p]["forecast"]["mae"] for p in c.PERIODS} for k, v in m["models"].items() if "forecast" in v.get("train", {})})` |
| 6 | Rules: gate, half-Kelly, caps, risk stops | Markdown only: copy spec §6–7 in plain words |
| 7 | Results by period | `pd.DataFrame({(k, p): {x: v[p][x] for x in ["total_pnl", "trades", "win_rate", "win_loss_ratio", "max_drawdown_pct", "sharpe"]} for k, v in m["models"].items() for p in c.PERIODS}).T.round(2)` |
| 8 | Equity curves | `from IPython.display import Image; Image("../output/equity_chronos_cov_test_a.png")`, plus `test_b` |
| 9 | Luck or skill? | `{k: v["test_a"]["sharpe_ci"] for k, v in m["models"].items()}` |
| 10 | Costs | `m["cost_sensitivity"]` |
| 11 | Case studies | `log = pd.read_csv("../output/trade_log_chronos_cov_test_a.csv"); pd.concat([log.nlargest(1, "pnl"), log.nsmallest(1, "pnl")])` and `d = pd.read_csv("../output/daily_chronos_cov_test_a.csv"); d[d.multiplier < 1].head()` |
| 12 | What we learned and limitations | Markdown: spec §9, plus the actual findings (written by the team, not generated) |

- [ ] **Step 1:** Create the notebook with the cells above.
- [ ] **Step 2:** Restart the kernel and run all cells. Every cell runs without errors.
- [ ] **Step 3: Commit** (with outputs, so the professor sees charts without running it)

```bash
git add notebooks/ptp_strategy.ipynb
git commit -m "docs: deliverable notebook"
```

---

### Task 15: CLAUDE.md

- [ ] **Step 1:** Replace the "Environment & commands" and "Data pipeline" sections of `CLAUDE.md` with:

```markdown
## Commands

- Python ≥ 3.13 via `uv`. Tests: `uv run pytest -v`; one test: `uv run pytest tests/test_strategy.py::test_alpha_selects_quantile -v`.
- Credentials for downloads: `ERCOT_USERNAME`, `ERCOT_PASSWORD`, `ERCOT_SUBSCRIPTION_KEY` in the environment.
- Pipeline (all resumable, constants in `src/config.py`, rules in `docs/superpowers/specs/2026-11-13-ptp-strategy-design.md`):
  1. `uv run python src/data_import.py`: prices
  2. `uv run python src/processing.py`
  3. `uv run python src/screen.py`, then set `PAIRS` in config
  4. `uv run python src/covariates.py` then `uv run python src/covariates.py build`: wind/load forecasts
  5. `uv run python src/chronos_model.py chronos_cov` and `... chronos_nocov`: slow, cached in `output/forecasts/`
  6. `uv run python src/run_backtest.py`: `output/metrics.json`
  7. `notebooks/ptp_strategy.ipynb`: the deliverable

## Timing rule (never break it)

Bids for delivery day T are made at 10:00 on T−1. Usable: RT prices through T−2, DA prices through T−1, forecasts published before 10:00 on T−1. Look-ahead tests live in `tests/test_chronos_model.py`, `tests/test_covariates.py` and `tests/test_backtest.py`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: pipeline commands"
```
