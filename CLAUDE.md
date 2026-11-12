# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

University group project (Applied Algorithmic Trading): trade ERCOT (Texas grid) congestion via **PTP Obligations**, source `HB_WEST` → sink `HB_NORTH`. Predict tomorrow's Real-Time (RTM) spread, compare with the Day-Ahead (DAM) spread, trade when the edge is large enough, then backtest. Deliverables are a deck (`texas_congestion_pitch.pptx`) and commented code (ideally a Jupyter notebook) emailed to the professor.

Key docs (read before changing strategy or data code):
- `project_idea.md`: strategy outline (features: historical congestion, West wind forecast, North load forecast, line outages).
- `data_loading.md`: workflow checklist (in Chinese) with the **concrete strategy rules**, and ERCOT data portal instructions plus column schemas.

Strategy rules from `data_loading.md` (the grading requires code to match the slides exactly):
- Edge = predicted RT gap − DA gap. Enter West→North if edge > $5/MWh, reverse if < −$5/MWh; hold to next-day settlement.
- Sizing: half-Kelly, capped at 10 MW/hour and 40 MW/day.
- Risk: bid cap $25/MWh; daily stop (prior-day loss > 2% → halve size, > 5% → no trading); strategy stop at 15% drawdown; extreme-weather filter.
- Backtest PnL = (RT spread − DA spread) × MW, minus ~$0.5/MWh cost. Avoid look-ahead bias.

## Environment & commands

- Python ≥ 3.13, local `.venv/` (PyCharm virtualenv, currently empty). `pyproject.toml` declares **no dependencies** yet, though the code needs `pandas` and an `ercot` package. Record any new deps there.
- No tests, linter, or build step exist.
- Script paths resolve from `__file__`, so the cwd doesn't matter:
  - `python src/data_ingestion.py`: dry run that previews renames of raw ERCOT downloads.
  - `python src/data_ingestion.py --apply`: performs the renames.
  - `python src/data_import.py`: downloads DAM/RTM for `HB_WEST`/`HB_NORTH` over the hard-coded `START_DATE`/`END_DATE`.

## Data pipeline

Layout: `data/raw/{DAM,RTM}` → `data/processed/` (planned). There are two ingestion paths, and their file naming conflicts:

1. **API download** (`src/data_import.py`, via `ercot` lib): writes `DAM_{point}_{start}_{end}.csv` in **wide** format (`date, settlement_point, HE1..HE24`, hourly) and RTM in **long** format (15-minute intervals, which must be aggregated to hourly before comparing with DAM). The committed January 2026 sample in `data/raw/DAM/` came from this path.
2. **Manual portal download** (`src/data_ingestion.py`): renames ERCOT files prefixed `cdr*` (DAM, daily) → `DAM_YYYY-MM-DD.csv` and `rpt*` (RTM, yearly xlsx) → `RTM_YYYY.xlsx`. It never overwrites existing files.

Gotchas:
- DST: DAM has `DSTFlag` and RTM has `Repeated Hour Flag`. Handle the duplicated or missing hour when aligning DAM and RTM.
- `data_loading.md` refers to `src/data_processing.py` (filter to the hubs and write to `data/processed/`) and to `data_imports.py`. Neither exists yet; the actual file is `data_import.py`.
- `src/data_import.py` reads `ERCOT_USERNAME`, `ERCOT_PASSWORD`, `ERCOT_SUBSCRIPTION_KEY` from the environment. Never commit real credentials.

## ERCOT Public API

`docs/ercot_pubapi.yaml` is the full OpenAPI spec (base URL `https://api.ercot.com/api/public-reports`, about 245 report endpoints). Subscription-key examples in it are redacted. Useful endpoints: `/np4-190-cd/dam_stlmnt_pnt_prices` (DAM prices), `/np6-905-cd/spp_node_zone_hub` (RTM 15-minute prices). Each request needs an `Ocp-Apim-Subscription-Key` header and a Bearer ID token; the token comes from ERCOT's B2C ROPC login and expires after 1 hour with no refresh, so long downloads must re-authenticate.
