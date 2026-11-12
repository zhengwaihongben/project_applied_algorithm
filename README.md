# Betting on Traffic Jams in the Texas Power Grid

Group project for **Applied Algorithmic Trading Strategies**, by the group "Multiple nations": Eric He, Zheng Wai Hong Ben, Hargun Chawla, Pun Chin For.

We trade **Point-to-Point (PTP) Obligations** in ERCOT's Day-Ahead Market: a cash-settled bet on the price gap between two hubs (for example HB_WEST → HB_NORTH). The day-ahead auction sets the price of the bet; ERCOT pays us the real-time gap the next day. Our edge is **forecasting the auction's error** (`real-time gap − day-ahead gap`) with the time-series foundation model Chronos-2, using ERCOT's own wind and load forecasts. The deck's sizing and risk rules are all implemented in code.

- **Design (rules, data, models, evaluation):** [`docs/superpowers/specs/2026-11-13-ptp-strategy-design.md`](docs/superpowers/specs/2026-11-13-ptp-strategy-design.md)
- **Implementation plan (task by task):** [`docs/superpowers/plans/2026-11-13-ptp-backtest-v1.md`](docs/superpowers/plans/2026-11-13-ptp-backtest-v1.md)
- **Pitch deck:** `texas_congestion_pitch.pptx`
- **Final deliverable:** `notebooks/ptp_strategy.ipynb` (built in plan Task 14)

## Status

| Plan task | State |
|---|---|
| 0 Setup, 1 Price download code | Done |
| 1 Run the price download | **Next:** needs your credentials (see below) |
| 2–15 Processing → forecasts → backtest → notebook | To do |

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # installs everything, including Chronos-2 and PyTorch (large download)
uv run pytest -v        # all tests should pass
```

## Access to ERCOT

### Connect to a US VPN first

ERCOT's websites and portals (apiexplorer.ercot.com, developer.ercot.com, ercot.com) may not load from outside the US. **Connect to a US VPN** before registering, browsing the portals, or running any download script.

### Get your own API credentials

Each team member registers their **own** account; never share one login or key. Access is free.

1. **Create an account.** Go to https://apiexplorer.ercot.com and click **Sign In/Sign Up**. Verify your email with the code you receive, then set a password and your name. Your email and that password are your username and password.
2. **Get the subscription key.** While signed in, open **Products**, choose **Public API**, give the subscription a name (for example "Public API") and click **Subscribe**. On your **Profile** page, click **Show** next to **Primary key**.
3. **ID token.** You don't need to do anything here. The scripts log in with your username and password to get a token. Tokens expire after 1 hour and are refreshed automatically.

Official guide: https://developer.ercot.com/applications/pubapi/user-guide/registration-and-authentication/

### Give the credentials to the scripts

The scripts read three environment variables: `ERCOT_USERNAME`, `ERCOT_PASSWORD` and `ERCOT_SUBSCRIPTION_KEY`.

- **PyCharm:** Run → Edit Configurations → select the script → **Environment variables**, and add all three.
- **Terminal:** export them in your shell session before running a script.

Rules:
- **Never** put credentials in code, in a committed file, or in a chat (including AI assistants). If a key leaks, regenerate it on your ERCOT profile page.
- Older versions of `src/data_import.py` saved credentials in plain text to `~/.ercot/credentials.json`. If that file exists on your machine, delete it.

## Running the pipeline

All steps resume after interruption. Constants live in `src/config.py`.

| Step | Command | Output | Time |
|---|---|---|---|
| Prices, 2023-01 → 2026-10, 5 hubs | `uv run python src/data_import.py` | `data/raw/{DAM,RTM}/*_hubs_YYYY-MM.csv` | long |
| Hourly prices | `uv run python src/processing.py` | `data/processed/hourly_prices.csv` | minutes |
| Pair screen (2024) | `uv run python src/screen.py`, then set `PAIRS` in `config.py` | `output/pair_screen.csv` | seconds |
| Wind/load forecasts | `uv run python src/covariates.py probe 2024-06-02`, then `uv run python src/covariates.py`, then `... build` | `data/raw/COV/`, `data/processed/covariates.csv` | hours |
| Chronos-2 forecasts | `uv run python src/chronos_model.py chronos_cov` (and `chronos_nocov`) | `output/forecasts/` | 1–2 h each |
| Backtest and metrics | `uv run python src/run_backtest.py` | `output/metrics.json`, trade logs, charts | minutes |
| Story | open `notebooks/ptp_strategy.ipynb` | — | 1 min |

Only steps 1 and the tests exist so far; the rest are built following the plan.

**One rule the code must never break:** bids for delivery day T are submitted at 10:00 on T−1. The code may use real-time prices through T−2, day-ahead prices through T−1, and forecasts published before 10:00 on T−1, nothing later. Tests check this.

## Repository layout

```
src/            pipeline modules (one responsibility each; see the plan)
tests/          pytest tests, one file per module
docs/           design spec, implementation plan, ERCOT OpenAPI spec (ercot_pubapi.yaml)
data/raw/       downloads (gitignored, regenerate with the scripts)
data/processed/ cleaned tables (gitignored)
output/         results that go into the deck and notebook
notebooks/      the deliverable notebook
```

`src/data_ingestion.py` and `data_loading.md` describe the earlier manual workflow (renaming files downloaded from the ERCOT portal by hand). The API scripts above replace it.

## Data sources

- ERCOT Public API (`https://api.ercot.com/api/public-reports`), via the [`ercot`](https://pypi.org/project/ercot/) Python package and direct calls. Full API description: `docs/ercot_pubapi.yaml`.
  - DAM Settlement Point Prices (NP4-190-CD)
  - RTM Settlement Point Prices (NP6-905-CD)
  - Wind Power Production, Hourly Averaged Actual and Forecasted Values by Geographical Region (NP4-742-CD)
  - Seven-Day Load Forecast by Model and Weather Zone (NP3-565-CD)
- [Chronos-2](https://huggingface.co/amazon/chronos-2) (Amazon, Apache 2.0), zero-shot.
