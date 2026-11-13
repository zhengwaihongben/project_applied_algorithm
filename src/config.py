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

DOWNLOAD_START = "2023-12-01"   # ERCOT API search only goes back to 2023-11-30; Dec 2023 = model context only
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
EDGE_THRESHOLD = 5.0     # deck's original rule; still used by the flat alternatives S1-S4 and quick model checks
EDGE_THRESHOLDS = [2.0, 3.0, 5.0]   # tuned jointly with ALPHAS on 2024 only, then frozen (spec section 6)
ALPHAS = [0.5, 0.6, 0.7]
MIN_TUNE_TRADES = 1000   # a gate setting needs this many 2024 trades to be eligible (spec section 6)
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
