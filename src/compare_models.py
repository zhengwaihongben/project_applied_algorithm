"""Zero-shot foundation models forecasting the auction's error, compared on 2024 (spec section 5).

For each delivery day T, each model sees the error history up to the end of T-2 (what is known
at 10:00 on T-1) and forecasts quantiles q10..q90 of `error` for every hour of T, for all 10 pairs.
Univariate for now: wind/load/solar inputs are added once the covariate table exists (plan Task 6).

    uv run python src/compare_models.py chronos     # Amazon Chronos-2, local CPU
    uv run python src/compare_models.py timesfm     # Google TimesFM 3, local CPU
    uv run python src/compare_models.py tabpfn-cost # quote the TabPFN run first (free tier: 5M tokens/day, 20M/month)
    uv run python src/compare_models.py tabpfn      # Prior Labs TabPFN-TS, cloud; needs TABPFN_TOKEN
    uv run python src/compare_models.py baseline    # trailing 90-day quantiles per hour of day
    uv run python src/compare_models.py zero        # "the auction is right"
    uv run python src/compare_models.py score       # compare everything forecast so far

Forecasts are cached in output/forecasts/compare/<model>_ctx<hours>.csv; reruns skip finished days,
and changing a model's history length starts a new file.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

from config import CLIP, CONTEXT_HOURS, COST_PER_MWH, EDGE_THRESHOLD, FORECASTS, HUBS, OUTPUT, QCOLS, QUANTILES, \
    TRAIN_END, TRAIN_START
from processing import load_hourly
from spreads import build_spreads

TZ = "America/Chicago"
OUT = FORECASTS / "compare"
ALL_PAIRS = list(itertools.combinations(HUBS, 2))   # one direction per pair; the other is the negative
TABPFN_CONTEXT = 2016                               # 12 weeks; cloud cost grows with context rows
BASELINE_DAYS = 90
ALPHA = 0.7                                         # middle of the spec's grid {0.6, 0.7, 0.8}


def day_window(day):
    """(origin, start, end) in UTC for delivery day T: origin = end of T-2, T's hours are (start, end]."""
    d = pd.Timestamp(day)
    origin, start, end = (x.tz_localize(TZ).tz_convert("UTC") for x in (d - pd.Timedelta(days=1), d, d + pd.Timedelta(days=1)))
    return origin, start, end


def error_grid():
    """Hourly error per pair on a regular UTC grid, columns = pairs (unclipped)."""
    s = build_spreads(load_hourly(), ALL_PAIRS)
    wide = s.pivot_table(index="hour_ending", columns="pair", values="error")
    return wide.reindex(pd.date_range(wide.index[0], wide.index[-1], freq="h"))


# --- models: f(context: DataFrame[time x pair], future: DatetimeIndex) -> {pair: array (len(future), 9)} ---

_cache = {}


def chronos(ctx, future):
    from chronos import Chronos2Pipeline
    if "chronos" not in _cache:
        _cache["chronos"] = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    long = ctx.tail(CONTEXT_HOURS).rename_axis("timestamp").reset_index().melt("timestamp", var_name="id", value_name="target")
    long["timestamp"] = long["timestamp"].dt.tz_localize(None)       # Chronos-2 wants naive timestamps
    pred = _cache["chronos"].predict_df(long, prediction_length=len(future), quantile_levels=QUANTILES,
                                        id_column="id", timestamp_column="timestamp", target="target")
    pred = pred.sort_values(["id", "timestamp"])
    return {p: g[[str(q) for q in QUANTILES]].to_numpy() for p, g in pred.groupby("id")}


def timesfm(ctx, future):
    import timesfm as tfm
    if "timesfm" not in _cache:
        _cache["timesfm"] = tfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")
    ctx = ctx.tail(CONTEXT_HOURS)
    outs = _cache["timesfm"].predict_batch([ctx[p].to_numpy("float32") for p in ctx], horizon=len(future),
                                           return_quantiles=True)
    return {p: o.quantiles[: len(future)] for p, o in zip(ctx.columns, outs)}   # quantiles are 0.1..0.9


def tabpfn(ctx, future):
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
    if "tabpfn" not in _cache:
        if not os.environ.get("TABPFN_TOKEN"):
            sys.exit("Set TABPFN_TOKEN (Prior Labs API key) in the run configuration's environment variables.")
        _cache["tabpfn"] = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.CLIENT)
    long = ctx.tail(TABPFN_CONTEXT).rename_axis("timestamp").reset_index().melt("timestamp", var_name="item_id", value_name="target")
    long["timestamp"] = long["timestamp"].dt.tz_localize(None)
    pred = _cache["tabpfn"].predict_df(long, prediction_length=len(future), quantiles=QUANTILES).reset_index()
    qcols = {str(c): c for c in pred.columns}
    pred = pred.sort_values(["item_id", "timestamp"])
    return {p: g[[qcols[str(q)] for q in QUANTILES]].to_numpy() for p, g in pred.groupby("item_id")}


def baseline(ctx, future):
    """Empirical quantiles of the last 90 days, by local hour of day."""
    recent = ctx.tail(BASELINE_DAYS * 24)
    hour = lambda idx: (idx.tz_convert(TZ) - pd.Timedelta(hours=1)).hour
    q = recent.groupby(hour(recent.index)).quantile(QUANTILES)        # (hour, q) x pair
    fh = hour(future)
    return {p: np.stack([q[p].xs(h, level=0).to_numpy() for h in fh]) for p in ctx}


def zero(ctx, future):
    return {p: np.zeros((len(future), len(QUANTILES))) for p in ctx}


MODELS = {"chronos": chronos, "timesfm": timesfm, "tabpfn": tabpfn, "baseline": baseline, "zero": zero}


CONTEXT = {"chronos": CONTEXT_HOURS, "timesfm": CONTEXT_HOURS, "tabpfn": TABPFN_CONTEXT,
           "baseline": BASELINE_DAYS * 24, "zero": 0}   # hours of history each model sees; part of the file name


def forecast_day(grid, day, model):
    """Quantile rows for each hour of delivery day T, using only history up to the end of T-2."""
    origin, start, end = day_window(day)
    ctx = grid.loc[:origin].ffill().clip(-CLIP, CLIP)              # nothing after the end of T-2
    future = pd.date_range(origin + pd.Timedelta(hours=1), end, freq="h")
    keep = future > start                                           # hours of day T only
    rows = []
    for pair, q in model(ctx, future).items():
        part = pd.DataFrame(np.sort(q[keep], axis=1), columns=QCOLS)    # sorted: no quantile crossing
        part.insert(0, "hour_ending", future[keep])
        part.insert(0, "pair", pair)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "delivery_date", pd.Timestamp(day).strftime("%Y-%m-%d"))
    return out


def run(name):
    grid = error_grid()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_ctx{CONTEXT[name]}.csv"   # a new setting starts a new file instead of mixing runs
    done = set(pd.read_csv(path, usecols=["delivery_date"])["delivery_date"]) if path.exists() else set()
    days = [d for d in pd.date_range(TRAIN_START, TRAIN_END) if d.strftime("%Y-%m-%d") not in done]
    t0 = time.time()
    for i, day in enumerate(days, 1):
        forecast_day(grid, day, MODELS[name]).to_csv(path, mode="a", header=not path.exists(), index=False)
        if i % 10 == 0 or i == len(days):
            print(f"{name}: {i}/{len(days)} days, {time.time() - t0:.0f}s", flush=True)


def tabpfn_cost(n_features=40):
    """Ask Prior Labs what one series costs (shapes only, no data sent, no quota used) and scale to the run."""
    import tabpfn_client
    if not os.environ.get("TABPFN_TOKEN"):
        sys.exit("Set TABPFN_TOKEN (Prior Labs API key) in the run configuration's environment variables.")
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    est = tabpfn_client.estimate_cost(np.zeros((TABPFN_CONTEXT, n_features)), np.zeros((48, n_features)))
    n_series = len(pd.date_range(TRAIN_START, TRAIN_END)) * len(ALL_PAIRS)
    print(f"one series ({TABPFN_CONTEXT} rows x ~{n_features} features): {est.estimated_cost} tokens")
    print(f"full run ({n_series} series): ~{float(est.estimated_cost) * n_series:,.0f} tokens")
    print(tabpfn_client.get_api_usage())


def pinball(y, f, q):
    d = y - f
    return np.maximum(q * d, (q - 1) * d)


def score():
    grid = error_grid()
    actual = grid.stack().rename("error").rename_axis(["hour_ending", "pair"]).reset_index()
    frames = {}
    for path in sorted(OUT.glob("*.csv")):
        f = pd.read_csv(path)
        f["hour_ending"] = pd.to_datetime(f["hour_ending"], utc=True)
        frames[path.stem] = f.merge(actual, on=["pair", "hour_ending"])
    common = set.intersection(*(set(f["delivery_date"]) for f in frames.values()))
    print(f"Scoring {len(common)} delivery days that every model has, {len(ALL_PAIRS)} pairs.\n")
    rows, pnl_by_pair = {}, {}
    lo, hi = QCOLS[QUANTILES.index(round(1 - ALPHA, 1))], QCOLS[QUANTILES.index(ALPHA)]
    for name, f in frames.items():
        f = f[f["delivery_date"].isin(common)]
        y = f["error"].to_numpy()
        side = np.select([f[lo] > EDGE_THRESHOLD, f[hi] < -EDGE_THRESHOLD], [1, -1], 0)   # long / short / flat
        traded = side != 0
        pnl = side * y - COST_PER_MWH * traded
        rows[name] = {
            "pinball": np.mean([pinball(y, f[c], q).mean() for c, q in zip(QCOLS, QUANTILES)]),
            "MAE_median": np.abs(y - f["q50"]).mean(),
            "cover_10_90": ((y >= f["q10"]) & (y <= f["q90"])).mean(),
            "trade_hours": int(traded.sum()),
            "hit_rate": (pnl[traded] > 0).mean() if traded.any() else np.nan,
            "pnl_per_trade": pnl[traded].mean() if traded.any() else np.nan,
            "total_pnl_1MW": pnl.sum(),
        }
        pnl_by_pair[name] = pd.Series(pnl).groupby(f["pair"].to_numpy()).sum()
    table = pd.DataFrame(rows).T.sort_values("pinball")
    print("Forecast quality (lower pinball/MAE is better; cover_10_90 should be ~0.80)")
    print(f"and a simple signal check: trade 1 MW when q{round((1 - ALPHA) * 100)} > ${EDGE_THRESHOLD:.0f} (long) "
          f"or q{round(ALPHA * 100)} < -${EDGE_THRESHOLD:.0f} (short), ${COST_PER_MWH}/MWh cost, no sizing or risk rules.\n")
    print(table.round(3).to_string())
    print("\nTotal PnL per pair at 1 MW ($)")
    print(pd.DataFrame(pnl_by_pair).round(0).to_string())
    table.to_csv(OUTPUT / "model_comparison_2024.csv")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "score":
        score()
    elif arg == "tabpfn-cost":
        tabpfn_cost()
    elif arg in MODELS:
        run(arg)
    else:
        sys.exit(__doc__)
