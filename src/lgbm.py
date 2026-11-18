"""LightGBM quantile regression for the auction's error: the supervised benchmark for Chronos-2.

One model per quantile, all pairs pooled (pair is a feature), retrained at the start of every month
on rows whose result was known by then. Every feature for delivery day T is known at 10:00 on T-1:
- errors (RT gap - DA gap) up to the end of T-2,
- DA gaps up to T-1 (the T-1 auction clears on T-2),
- wind/solar/load forecasts as published before 10:00 on T-1 (covariates.py).

    uv run python src/lgbm.py            # 2024, with and without wind/solar/load
    uv run python src/lgbm.py test_a     # another period from config.PERIODS

Writes output/forecasts/compare/lgbm_{cov,lags}_monthly[_<period>].csv in the compare_models format.
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from compare_models import ALL_PAIRS, CALENDAR, FUNDAMENTALS, OUT
from config import CLIP, PERIODS, QCOLS, QUANTILES
from covariates import calendar, load_covariates
from extra_data import EXTRA_FILE, load_extra
from processing import load_hourly
from spreads import build_spreads

MISSES = ["wind_west", "wind_pan", "wind_sys", "solar_sys", "load_sys", "load_ncent", "load_farwest"]
SHADOW = ["sp_westex", "sp_pnhndl", "sp_total", "sp_max", "sp_count"]
OUTAGE = ["out_west", "out_north", "out_south", "out_houston", "out_irr_west", "out_irr_north"]
EXTRA = ([f"{m}_miss_lag2" for m in MISSES] + [f"{m}_miss_day2" for m in MISSES] + [f"{c}_lag1" for c in SHADOW]
         + ["sp_westex_day1", "sp_pnhndl_day1"] + OUTAGE)
LAGS = ["err_lag2", "err_lag3", "err_lag7", "err_mean7", "err_std7", "err_day2", "rt_lag2", "da_lag1", "da_day1"]
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)   # untuned defaults: tuning on 2024 is in-sample


def features(s, cov=None, extra=None):
    """One row per (pair, delivery_date, hour) with lagged errors/gaps, calendar and (optionally) covariates
    and extra inputs (forecast misses on T-2, DAM shadow prices on T-1, outage forecast for T)."""
    s = s.groupby(["pair", "delivery_date", "hour"], as_index=False).agg(   # fall-back DST: two HE2s -> one row
        hour_ending=("hour_ending", "first"), error=("error", "mean"), da_gap=("da_gap", "mean"),
        rt_gap=("rt_gap", "mean"))
    daily = ["error", "da_gap"]
    if extra is not None:   # same-hour values, only ever used lagged below (misses T-2, shadow prices T-1)
        e = extra.join(cov[MISSES])
        hourly = pd.DataFrame({f"{m}_miss": e[f"{m}_act"] - e[m] for m in MISSES} | {c: e[c] for c in SHADOW})
        s = s.join(hourly, on="hour_ending")
        daily += [f"{m}_miss" for m in MISSES] + ["sp_westex", "sp_pnhndl"]
    by_hour = s.set_index(["pair", "delivery_date", "hour"])
    by_day = s.groupby(["pair", "delivery_date"])[daily].mean()

    def lag(col, days, table=by_hour, keys=("pair", "delivery_date", "hour")):
        idx = pd.MultiIndex.from_arrays([s[k] - pd.Timedelta(days=days) if k == "delivery_date" else s[k] for k in keys])
        return table[col].reindex(idx).to_numpy()

    out = s[["pair", "delivery_date", "hour", "hour_ending", "error"]].copy()
    out["err_lag2"], out["err_lag3"], out["err_lag7"] = lag("error", 2), lag("error", 3), lag("error", 7)
    week = np.column_stack([lag("error", k) for k in range(2, 9)])            # same hour, T-8 .. T-2
    out["err_mean7"], out["err_std7"] = np.nanmean(week, axis=1), np.nanstd(week, axis=1)
    out["err_day2"] = lag("error", 2, by_day, ("pair", "delivery_date"))
    out["rt_lag2"] = lag("rt_gap", 2)
    out["da_lag1"] = lag("da_gap", 1)                                         # T-1 auction, public on T-2
    out["da_day1"] = lag("da_gap", 1, by_day, ("pair", "delivery_date"))
    cal = calendar(out["hour_ending"])
    out[CALENDAR] = cal[CALENDAR].to_numpy()
    if cov is not None:
        out = out.join(cov[FUNDAMENTALS], on="hour_ending")
    if extra is not None:
        for m in MISSES:                                                      # actuals for T-2 posted by 09:55 T-1
            out[f"{m}_miss_lag2"] = lag(f"{m}_miss", 2)
            out[f"{m}_miss_day2"] = lag(f"{m}_miss", 2, by_day, ("pair", "delivery_date"))
        for c in SHADOW:                                                      # T-1's auction, public on T-2
            out[f"{c}_lag1"] = lag(c, 1)
        for c in ["sp_westex", "sp_pnhndl"]:
            out[f"{c}_day1"] = lag(c, 1, by_day, ("pair", "delivery_date"))
        out = out.join(extra[OUTAGE], on="hour_ending")                       # forecast for T, 10:00 T-1 vintage
    out["pair"] = out["pair"].astype("category")
    return out


def train_cutoff(month_start):
    """Last delivery day whose result is known before any bid in this month (bids for T go in on T-1, know T-2)."""
    return pd.Timestamp(month_start) - pd.Timedelta(days=2)


def forecast(feat, cols, start, end):
    rows = []
    for m in pd.period_range(start, end, freq="M"):
        first, last = max(m.start_time, pd.Timestamp(start)), min(m.end_time.normalize(), pd.Timestamp(end))
        train = feat[(feat["delivery_date"] <= train_cutoff(m.start_time)) & feat["error"].notna()]
        test = feat[(feat["delivery_date"] >= first) & (feat["delivery_date"] <= last)]
        if FUNDAMENTALS[0] in cols:
            test = test[test[FUNDAMENTALS].notna().all(axis=1)]   # spec 4.2: no forecast -> no trade that hour
        y = train["error"].clip(-CLIP, CLIP)
        q = np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=a, **PARAMS)
                             .fit(train[cols], y).predict(test[cols]) for a in QUANTILES])
        part = pd.DataFrame(np.sort(q, axis=1), columns=QCOLS, index=test.index)   # no quantile crossing
        part.insert(0, "hour_ending", test["hour_ending"])
        part.insert(0, "pair", test["pair"].astype(str))
        part.insert(0, "delivery_date", test["delivery_date"].dt.strftime("%Y-%m-%d"))
        rows.append(part)
        print(f"{m}: trained on {len(train):,} rows up to {train_cutoff(m.start_time):%Y-%m-%d}, "
              f"forecast {len(part):,}", flush=True)
    return pd.concat(rows, ignore_index=True)


def main(period="train"):
    start, end = PERIODS[period]
    extra = load_extra() if EXTRA_FILE.exists() else None
    feat = features(build_spreads(load_hourly(), ALL_PAIRS), load_covariates(), extra)
    suffix = "" if period == "train" else f"_{period}"
    models = [("lgbm_lags", ["pair"] + LAGS + CALENDAR), ("lgbm_cov", ["pair"] + LAGS + CALENDAR + FUNDAMENTALS)]
    if extra is not None:
        base = ["pair"] + LAGS + CALENDAR + FUNDAMENTALS
        models.append(("lgbm_extra", base + EXTRA))
        models += [(f"lgbm_{g}", base + [c for c in EXTRA if c.startswith(pre)])     # one extra group at a time
                   for g, pre in [("miss", tuple(f"{m}_miss" for m in MISSES)), ("shadow", "sp_"), ("outage", "out_")]]
    if len(sys.argv) > 2:                                                     # e.g. `lgbm.py test_a lgbm_extra`
        models = [m for m in models if m[0] in sys.argv[2:]]
    for name, cols in models:
        out = forecast(feat, cols, start, end)
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT / f"{name}_monthly{suffix}.csv", index=False)
        print(f"saved {name}: {out['delivery_date'].nunique()} days")


if __name__ == "__main__":
    main(*sys.argv[1:2])
