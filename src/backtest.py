"""Backtest of the trading rules in the spec (sections 6-8) on HB_WEST -> HB_NORTH.

Per model: tune the gate (theta, alpha) on 2024 only, freeze it, then trade Test A and Test B.
Each period starts from fresh capital. Every rule only uses information known at 10:00 on T-1:
the forecast for T, and settled PnL / RT prices up to T-2.

    uv run python src/backtest.py      # -> output/metrics.json, output/backtest/{trades,daily}_<model>_<period>.csv
"""
import json

import numpy as np
import pandas as pd

from compare_models import OUT
from config import (ALPHAS, BID_CAP, CAPITAL, COST_PER_MWH, COSTS, DAILY_HALF, DAILY_STOP, EDGE_THRESHOLDS,
                    MAX_DRAWDOWN, MAX_MW_DAY, MAX_MW_HOUR, MIN_TUNE_TRADES, OUTPUT, PERIODS, QCOLS, QUANTILES,
                    SCARCITY_PRICE, SIGMA_FLOOR)
from processing import load_hourly
from spreads import build_spreads

PAIR = ("HB_WEST", "HB_NORTH")
BT = OUTPUT / "backtest"
MODELS = {   # key -> forecast file per period ({s} = "" for 2024, "_test_a", "_test_b")
    "lgbm": lambda s: f"lgbm_cov_monthly{s}",                         # main model: LightGBM + fundamentals
    "chronos": lambda s: f"chronos_ctx2048_WEST-NORTH{s}",            # comparison: zero-shot Chronos-2
    "baseline": lambda s: f"baseline_ctx2160{'_WEST-NORTH' if s else ''}{s}",   # hour-of-day, last 90 days
}
LABELS = {"lgbm": "LightGBM + fundamentals", "chronos": "Chronos-2 (zero-shot)", "baseline": "Hour-of-day baseline"}


def suffix(period):
    return "" if period == "train" else f"_{period}"


def load_forecast(model, period, spreads):
    f = pd.read_csv(OUT / f"{MODELS[model](suffix(period))}.csv")
    f = f[f["pair"] == "->".join(PAIR)].drop(columns=["delivery_date", "pair"])
    f["hour_ending"] = pd.to_datetime(f["hour_ending"], utc=True)
    start, end = PERIODS[period]
    s = spreads[(spreads["delivery_date"] >= start) & (spreads["delivery_date"] <= end)]
    return s.merge(f, on="hour_ending").dropna(subset=QCOLS)


def plan(f, theta, alpha):
    """Bids decided before the auction: gate + direction, half-Kelly size (10 MW cap), 40 MW/day on largest |q50|."""
    lo, hi = (QCOLS[QUANTILES.index(round(a, 1))] for a in (1 - alpha, alpha))
    f = f.assign(direction=np.select([f[lo] > theta, f[hi] < -theta], [1, -1], 0))
    f = f[f["direction"] != 0].copy()
    sigma = ((f["q90"] - f["q10"]) / 2.563).clip(lower=SIGMA_FLOOR)
    f["mw"] = (0.5 * CAPITAL * f["q50"].abs() / sigma ** 2).clip(upper=MAX_MW_HOUR)
    f["conf"] = f["q50"].abs()
    f = f.sort_values(["delivery_date", "conf"], ascending=[True, False])
    used = f.groupby("delivery_date")["mw"].cumsum() - f["mw"]
    f["mw"] = np.minimum(f["mw"], (MAX_MW_DAY - used).clip(lower=0))
    return f[f["mw"] > 0]


def risk_scale(day_pnl, i, spike, stopped):
    """Size multiplier for day i from settled days <= i-2 only. Returns (scale, stopped)."""
    settled = day_pnl[:max(i - 1, 0)]                 # days 0 .. i-2
    equity = CAPITAL + settled.cumsum()
    if len(settled) and 1 - equity[-1] / max(CAPITAL, equity.max()) > MAX_DRAWDOWN:
        stopped = True
    if stopped:
        return 0.0, True
    scale = 1.0
    if len(settled):
        loss = -settled[-1] / (equity[-1] - settled[-1])   # T-2's loss as a share of equity before it
        scale = 0.0 if loss > DAILY_STOP else 0.5 if loss > DAILY_HALF else 1.0
    return scale * (0.5 if spike else 1.0), False


def simulate(bids, days, spike_days, cost=COST_PER_MWH):
    """Day-by-day PnL with the risk rules. Returns (trade log, daily frame)."""
    by_day = dict(tuple(bids.groupby("delivery_date")))
    day_pnl, scales, stopped, logs = np.zeros(len(days)), np.ones(len(days)), False, []
    for i, d in enumerate(days):
        scales[i], stopped = risk_scale(day_pnl, i, d - pd.Timedelta(days=2) in spike_days, stopped)
        if d in by_day:
            b = by_day[d].assign(scale=scales[i])
            b["cleared"] = b["direction"] * b["da_gap"] <= BID_CAP
            b["mw_final"] = b["mw"] * b["scale"] * b["cleared"]
            b["pnl"] = b["mw_final"] * (b["direction"] * b["error"] - cost)
            day_pnl[i] = b["pnl"].sum()
            logs.append(b)
    cols = ["delivery_date", "hour", "hour_ending", "direction", "q10", "q50", "q90", "mw", "scale", "da_gap",
            "rt_gap", "error", "cleared", "mw_final", "pnl"]
    trades = pd.concat(logs)[cols] if logs else pd.DataFrame(columns=cols)
    daily = pd.DataFrame({"delivery_date": days, "pnl": day_pnl, "scale": scales})
    daily["equity"] = CAPITAL + daily["pnl"].cumsum()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax().clip(lower=CAPITAL) - 1
    return trades, daily


def block_bootstrap(x, n=5000, block=7, seed=0):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(x) / block))
    idx = (rng.integers(0, len(x) - block + 1, size=(n, nb))[:, :, None] + np.arange(block)).reshape(n, -1)[:, :len(x)]
    return x[idx]


def sharpe(x, axis=None):
    sd = x.std(axis=axis, ddof=1)
    return np.where(sd > 0, x.mean(axis=axis) / np.where(sd > 0, sd, 1) * np.sqrt(365), 0.0)


def metrics(trades, daily):
    t = trades[trades["mw_final"] > 0]
    wins, losses = t.loc[t["pnl"] > 0, "pnl"], t.loc[t["pnl"] <= 0, "pnl"]
    x = daily["pnl"].to_numpy()
    boot = block_bootstrap(x)
    return {
        "total_pnl": float(x.sum()), "return_pct": float(x.sum() / CAPITAL * 100), "trades": int(len(t)),
        "mwh": float(t["mw_final"].sum()), "win_rate": float((t["pnl"] > 0).mean()) if len(t) else None,
        "avg_win": float(wins.mean()) if len(wins) else None, "avg_loss": float(losses.mean()) if len(losses) else None,
        "win_loss_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None,
        "max_drawdown_pct": float(daily["drawdown"].min() * 100), "sharpe": float(sharpe(x)),
        "mean_daily_ci": [float(v) for v in np.percentile(boot.mean(1), [2.5, 97.5])],
        "sharpe_ci": [float(v) for v in np.percentile(sharpe(boot, axis=1), [2.5, 97.5])],
        "days_scaled_down": int((daily["scale"] < 1).sum()),
    }


def period_days(period):
    return list(pd.date_range(*PERIODS[period]))


def tune(f, spike_days):
    """Spec 6: grid on 2024; eligible if >= MIN_TUNE_TRADES gated hours; best Sharpe, ties -> fewer trades."""
    grid = []
    for theta in EDGE_THRESHOLDS:
        for alpha in ALPHAS:
            bids = plan(f, theta, alpha)
            n_signals = int(((f[QCOLS[QUANTILES.index(round(1 - alpha, 1))]] > theta)
                             | (f[QCOLS[QUANTILES.index(alpha)]] < -theta)).sum())
            _, daily = simulate(bids, period_days("train"), spike_days)
            grid.append({"theta": theta, "alpha": alpha, "signals": n_signals, "sharpe": float(sharpe(daily["pnl"].to_numpy())),
                         "pnl": float(daily["pnl"].sum())})
    ok = [g for g in grid if g["signals"] >= MIN_TUNE_TRADES]
    best = min(ok, key=lambda g: (-g["sharpe"], g["signals"])) if ok else None
    return best, grid


def main():
    hourly = load_hourly()
    spreads = build_spreads(hourly, [PAIR])
    spike_days = set(hourly.loc[hourly["rtm"] > SCARCITY_PRICE, "delivery_date"])   # weather-filter proxy
    BT.mkdir(parents=True, exist_ok=True)
    out = {"pair": "->".join(PAIR), "capital": CAPITAL, "cost_per_mwh": COST_PER_MWH, "models": {}}
    for model in MODELS:
        fc = {p: load_forecast(model, p, spreads) for p in PERIODS}
        best, grid = tune(fc["train"], spike_days)
        res = {"label": LABELS[model], "tuning_grid": grid, "gate": best, "periods": {}}
        if best is None:
            print(f"{model}: no gate setting reaches {MIN_TUNE_TRADES} trades in 2024 -> does not trade")
        else:
            for p in PERIODS:
                trades, daily = simulate(plan(fc[p], best["theta"], best["alpha"]), period_days(p), spike_days)
                trades.to_csv(BT / f"trades_{model}_{p}.csv", index=False)
                daily.to_csv(BT / f"daily_{model}_{p}.csv", index=False)
                res["periods"][p] = metrics(trades, daily)
            if model == "lgbm":   # spec 8: cost sensitivity + breakeven on Test A + B
                sens = {}
                for c in COSTS:
                    sens[str(c)] = sum(simulate(plan(fc[p], best["theta"], best["alpha"]), period_days(p), spike_days, c)[1]
                                       ["pnl"].sum() for p in ("test_a", "test_b"))
                gross, mwh = sens["0.0"], sum(res["periods"][p]["mwh"] for p in ("test_a", "test_b"))
                # shortcut: ignores that a different cost would change risk-rule triggers; fine while few days are scaled
                res["cost_sensitivity"] = sens
                res["breakeven_cost"] = gross / mwh if mwh else None
            print(f"{model}: gate theta=${best['theta']}, alpha={best['alpha']}")
            print(pd.DataFrame(res["periods"]).T[["total_pnl", "trades", "win_rate", "max_drawdown_pct", "sharpe",
                                                  "mean_daily_ci"]].to_string(), "\n")
        out["models"][model] = res
    (OUTPUT / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"saved {OUTPUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
