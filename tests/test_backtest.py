import numpy as np
import pandas as pd

from backtest import plan, risk_scale, simulate
from config import CAPITAL


def _bids(n_hours=6, q50=5.0):
    rows = [{"delivery_date": pd.Timestamp("2024-06-01"), "hour": h, "hour_ending": h, "q10": q50 - 4, "q20": 0,
             "q30": 0, "q40": 0, "q50": q50 + h, "q60": 0, "q70": 0, "q80": 0, "q90": q50 + 4, "da_gap": 1.0,
             "rt_gap": 0.0, "error": 1.0} for h in range(n_hours)]
    return pd.DataFrame(rows)


def test_daily_budget_spends_40_mw_on_most_confident_hours():
    b = plan(_bids(), theta=2, alpha=0.5)
    assert b["mw"].sum() == 40 and set(b["hour"]) == {5, 4, 3, 2}


def test_risk_uses_only_days_up_to_t_minus_2():
    pnl = np.array([0.0, -0.03 * CAPITAL, 0.0])
    assert risk_scale(pnl, 2, False, False)[0] == 1.0     # day 1's loss is not settled at bid time for day 2
    assert risk_scale(pnl, 3, False, False)[0] == 0.5     # ... but is for day 3 (2-5% loss -> half size)
    assert risk_scale(np.array([-0.06 * CAPITAL]), 2, False, False)[0] == 0.0
    assert risk_scale(np.array([-0.03 * CAPITAL]), 2, True, False)[0] == 0.25   # stacks with weather filter


def test_drawdown_stop_is_permanent():
    pnl = np.array([-0.04 * CAPITAL] * 4 + [0.0] * 5)
    scale, stopped = risk_scale(pnl, 7, False, False)
    assert stopped and scale == 0 and risk_scale(np.zeros(9), 8, False, True) == (0.0, True)


def test_bids_above_cap_do_not_clear():
    b = plan(_bids(2).assign(da_gap=30.0), theta=2, alpha=0.5)
    trades, daily = simulate(b, [pd.Timestamp("2024-06-01")], set())
    assert not trades["cleared"].any() and daily["pnl"].iloc[0] == 0
