import numpy as np
import pytest
import pandas as pd

from compare_models import baseline, day_window, forecast_day, pinball


def _hours(day):
    origin, start, end = day_window(day)
    return (end - origin) / pd.Timedelta(hours=1), (end - start) / pd.Timedelta(hours=1)


def test_window_starts_at_end_of_t_minus_2_and_handles_dst():
    origin, _, _ = day_window("2024-06-02")
    assert origin == pd.Timestamp("2024-06-01 05:00", tz="UTC")   # midnight starting T-1, Chicago (CDT)
    assert _hours("2024-06-02") == (48, 24)
    assert _hours("2024-03-10") == (47, 23)   # spring forward on T
    assert _hours("2024-11-03") == (49, 25)   # fall back on T
    assert _hours("2024-11-04") == (49, 24)   # fall back on T-1: horizon longer, T normal


def test_pinball_penalises_the_right_side():
    assert pinball(np.array([10.0]), np.array([0.0]), 0.9)[0] == pytest.approx(9.0)
    assert pinball(np.array([-10.0]), np.array([0.0]), 0.9)[0] == pytest.approx(1.0)


def test_baseline_uses_hour_of_day_quantiles():
    idx = pd.date_range("2024-01-01 07:00", periods=48, freq="h", tz="UTC")   # HE1 Chicago = 07:00 UTC
    ctx = pd.DataFrame({"A->B": np.tile(np.arange(24.0), 2)}, index=idx)    # error = local hour
    future = pd.date_range("2024-01-03 07:00", periods=2, freq="h", tz="UTC")
    out = baseline(ctx, future)["A->B"]
    assert out.shape == (2, 9)
    assert (out[0] == 0).all() and (out[1] == 1).all()


@pytest.mark.parametrize("day, hours", [("2024-06-02", 24), ("2024-11-03", 25), ("2024-03-10", 23)])
def test_forecast_day_sees_only_history_to_end_of_t_minus_2(day, hours):
    idx = pd.date_range("2024-01-01", "2024-12-31", freq="h", tz="UTC")
    grid = pd.DataFrame({"A->B": np.arange(len(idx), dtype=float)}, index=idx)
    seen = {}

    def fake(ctx, future):
        seen["last"] = ctx.index[-1]
        return {"A->B": np.zeros((len(future), 9))}

    out = forecast_day(grid, pd.Timestamp(day), fake)
    origin, start, end = day_window(day)
    assert seen["last"] == origin                      # the model never sees day T-1 or later
    assert len(out) == hours                           # one row per hour of day T, DST included
    assert out["hour_ending"].min() == start + pd.Timedelta(hours=1)
    assert out["hour_ending"].max() == end
    assert (out["delivery_date"] == day).all()
