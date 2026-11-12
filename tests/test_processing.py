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


def test_rtm_hourly_ignores_exact_duplicate_intervals():
    rtm = _rtm("2024-01-01 00:15", 4, [1.0, 2.0, 3.0, 4.0])
    out = rtm_hourly(pd.concat([rtm, rtm.tail(1)], ignore_index=True))   # ERCOT sometimes repeats an interval
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
