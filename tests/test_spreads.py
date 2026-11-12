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
