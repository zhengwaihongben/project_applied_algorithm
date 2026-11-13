import pandas as pd

from covariates import pick_vintage


def _raw(posted):
    return pd.DataFrame({"postedDatetime": posted, "hourEnding": ["01:00"] * len(posted), "v": range(len(posted))})


def test_picks_latest_version_before_10am_day_before():
    raw = _raw(["2024-06-01T08:15:00", "2024-06-01T09:15:00", "2024-06-01T10:15:00"])
    out = pick_vintage(raw, pd.Timestamp("2024-06-02"))
    assert out["v"].tolist() == [1]          # 09:15 on D-1; the 10:15 version is too late


def test_falls_back_to_older_version_and_never_to_a_later_one():
    assert pick_vintage(_raw(["2024-05-31T23:00:00"]), pd.Timestamp("2024-06-02"))["v"].tolist() == [0]
    assert pick_vintage(_raw(["2024-06-01T10:00:00"]), pd.Timestamp("2024-06-02")).empty
