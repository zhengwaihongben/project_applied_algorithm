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


from covariates import calendar, to_hour_ending


def test_hour_ending_handles_he24_and_repeated_dst_hour():
    df = pd.DataFrame({
        "deliveryDate": ["2024-01-01", "2025-11-02", "2025-11-02", "2024-06-01"],
        "hourEnding": ["24:00", "02:00", "02:00", 1],     # wind/solar files use plain integers
        "DSTFlag": [False, False, True, False],
    })
    out = to_hour_ending(df).tolist()
    assert out[0] == pd.Timestamp("2024-01-02 06:00", tz="UTC")
    assert out[1] == pd.Timestamp("2025-11-02 07:00", tz="UTC")   # first 01:00-02:00 (CDT)
    assert out[2] == pd.Timestamp("2025-11-02 08:00", tz="UTC")   # repeated hour (CST)
    assert out[3] == pd.Timestamp("2024-06-01 06:00", tz="UTC")


def test_calendar_marks_local_hour_and_holiday():
    he = pd.Series(pd.to_datetime(["2024-07-04 06:00", "2024-07-05 06:00"], utc=True))  # HE1 local
    cal = calendar(he)
    assert cal["hour"].tolist() == [0, 0]
    assert cal["holiday"].tolist() == [1, 0]
    assert cal["dow"].tolist() == [3, 4]
