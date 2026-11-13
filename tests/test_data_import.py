import pandas as pd
import pytest
import requests

import data_import
from data_import import fetch_month, month_ranges, to_long_dam, to_long_rtm, with_retry


def test_month_ranges_clips_to_start_and_end():
    assert month_ranges("2024-01-15", "2024-03-10") == [
        ("2024-01-15", "2024-01-31"),
        ("2024-02-01", "2024-02-29"),
        ("2024-03-01", "2024-03-10"),
    ]


def _http_error(status):
    r = requests.Response()
    r.status_code = status
    return requests.HTTPError(response=r)


def test_retry_waits_on_429_then_succeeds(monkeypatch):
    waits = []
    monkeypatch.setattr(data_import.time, "sleep", waits.append)
    calls = iter([_http_error(429), _http_error(429), "ok"])

    def call():
        x = next(calls)
        if isinstance(x, Exception):
            raise x
        return x

    assert with_retry(call) == "ok"
    assert waits == [60, 120]


def test_retry_does_not_swallow_other_errors(monkeypatch):
    monkeypatch.setattr(data_import.time, "sleep", lambda s: None)

    def call():
        raise _http_error(500)

    with pytest.raises(requests.HTTPError):
        with_retry(call)


def test_fetch_month_requests_one_hub_at_a_time(monkeypatch):
    monkeypatch.setattr(data_import, "HUBS", ["A", "B"])
    monkeypatch.setattr(data_import.time, "sleep", lambda s: None)
    seen = []

    def fake(point, first, last):
        seen.append(point)
        return pd.DataFrame({"settlement_point": [point], "price": [1.0]})

    out = fetch_month(fake, "2024-01-01", "2024-01-31")
    assert seen == ["A", "B"]
    assert sorted(out["settlement_point"]) == ["A", "B"]


def test_fetch_month_refuses_missing_hub(monkeypatch):
    monkeypatch.setattr(data_import, "HUBS", ["A", "B"])
    monkeypatch.setattr(data_import.time, "sleep", lambda s: None)

    def fake(point, first, last):
        return pd.DataFrame({"settlement_point": [point], "price": [1.0]}) if point == "A" else pd.DataFrame()

    with pytest.raises(RuntimeError, match="B"):
        fetch_month(fake, "2024-01-01", "2024-01-31")


def test_dam_long_format_and_he24():
    raw = pd.DataFrame({"deliveryDate": ["2024-01-01"], "hourEnding": ["24:00"], "settlementPoint": ["HB_WEST"],
                        "settlementPointPrice": ["12.5"], "DSTFlag": ["N"]})
    out = to_long_dam(raw)
    assert list(out.columns) == ["date", "settlement_point", "hour", "interval_ending", "price"]
    row = out.iloc[0]
    assert (row["hour"], row["price"]) == (24, 12.5)
    assert row["interval_ending"] == pd.Timestamp("2024-01-02 06:00", tz="UTC")


def test_rtm_long_format_and_repeated_dst_hour():
    raw = pd.DataFrame({"deliveryDate": ["2025-11-02"] * 2, "deliveryHour": [2, 2], "deliveryInterval": [1, 1],
                        "settlementPoint": ["HB_WEST"] * 2, "settlementPointPrice": [10.0, 11.0], "DSTFlag": ["N", "Y"]})
    out = to_long_rtm(raw)
    assert list(out.columns) == ["date", "settlement_point", "interval", "interval_ending", "price"]
    assert out["interval"].tolist() == [5, 5]
    assert out["interval_ending"].tolist() == [pd.Timestamp("2025-11-02 06:15", tz="UTC"),   # first 01:00-01:15 (CDT)
                                               pd.Timestamp("2025-11-02 07:15", tz="UTC")]   # repeated (CST)
