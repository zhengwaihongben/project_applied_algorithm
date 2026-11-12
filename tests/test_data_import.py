from data_import import month_ranges


def test_month_ranges_clips_to_start_and_end():
    assert month_ranges("2024-01-15", "2024-03-10") == [
        ("2024-01-15", "2024-01-31"),
        ("2024-02-01", "2024-02-29"),
        ("2024-03-01", "2024-03-10"),
    ]
