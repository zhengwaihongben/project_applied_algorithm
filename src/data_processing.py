
from pathlib import Path
import pandas as pd


# ============================================================
# 1. Project paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DAM_DIR = PROJECT_ROOT / "data" / "raw" / "DAM"
RAW_RTM_DIR = PROJECT_ROOT / "data" / "raw" / "RTM"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

POINTS = ["HB_WEST", "HB_NORTH"]

RTM_INTERVALS_PER_HOUR = 4

TIME_KEYS = [
    "delivery_date",
    "hour_ending",
    "repeated_hour"
]


# ============================================================
# 2. Load raw data
# ============================================================

def load_raw_data():
    """
    Load raw DAM and RTM CSV files for both settlement points.
    """

    dam_data = {}
    rtm_data = {}

    for point in POINTS:

        dam_path = RAW_DAM_DIR / (
            f"DAM_{point}_2026-01-01_2026-05-31.csv"
        )

        rtm_path = RAW_RTM_DIR / (
            f"RTM_{point}_2026-01-01_2026-05-31.csv"
        )

        if not dam_path.exists():
            raise FileNotFoundError(dam_path)

        if not rtm_path.exists():
            raise FileNotFoundError(rtm_path)

        dam_data[point] = pd.read_csv(dam_path)
        rtm_data[point] = pd.read_csv(rtm_path)

        print(f"Loaded {point} DAM: {dam_data[point].shape}")
        print(f"Loaded {point} RTM: {rtm_data[point].shape}")

    return dam_data, rtm_data


# ============================================================
# 3. Clean RTM data
# ============================================================

def clean_rtm_data(df):
    """
    Clean raw 15-minute RTM data.

    - Remove exact duplicate rows.
    - Convert data types.
    - Identify duplicated time keys.
    - Preserve missing intervals; do not impute prices.
    """

    df = df.copy()

    original_count = len(df)

    # Remove exact duplicate records
    df = df.drop_duplicates().copy()

    exact_duplicates_removed = original_count - len(df)

    # Convert data types
    df["delivery_date"] = pd.to_datetime(
        df["delivery_date"]
    ).dt.date

    df["hour_ending"] = pd.to_numeric(
        df["hour_ending"], errors="coerce"
    )

    df["interval"] = pd.to_numeric(
        df["interval"], errors="coerce"
    )

    df["price"] = pd.to_numeric(
        df["price"], errors="coerce"
    )

    df["repeated_hour"] = (
        df["repeated_hour"]
        .astype(str)
        .str.upper()
        .isin(["TRUE", "Y", "YES", "1"])
    )

    # Identify duplicated time keys
    duplicate_keys = df[
        df.duplicated(
            subset=TIME_KEYS + ["interval"],
            keep=False
        )
    ].copy()

    # Identify invalid intervals
    invalid_intervals = df[
        ~df["interval"].isin([1, 2, 3, 4])
    ].copy()

    report = {
        "original_rows": original_count,
        "rows_after_cleaning": len(df),
        "exact_duplicates_removed": exact_duplicates_removed,
        "duplicate_time_key_rows": len(duplicate_keys),
        "invalid_interval_rows": len(invalid_intervals),
        "missing_price_rows": int(df["price"].isna().sum())
    }

    return df, report, duplicate_keys, invalid_intervals


# ============================================================
# 4. Aggregate RTM from 15-minute to hourly
# ============================================================

def aggregate_rtm_hourly(df):
    """
    Aggregate 15-minute RTM prices into hourly averages.

    An hourly average is calculated only when all four unique
    intervals are present and all four prices are available.

    Incomplete hours are retained with NaN prices.
    """

    df = df.copy()

    grouping_keys = TIME_KEYS + ["settlement_point"]

    hourly = (
        df.groupby(grouping_keys)
        .agg(
            price_mean=("price", "mean"),
            interval_count=("interval", "nunique"),
            row_count=("interval", "size"),
            valid_price_count=("price", "count")
        )
        .reset_index()
    )

    # Only accept hours with exactly four unique intervals
    # and four available prices.
    valid_hours = (
        (hourly["interval_count"] == RTM_INTERVALS_PER_HOUR)
        & (hourly["row_count"] == RTM_INTERVALS_PER_HOUR)
        & (hourly["valid_price_count"] == RTM_INTERVALS_PER_HOUR)
    )

    hourly["price"] = hourly["price_mean"].where(valid_hours)

    hourly = hourly.drop(columns=["price_mean"])

    return hourly


# ============================================================
# 5. Save processed data
# ============================================================

def save_rtm_outputs(point, cleaned, hourly):
    """
    Save cleaned 15-minute and aggregated hourly RTM data.
    """

    cleaned_path = PROCESSED_DIR / f"RTM_15min_clean_{point}.csv"
    hourly_path = PROCESSED_DIR / f"RTM_hourly_{point}.csv"

    cleaned.to_csv(cleaned_path, index=False)
    hourly.to_csv(hourly_path, index=False)

    print(f"Saved: {cleaned_path}")
    print(f"Saved: {hourly_path}")


# ============================================================
# 5. Normalize DAM data from wide to long format
# ============================================================

def normalize_dam_data(df):
    """
    Convert DAM data from wide format to long format.

    Original format:
        date | settlement_point | HE1 | HE2 | ... | HE24

    Output format:
        delivery_date | hour_ending | settlement_point | price
    """

    df = df.copy()

    he_columns = [
        col for col in df.columns
        if col.startswith("HE")
    ]

    if len(he_columns) != 24:
        raise ValueError(
            f"Expected 24 HE columns, found {len(he_columns)}"
        )

    # Convert wide format to long format
    long_df = df.melt(
        id_vars=["date", "settlement_point"],
        value_vars=he_columns,
        var_name="hour_ending",
        value_name="price"
    )

    # Convert data types
    long_df["delivery_date"] = pd.to_datetime(
        long_df["date"]
    ).dt.date

    long_df["hour_ending"] = (
        long_df["hour_ending"]
        .str.replace("HE", "", regex=False)
        .astype(int)
    )

    long_df["price"] = pd.to_numeric(
        long_df["price"], errors="coerce"
    )

    # No repeated hour occurs in the current Jan-May dataset.
    long_df["repeated_hour"] = False

    long_df = long_df.drop(columns=["date"])

    long_df = long_df[
        [
            "delivery_date",
            "hour_ending",
            "repeated_hour",
            "settlement_point",
            "price"
        ]
    ]

    return long_df


# ============================================================
# 6. Save DAM data
# ============================================================

def save_dam_outputs(point, dam_hourly):

    output_path = (
        PROCESSED_DIR / f"DAM_hourly_{point}.csv"
    )

    dam_hourly.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


# ============================================================
# 7. Combine DAM and RTM data
# ============================================================

def combine_market_data(dam_data, rtm_hourly_data):
    """
    Combine DAM and RTM hourly prices for both settlement points.

    Output:
        One row per delivery date and hour.
        Separate columns for DAM and RTM prices.
    """

    combined = None

    for point in POINTS:

        dam = dam_data[point].copy()
        rtm = rtm_hourly_data[point].copy()

        # Rename price columns
        dam = dam.rename(
            columns={"price": f"DAM_{point}"}
        )

        rtm = rtm.rename(
            columns={"price": f"RTM_{point}"}
        )

        # Keep only necessary columns
        dam = dam[
            TIME_KEYS + [f"DAM_{point}"]
        ]

        rtm = rtm[
            TIME_KEYS + [f"RTM_{point}"]
        ]

        # Check uniqueness before merging
        if dam.duplicated(TIME_KEYS).any():
            raise ValueError(
                f"Duplicate DAM timestamps found for {point}"
            )

        if rtm.duplicated(TIME_KEYS).any():
            raise ValueError(
                f"Duplicate RTM timestamps found for {point}"
            )

        # Merge DAM and RTM for this settlement point
        point_data = pd.merge(
            dam,
            rtm,
            on=TIME_KEYS,
            how="outer",
            validate="one_to_one"
        )

        # Merge with the other settlement point
        if combined is None:
            combined = point_data
        else:
            combined = pd.merge(
                combined,
                point_data,
                on=TIME_KEYS,
                how="outer",
                validate="one_to_one"
            )

    combined = combined.sort_values(
        TIME_KEYS
    ).reset_index(drop=True)

    return combined


# ============================================================
# 8. Calculate price spreads
# ============================================================

def calculate_spreads(combined):
    """
    Calculate DA and RT spreads.

    Spread convention:
        North - West
    """

    combined = combined.copy()

    combined["DA_spread"] = (
        combined["DAM_HB_NORTH"]
        - combined["DAM_HB_WEST"]
    )

    combined["RT_spread"] = (
        combined["RTM_HB_NORTH"]
        - combined["RTM_HB_WEST"]
    )

    return combined


# ============================================================
# 9. Save combined data
# ============================================================

def save_combined_outputs(combined):

    combined_path = (
        PROCESSED_DIR / "combined_hourly.csv"
    )

    spread_path = (
        PROCESSED_DIR / "spread_data.csv"
    )

    combined.to_csv(combined_path, index=False)

    spread_columns = (
        TIME_KEYS
        + [
            "DA_spread",
            "RT_spread"
        ]
    )

    combined[spread_columns].to_csv(
        spread_path,
        index=False
    )

    print(f"Saved: {combined_path}")
    print(f"Saved: {spread_path}")



# ============================================================
# 10. Main
# ============================================================

def main():

    # Load raw data
    dam_raw, rtm_raw = load_raw_data()

    quality_reports = {}

    dam_hourly_data = {}
    rtm_hourly_data = {}

    for point in POINTS:

        print(f"\nProcessing {point}")

        # ----------------------------------------------------
        # A. Process DAM
        # ----------------------------------------------------

        dam_hourly = normalize_dam_data(
            dam_raw[point]
        )

        save_dam_outputs(point, dam_hourly)

        dam_hourly_data[point] = dam_hourly

        # ----------------------------------------------------
        # B. Process RTM
        # ----------------------------------------------------

        cleaned, report, duplicate_keys, invalid_intervals = (
            clean_rtm_data(rtm_raw[point])
        )

        hourly = aggregate_rtm_hourly(cleaned)

        save_rtm_outputs(point, cleaned, hourly)

        rtm_hourly_data[point] = hourly

        quality_reports[point] = report

        # Save diagnostic records
        if not duplicate_keys.empty:
            duplicate_keys.to_csv(
                PROCESSED_DIR / f"RTM_duplicate_keys_{point}.csv",
                index=False
            )

        if not invalid_intervals.empty:
            invalid_intervals.to_csv(
                PROCESSED_DIR / f"RTM_invalid_intervals_{point}.csv",
                index=False
            )

        # Print quality report
        print("\nRTM Data Quality Report:")

        for key, value in report.items():
            print(f"{key}: {value}")

        print(f"RTM hourly records: {len(hourly)}")
        print(
            f"Valid RTM prices: "
            f"{hourly['price'].notna().sum()}"
        )

        # DAM quality information
        print(f"DAM hourly records: {len(dam_hourly)}")
        print(
            f"Missing DAM prices: "
            f"{dam_hourly['price'].isna().sum()}"
        )

    # --------------------------------------------------------
    # C. Combine DAM and RTM
    # --------------------------------------------------------

    print("\nCombining DAM and RTM data...")

    combined = combine_market_data(
        dam_hourly_data,
        rtm_hourly_data
    )

    # --------------------------------------------------------
    # D. Calculate spreads
    # --------------------------------------------------------

    spread_data = calculate_spreads(combined)

    # --------------------------------------------------------
    # E. Save outputs
    # --------------------------------------------------------

    save_combined_outputs(spread_data)

    # Save quality report
    report_df = pd.DataFrame(quality_reports).T

    report_df.to_csv(
        PROCESSED_DIR / "data_quality_report.csv"
    )

    # --------------------------------------------------------
    # F. Final summary
    # --------------------------------------------------------

    print("\nProcessing completed.")

    print(f"Combined hourly records: {len(spread_data)}")

    print(
        f"Valid DA spreads: "
        f"{spread_data['DA_spread'].notna().sum()}"
    )

    print(
        f"Valid RT spreads: "
        f"{spread_data['RT_spread'].notna().sum()}"
    )

    print(
        f"Missing DA spreads: "
        f"{spread_data['DA_spread'].isna().sum()}"
    )

    print(
        f"Missing RT spreads: "
        f"{spread_data['RT_spread'].isna().sum()}"
    )


if __name__ == "__main__":
    main()
