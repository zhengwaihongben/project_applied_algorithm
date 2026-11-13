
import os
import pandas as pd
import ercot as er

from pathlib import Path
from ercot.client import ErcotClient


# ============================================================
# 1. ERCOT API Configuration
# ============================================================

USERNAME = os.getenv("ERCOT_USERNAME")
PASSWORD = os.getenv("ERCOT_PASSWORD")
SUBSCRIPTION_KEY = os.getenv("ERCOT_SUBSCRIPTION_KEY")

if not all([USERNAME, PASSWORD, SUBSCRIPTION_KEY]):
    raise ValueError(
        "Missing ERCOT credentials. "
        "Please set ERCOT_USERNAME, ERCOT_PASSWORD, "
        "and ERCOT_SUBSCRIPTION_KEY as environment variables."
    )

er.configure(USERNAME, PASSWORD, SUBSCRIPTION_KEY)


# ============================================================
# 2. Project Configuration
# ============================================================

START_DATE = "2026-01-01"
END_DATE = "2026-05-31"

POINTS = [
    "HB_WEST",
    "HB_NORTH"
]

BASE_DIR = Path(__file__).resolve().parent.parent

DAM_DIR = BASE_DIR / "data" / "raw" / "DAM"
RTM_DIR = BASE_DIR / "data" / "raw" / "RTM"

DAM_DIR.mkdir(parents=True, exist_ok=True)
RTM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Fix ERCOT RTM Positional Column Mapping
# ============================================================

_original_assign = ErcotClient._assign_positional_names


def fixed_assign_positional_names(df):
    """
    Correctly identify ERCOT RTM's seven positional columns.

    Expected RTM columns:
        0: delivery_date
        1: hour_ending
        2: interval
        3: settlement_point
        4: settlement_point_type
        5: price
        6: repeated_hour
    """

    df = df.copy()

    # ERCOT RTM returns seven positional columns
    if list(df.columns) == [str(i) for i in range(7)]:

        df.columns = [
            "delivery_date",
            "hour_ending",
            "interval",
            "settlement_point",
            "settlement_point_type",
            "price",
            "repeated_hour"
        ]

        # Convert numeric columns
        df["hour_ending"] = pd.to_numeric(
            df["hour_ending"],
            errors="coerce"
        )

        df["interval"] = pd.to_numeric(
            df["interval"],
            errors="coerce"
        ).fillna(1).astype(int)

        df["price"] = pd.to_numeric(
            df["price"],
            errors="coerce"
        )

        # Convert repeated_hour to Boolean
        df["repeated_hour"] = (
            df["repeated_hour"]
            .astype(str)
            .str.upper()
            .isin(["TRUE", "Y", "YES", "1"])
        )

        return df

    # Keep the original behavior for other data formats
    return _original_assign(df)


ErcotClient._assign_positional_names = staticmethod(
    fixed_assign_positional_names
)


# ============================================================
# 4. Fix ERCOT RTM Formatting
# ============================================================

_original_format_spp = ErcotClient._format_spp


def fixed_format_spp(self, df, market, fmt):
    """
    Preserve normalized raw RTM data instead of converting
    it into the package's long-format representation.

    This is necessary because the package's RTM long-format
    conversion appears to produce incorrect interval_ending
    timestamps by not preserving hour_ending correctly.

    DAM continues to use the original formatting function.
    """

    normalized = self._normalize_columns(df)

    if market.upper() == "RTM":
        return normalized

    # Keep original formatting behavior for DAM
    return _original_format_spp(self, df, market, fmt)


ErcotClient._format_spp = fixed_format_spp


# ============================================================
# 5. Download and Save DAM Data
# ============================================================

def download_dam():

    print("\nStarting DAM download...")

    for point in POINTS:

        print(f"\nDownloading DAM data for {point}...")

        df_dam = er.dam(
            point,
            START_DATE,
            END_DATE
        )

        output_file = (
            DAM_DIR /
            f"DAM_{point}_{START_DATE}_{END_DATE}.csv"
        )

        df_dam.to_csv(
            output_file,
            index=False
        )

        print(f"Saved: {output_file}")
        print(f"Shape: {df_dam.shape}")

    print("\nDAM download completed.")


# ============================================================
# 6. Download and Save RTM Data
# ============================================================

def download_rtm():

    print("\nStarting RTM download...")

    for point in POINTS:

        print(f"\nDownloading RTM data for {point}...")

        df_rtm = er.rtm(
            [point],
            START_DATE,
            END_DATE,
            format="long"
        )

        # Save raw normalized RTM data
        # Note: format="long" is intentionally bypassed
        # by fixed_format_spp() above.

        output_file = (
            RTM_DIR /
            f"RTM_{point}_{START_DATE}_{END_DATE}.csv"
        )

        df_rtm.to_csv(
            output_file,
            index=False
        )

        print(f"Saved: {output_file}")
        print(f"Shape: {df_rtm.shape}")

        print("\nColumns:")
        print(df_rtm.columns.tolist())

        print("\nFirst 12 rows:")
        print(df_rtm.head(12).to_string(index=False))

    print("\nRTM download completed.")


# ============================================================
# 7. Validate RTM Data
# ============================================================

def validate_rtm():

    print("\nStarting RTM validation...")

    for point in POINTS:

        print(f"\nValidating {point}...")

        input_file = (
            RTM_DIR /
            f"RTM_{point}_{START_DATE}_{END_DATE}.csv"
        )

        df = pd.read_csv(input_file)

        df["delivery_date"] = pd.to_datetime(
            df["delivery_date"],
            errors="coerce"
        ).dt.date

        # 1. Basic validation
        print("\nDate range:")
        print(df["delivery_date"].min())
        print(df["delivery_date"].max())

        daily_counts = df.groupby("delivery_date").size()

        print("\nRecords per date:")
        print(daily_counts.describe())

        print("\nDates with record counts different from 96:")
        print(daily_counts[daily_counts != 96])

        print("\nUnique intervals:")
        print(sorted(df["interval"].dropna().unique()))

        print("\nUnique hour-ending values:")
        print(sorted(df["hour_ending"].dropna().unique()))

        print("\nRepeated-hour distribution:")
        print(df["repeated_hour"].value_counts(dropna=False))

        print("\nMissing values:")
        print(df.isnull().sum())

        # 2. Identify duplicate interval records
        duplicate_key = [
            "delivery_date",
            "hour_ending",
            "interval",
            "repeated_hour",
            "settlement_point"
        ]

        duplicates = df[
            df.duplicated(
                subset=duplicate_key,
                keep=False
            )
        ].sort_values(duplicate_key)

        print("\n========== DUPLICATE RECORDS ==========")

        if duplicates.empty:
            print("No duplicate records found.")
        else:
            print(duplicates.to_string(index=False))

        # 3. Check intervals in each delivery hour
        hourly_counts = (
            df.groupby(
                [
                    "delivery_date",
                    "hour_ending",
                    "repeated_hour"
                ]
            )
            .agg(
                record_count=("interval", "size"),
                unique_intervals=("interval", "nunique"),
                intervals=("interval", lambda x: sorted(x.tolist()))
            )
            .reset_index()
        )

        abnormal_hours = hourly_counts[
            (hourly_counts["record_count"] != 4) |
            (hourly_counts["unique_intervals"] != 4)
        ]

        print("\n========== ABNORMAL DELIVERY HOURS ==========")

        if abnormal_hours.empty:
            print("All delivery hours have exactly 4 intervals.")
        else:
            print(abnormal_hours.to_string(index=False))

        # 4. Identify missing intervals
        print("\n========== MISSING INTERVALS ==========")

        expected_intervals = {1, 2, 3, 4}

        for _, row in hourly_counts.iterrows():

            actual_intervals = set(row["intervals"])

            missing_intervals = (
                expected_intervals - actual_intervals
            )

            if missing_intervals:
                print(
                    f"Date: {row['delivery_date']}, "
                    f"Hour Ending: {row['hour_ending']}, "
                    f"Repeated Hour: {row['repeated_hour']}, "
                    f"Missing: {sorted(missing_intervals)}"
                )

        # 5. Check delivery hours per day
        print("\n========== DELIVERY HOURS PER DAY ==========")

        daily_hours = df.groupby("delivery_date")["hour_ending"].nunique()

        abnormal_days = daily_hours[daily_hours != 24]

        if abnormal_days.empty:
            print("All dates have exactly 24 delivery hours.")
        else:
            print("Dates with abnormal delivery hours:")
            print(abnormal_days.to_string())

# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ERCOT DATA DOWNLOAD")
    print("=" * 60)

    download_dam()

    download_rtm()

    validate_rtm()

    print("\nAll tasks completed.")