
"""
data_ingestion.py

Purpose:
    Batch-identify and rename raw ERCOT DAM and RTM data files.

File naming conventions:
    DAM: DAM_YYYY-MM-DD.csv
    RTM: RTM_YYYY.xlsx

Expected project structure:
    Project/
    ├── data/
    │   └── raw/
    │       ├── DAM/
    │       └── RTM/
    └── src/
        └── data_ingestion.py
"""

from pathlib import Path
import re
import argparse


# ============================================================
# 1. Define project directories
# ============================================================

# Get the project root directory (the parent of the src folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the raw data directories
DAM_DIR = PROJECT_ROOT / "data" / "raw" / "DAM"
RTM_DIR = PROJECT_ROOT / "data" / "raw" / "RTM"


# ============================================================
# 2. Extract date and year from original filenames
# ============================================================

def extract_dam_date(filename):
    """
    Extract the delivery date from a DAM filename.

    Expected date format in the original filename:
        YYYYMMDD

    Example:
        cdr.00012331.0000000000000000.20261111.123301.DAMSPNP4190.csv

    Returns:
        A date string in YYYY-MM-DD format.
        Returns None if no valid date is found.
    """

    match = re.search(r"(?<!\d)(20\d{6})(?!\d)", filename)

    if match is None:
        return None

    date_str = match.group(1)

    # Convert YYYYMMDD into YYYY-MM-DD
    formatted_date = (
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    )

    # Validate the date
    from datetime import datetime

    try:
        datetime.strptime(formatted_date, "%Y-%m-%d")
    except ValueError:
        return None

    return formatted_date


def extract_rtm_year(filename):
    """
    Extract the year from an RTM filename.

    Example:
        rpt.00013061.0000000000000000.RTMLZHBSPP_2026.xlsx

    Returns:
        A year string in YYYY format.
        Returns None if no year is found.
    """

    match = re.search(r"(20\d{2})(?!\d)", filename)

    if match is None:
        return None

    return match.group(1)


# ============================================================
# 3. Generate new filenames
# ============================================================

def generate_new_filename(file_path, market_type):
    """
    Generate a standardized filename based on the market type.

    DAM:
        DAM_YYYY-MM-DD.csv

    RTM:
        RTM_YYYY.xlsx

    Returns:
        The new filename.
        Returns None if the original filename is invalid.
    """

    filename = file_path.name

    if market_type == "DAM":

        # Extract the delivery date
        date = extract_dam_date(filename)

        if date is None:
            return None

        return f"DAM_{date}{file_path.suffix.lower()}"

    elif market_type == "RTM":

        # Extract the year
        year = extract_rtm_year(filename)

        if year is None:
            return None

        return f"RTM_{year}{file_path.suffix.lower()}"

    return None


# ============================================================
# 4. Process files in a directory
# ============================================================

def rename_files(directory, market_type, apply_changes=False):
    """
    Scan a directory and rename all recognized ERCOT files.

    Args:
        directory:
            The directory containing raw data files.

        market_type:
            Either "DAM" or "RTM".

        apply_changes:
            If False, only display the proposed changes.
            If True, actually rename the files.
    """

    print(f"\nProcessing {market_type} files...")
    print(f"Directory: {directory}")

    # Check whether the directory exists
    if not directory.exists():
        print("Warning: Directory does not exist.")
        return

    # Track the number of processed files
    renamed_count = 0
    skipped_count = 0
    error_count = 0

    # Track target filenames to detect duplicate plans
    planned_targets = set()

    # Scan all files in the directory
    for file_path in sorted(directory.iterdir()):

        # Ignore subdirectories
        if not file_path.is_file():
            continue

        filename = file_path.name.lower()

        # Identify files by their original ERCOT prefixes
        if market_type == "DAM" and not filename.startswith("cdr"):
            continue

        if market_type == "RTM" and not filename.startswith("rpt"):
            continue

        # Generate the standardized filename
        new_filename = generate_new_filename(file_path, market_type)

        if new_filename is None:
            print(f"[SKIP] Cannot identify date/year: {file_path.name}")
            skipped_count += 1
            continue

        new_path = directory / new_filename

        # Skip files that already have the standardized filename
        if file_path.name == new_filename:
            print(f"[SKIP] Already standardized: {file_path.name}")
            skipped_count += 1
            continue

        # Avoid overwriting existing files or duplicate rename targets
        if new_path.exists() or new_filename in planned_targets:
            print(f"[SKIP] Target already exists: {new_filename}")
            skipped_count += 1
            continue

        planned_targets.add(new_filename)

        # Display the proposed operation
        print(f"{file_path.name}  ->  {new_filename}")

        if apply_changes:
            try:
                file_path.rename(new_path)
                print("[RENAMED] Successfully renamed.")
                renamed_count += 1

            except OSError as error:
                print(f"[ERROR] {error}")
                error_count += 1

        else:
            renamed_count += 1

    # Display the processing summary
    print(f"\n{market_type} Summary")
    print("--------------------")
    print(f"Files to rename / renamed: {renamed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")


# ============================================================
# 5. Main function
# ============================================================

def main():
    """
    Main entry point of the data ingestion program.
    """

    parser = argparse.ArgumentParser(
        description="Batch rename ERCOT DAM and RTM raw data files."
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, only preview changes."
    )

    args = parser.parse_args()

    # Determine whether to preview or apply changes
    apply_changes = args.apply

    if apply_changes:
        print("MODE: APPLY CHANGES")
    else:
        print("MODE: DRY RUN (Preview only)")

    # Process DAM files
    rename_files(
        directory=DAM_DIR,
        market_type="DAM",
        apply_changes=apply_changes
    )

    # Process RTM files
    rename_files(
        directory=RTM_DIR,
        market_type="RTM",
        apply_changes=apply_changes
    )

    print("\nData ingestion completed.")


# ============================================================
# 6. Run the program
# ============================================================

if __name__ == "__main__":
    main()
