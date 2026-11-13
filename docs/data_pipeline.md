## Data Pipeline

1. [ERCOT API explorer](https://apiexplorer.ercot.com/)
2. Data Access Portal - Go to Data Access Portal
3. Search

    If no reaction for search, reload the page.

   1. `DAM Settlement Point Prices`
   2. `Historical RTM Load Zone and Hub Prices`

## Structure of Data

### Historical RTM Load Zone and Hub Prices（RPT）

`Delivery Date`

`Delivery Hour`

`Delivery Interval`

`Repeated Hour Flag` (是否為夏令時間重複小時)

`Settlement Point Name`: choose `HB_WEST` and `HB_NORTH`

`Settlement Point Type`

`Settlement Point Price`

### DAM Settlement Point Prices (CDR)

`Delivery Date`

`HourEnding`

`Settlement Point`

`SettlementPointPrice`

`DSTFlag` (夏令時間)

## Structure of the Folder

```bash
│           
├───data
│   ├───processed
│   └───raw
│       ├───DAM
│       └───RTM
└───src
```

## Importing Data

`src/data_import.py`

This script automatically downloads historical DAM and RTM data from the ERCOT API.

### Main Functions

1. **Authentication**
   - Authenticate with ERCOT using locally configured credentials.
   - Credentials are loaded through environment variables:
     - `ERCOT_USERNAME`
     - `ERCOT_PASSWORD`
     - `ERCOT_SUBSCRIPTION_KEY`

2. **Data Download**
   - Download DAM Settlement Point Prices.
   - Download Historical RTM Load Zone and Hub Prices.
   - Retrieve data for `HB_WEST` and `HB_NORTH`.
   - Specify the data period through `START_DATE` and `END_DATE`.

3. **Data Storage**
   - Save DAM data as CSV files in `data/raw/DAM/`.
   - Save RTM data as CSV files in `data/raw/RTM/`.
   - Use standardized filenames containing the settlement point and date range.

## Processing Data

`src/data_processing.py`

This script cleans, transforms, and integrates the raw DAM and RTM data to construct the hourly price spread dataset.

### Main Functions

1. **Load Raw Data**
   - Load DAM and RTM CSV files from `data/raw/DAM/` and `data/raw/RTM/`.
   - Process data for both `HB_WEST` and `HB_NORTH`.

2. **Clean and Validate Data**

   - **RTM:**
     - Remove exact duplicate records using `df.drop_duplicates()`.
     - Check duplicated timestamps and invalid delivery intervals.
     - Inspect missing 15-minute intervals.
     - Preserve missing values without artificial imputation.

   - **DAM:**
     - Inspect missing hourly prices.
     - Handle daylight saving time (DST) hours.
     - Preserve missing prices as `NaN`.

3. **RTM Aggregation**
   - Convert 15-minute RTM data into hourly data.
   - Calculate the arithmetic mean of the four 15-minute settlement prices.
   - Only calculate an hourly price when all four unique intervals are available.
   - Retain incomplete hours with `NaN` prices.

4. **DAM Normalization**
   - Transform DAM data from wide format (HE1–HE24) into long format.
   - Standardize the data structure to match the hourly RTM dataset.

5. **Data Integration**
   - Merge DAM and RTM hourly prices using:
     - `delivery_date`
     - `hour_ending`
     - `repeated_hour`
   - Use an outer join to preserve records with missing observations.

6. **Spread Calculation**

   Calculate the price spreads between HB_NORTH and HB_WEST:

   - **DA Spread:**

     `DA_spread = DAM_HB_NORTH - DAM_HB_WEST`

   - **RT Spread:**

     `RT_spread = RTM_HB_NORTH - RTM_HB_WEST`

7. **Data Quality Report**
   - Record the number of original and cleaned RTM observations.
   - Report removed duplicates, invalid intervals, and missing prices.
   - Summarize valid and missing hourly observations.

### Output

Processed datasets are saved in `data/processed/`.

| File | Description |
|---|---|
| `DAM_hourly_HB_WEST.csv` | Normalized hourly DAM prices for HB_WEST |
| `DAM_hourly_HB_NORTH.csv` | Normalized hourly DAM prices for HB_NORTH |
| `RTM_15min_clean_HB_WEST.csv` | Cleaned 15-minute RTM data for HB_WEST |
| `RTM_15min_clean_HB_NORTH.csv` | Cleaned 15-minute RTM data for HB_NORTH |
| `RTM_hourly_HB_WEST.csv` | Aggregated hourly RTM prices for HB_WEST |
| `RTM_hourly_HB_NORTH.csv` | Aggregated hourly RTM prices for HB_NORTH |
| `combined_hourly.csv` | Integrated DAM and RTM hourly prices |
| `spread_data.csv` | Final DA and RT price spread dataset |
| `data_quality_report.csv` | Data cleaning and validation summary |

## Processing Data

`src/data_processing.py`

1. Load Raw Data from `data/raw/DAM/` and `data/raw/RTM/`
2. Clean and Validate Data
    - RTM: remove the repeated records, `df.drop_duplicates()`; inspect missing intervals
    - DAM: inspect missing intervals, daylight saving time
3. RTM Aggregation: turning 15-minute Data into Hourly Data
4. Data Integration: `DAM hourly data` and `RTM hourly data`
5. Spread Calculation

Additional: a data quality report

### Workflow

```txt
                 ERCOT API
                     |
                     v
              data_import.py
                     |
          |----------|----------|
          |                     |
          v                     v
      Raw DAM Data          Raw RTM Data
          |                     |
          v                     v
   DAM Normalization       RTM Cleaning
          |                     |
          v                     v
   Clean DAM Hourly       RTM Aggregation
          |                     |
          |                     v
          |               Clean RTM Hourly
          |                     |
          |----------|----------|
                     |
                     v
              Data Integration
                     |
                     v
              Spread Calculation
                     |
                     v
              Processed Dataset
                     |
                     v
             Feature Engineering
```

Final structure
```
data/
│
├── raw/
│   ├── DAM/
│   └── RTM/
│
└── processed/
    │
    ├── DAM_hourly.csv
    │
    ├── RTM_15min_clean.csv
    │
    ├── RTM_hourly.csv
    │
    ├── combined_hourly.csv
    │
    └── spread_data.csv
```

## Subscription Key in Local

Local ERCOT credentials

In powershell
```powershell
$env:ERCOT_USERNAME = "your_username"
$env:ERCOT_PASSWORD = "your_password"
$env:ERCOT_SUBSCRIPTION_KEY = "your_subscription_key"
```