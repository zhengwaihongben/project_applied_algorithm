## Getting Data

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

`src/data_import.py`. It imports the data documents and save in `../data/raw/DAM` and `../data/raw/RTM` respectively.

## Processing Data

`src/data_processing.py`: process the data

1. Load Raw Data from `data/raw/DAM/` and `data/raw/RTM/`
2. Clean and Validate Data
    - 

### Workflow

```txt
          Raw DAM Data
               |
               v
       DAM Data Processing
               |
               v
       Clean DAM Hourly Data
               |
               |
               |------------------|
                                  |
                                  v
                           Data Integration
                                  |
                                  v
                           Spread Calculation
                                  |
                                  v
                           Processed Dataset
                                  ^
                                  |
               |------------------|
               |
       Clean RTM Hourly Data
               ^
               |
       RTM Data Processing
               ^
               |
          Raw RTM Data
```

## Subscription Key in Local

Local ERCOT credentials

In powershell
```powershell
$env:ERCOT_USERNAME = "your_username"
$env:ERCOT_PASSWORD = "your_password"
$env:ERCOT_SUBSCRIPTION_KEY = "your_subscription_key"
```