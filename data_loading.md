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

By `src/data_imports.py`. It imports the data documents and save in `../data/raw/DAM` and `../data/raw/RTM` respectively.

## Processing Data

Two codes(under `src/`):
1. `data_ingestion.py`: import and rename the data

    Importing the data to `data/raw/DAM` and `data/raw/RTM` respectively. `DAM` is for Day-Ahead Market, from `DAM Settlement Point Prices`; `RTM` is for Day-Ahead Market, from `Historical RTM Load Zone and Hub Prices`.

    Rename the data to `DAM_date.csv` and `RTM_data.csv`(or `.xlsx`).

2. `data_processing.py`: process the data

    Extracting desired columns, and settlement points: `HB_NORTH` and `HB_WEST`.

    Save the processed data by new documents and save them in `data/processed/`.

