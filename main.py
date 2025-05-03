from preprocessing import (
    load_data,
    handle_missing_data,
    remove_irrelevant_columns,
    normalize_quarters,
    add_quarterly_average_bin
)

DATA_PATH = 'data/dataset.csv'
QUARTER_COLS = ['2024 Quarter 1', '2024 Quarter 2', '2024 Quarter 3', '2024 Quarter 4']
REMOVE_COLS = ['Geolocation']

df = load_data(DATA_PATH)
df = handle_missing_data(df)
df = normalize_quarters(df, QUARTER_COLS)

print(df.head())

df.to_csv('data/processed_dataset.csv', index=False)
