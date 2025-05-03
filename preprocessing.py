import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    return pd.read_csv(path)

def handle_missing_data(df):
    return df.fillna(df.mean(numeric_only=True))

def remove_irrelevant_columns(df, columns_to_remove):
    return df.drop(columns=columns_to_remove)

def normalize_quarters(df, quarter_cols):
    df[quarter_cols] = df[quarter_cols].replace(r'[^0-9.]', '', regex=True)
    df[quarter_cols] = df[quarter_cols].apply(pd.to_numeric, errors='coerce')

    df[quarter_cols] = df[quarter_cols].fillna(df[quarter_cols].mean())

    scaler = MinMaxScaler()
    df[quarter_cols] = scaler.fit_transform(df[quarter_cols])
    return df

def add_quarterly_average_bin(df, quarter_cols):
    df['Quarterly Average'] = df[quarter_cols].mean(axis=1)
    df['AvgBin'] = pd.qcut(df['Quarterly Average'], q=3, labels=['Low', 'Medium', 'High'])
    return df
