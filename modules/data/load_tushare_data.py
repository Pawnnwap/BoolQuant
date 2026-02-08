import pandas as pd
import os
from glob import glob

def process_parquet_directory(
    data_directory: str,
    start_date: str,
    end_date: str,
    startswith: tuple = None
) -> pd.DataFrame:
    parquet_files = glob(os.path.join(data_directory, "*.parquet"))
    parquet_files = [
        f for f in parquet_files
        if (os.path.basename(f)[:8] >= start_date) and (os.path.basename(f)[:8] <= end_date)
    ]

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in '{data_directory}'")

    print(f"Loading {len(parquet_files)} parquet files...")

    dfs = [pd.read_parquet(f, engine='pyarrow') for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape: {df.shape}")

    before_filter = len(df)
    df = df[~df['ts_code'].str.endswith('.BJ')]
    if startswith:
        df = df[df['ts_code'].str.startswith(startswith)]
    after_filter = len(df)
    print(f"Filtered out {before_filter - after_filter} rows")

    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')

    dup_check = df.groupby(['trade_date', 'ts_code']).size()
    if (dup_check > 1).any():
        print(f"Warning: Found {dup_check.gt(1).sum()} duplicate trade_date / ts_code pairs")
        print("Keeping only the first occurrence")
        df = df.drop_duplicates(subset=['trade_date', 'ts_code'])

    df_pivot = df.pivot(index='trade_date', columns='ts_code')

    df_pivot = df_pivot.sort_index(axis=1, level=0)

    required_metrics = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']

    available_metrics = df_pivot.columns.get_level_values(0).unique()
    missing_metrics = set(required_metrics) - set(available_metrics)
    if missing_metrics:
        print(f"Warning: Requested metrics not found: {missing_metrics}")

    df_final = df_pivot.loc[:, df_pivot.columns.get_level_values(0).isin(required_metrics)]

    print(f"Final shape: {df_final.shape}")
    print(f"Stocks: {df_final.columns.get_level_values(1).nunique()}")
    print(f"Date range: {df_final.index.min()} to {df_final.index.max()}")

    df_final['vol'] *= 100
    df_final['amount'] *= 1000
    df_final = df_final.rename(columns={"vol": "volume"})

    return df_final
