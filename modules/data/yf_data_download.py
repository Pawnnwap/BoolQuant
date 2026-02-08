import pandas as pd
import yfinance as yf
import time
from io import StringIO
import os
import akshare as ak
from contextlib import redirect_stdout, redirect_stderr

DEFAULT_DATA_PATH = "df_main.parquet"

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

def fetch_all_ashare_codes():
    stock_codes = ak.stock_info_a_code_name()["code"]
    return [i for i in stock_codes if i[0] in ["0", "3", "6"]]

def get_latest_cached_date(data_path: str = DEFAULT_DATA_PATH) -> pd.Timestamp | None:
    if os.path.exists(data_path):
        df_existing = pd.read_parquet(data_path)
        if not df_existing.empty:
            return df_existing.index.max()
    return None

def update_yf_data(
    data_path: str = DEFAULT_DATA_PATH,
    batch_size: int = 200,
    force_full: bool = False,
) -> None:
    if force_full and os.path.exists(data_path):
        os.remove(data_path)
        print(f"Removed existing data at {data_path}")

    latest_date = get_latest_cached_date(data_path)

    if latest_date is not None:
        start_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Appending data starting from {start_date}")
    else:
        start_date = "2015-01-01"
        print("No existing data found. Starting fresh download.")

    if start_date > pd.Timestamp.now().strftime("%Y-%m-%d"):
        print("Data is already up to date.")
        return

    dfs = []
    tickers = fetch_all_ashare_codes()
    tickers = [
        ticker + ".SS" if ticker.startswith("6") else ticker + ".SZ"
        for ticker in tickers
    ]

    for batch in chunk_list(tickers, batch_size):
        while True:
            print(f"Downloading batch: {batch[0]} ... {batch[-1]}")
            stdout_buffer = StringIO()
            stderr_buffer = StringIO()
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                df_batch = yf.download(
                    tickers=batch, interval="1d", start=start_date, auto_adjust=False
                )

            captured_output = stdout_buffer.getvalue() + stderr_buffer.getvalue()
            if "too many requests" in captured_output.lower():
                print("YF rate limit message detected. Sleeping for 60 seconds.")
                time.sleep(60)
            else:
                break

        df_batch_clean = df_batch.dropna(axis=1, how="all")
        if not df_batch_clean.empty:
            dfs.append(df_batch_clean)
        time.sleep(3)

    if not dfs:
        print("No new data downloaded.")
        return

    df_main = pd.concat(dfs, axis=1)
    df_main.columns = pd.MultiIndex.from_tuples(
        [(x.lower(), y) for x, y in df_main.columns], names=df_main.columns.names
    )

    trading_calendar = ak.tool_trade_date_hist_sina()
    trading_calendar["trade_date"] = pd.to_datetime(trading_calendar["trade_date"])
    today = pd.Timestamp.now().normalize()
    trading_dates = trading_calendar[
        (trading_calendar["trade_date"] >= start_date) &
        (trading_calendar["trade_date"] <= today)
    ]["trade_date"]
    df_main = df_main.reindex(pd.DatetimeIndex(trading_dates))

    if latest_date is not None and os.path.exists(data_path):
        df_existing = pd.read_parquet(data_path)
        df_main = pd.concat([df_existing, df_main], axis=0).sort_index()

    df_main.to_parquet(data_path)
    print(f"Data saved to {data_path}")
    print(f"Date range: {df_main.index.min()} to {df_main.index.max()}")

if __name__ == "__main__":
    update_yf_data()
