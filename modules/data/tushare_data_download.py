import time
import os
import argparse
from typing import List, Optional
import akshare as ak
import tushare as ts
from tqdm import tqdm
import glob
import json

def get_api_token():
    if os.path.exists(".temp_tushare_config.json"):
        with open(".temp_tushare_config.json", "r") as f:
            config = json.load(f)
        return config.get("tushare_api_token")
    return None

class TushareDataDownloader:
    def __init__(self, api_token: str, output_dir: str = "tushare_data"):
        self.api = ts.pro_api(api_token)
        self.output_dir = output_dir
        self.min_interval = 1.2
        os.makedirs(output_dir, exist_ok=True)

    def get_latest_downloaded_date(self) -> Optional[str]:
        parquet_files = glob.glob(os.path.join(self.output_dir, "*.parquet"))
        if not parquet_files:
            return None

        dates = [os.path.basename(f).replace(".parquet", "") for f in parquet_files]
        dates = [d for d in dates if d.isdigit() and len(d) == 8]
        return max(dates) if dates else None

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        trade_cal = ak.tool_trade_date_hist_sina()
        trade_cal["trade_date"] = (
            trade_cal["trade_date"].astype(str).str.replace("-", "")
        )

        filtered = trade_cal[
            (trade_cal["trade_date"] > start_date)
            & (trade_cal["trade_date"] <= end_date)
        ]
        return filtered["trade_date"].tolist()

    def fetch_daily_data(self, trade_date: str, max_retries: int = 3) -> bool:
        for attempt in range(max_retries):
            try:
                df = self.api.daily(trade_date=trade_date)
                if not df.empty:
                    filepath = os.path.join(self.output_dir, f"{trade_date}.parquet")
                    df.to_parquet(filepath)
                    return True
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\nFailed to fetch {trade_date} after {max_retries} attempts: {e}")
                    return False
                time.sleep(2**attempt)
        return False

    def download(self, end_date: str, start_date: Optional[str] = None) -> None:
        if start_date is None:
            start_date = self.get_latest_downloaded_date()
        if start_date is None:
            print("No existing data found. Please provide a start_date.")
            return
        print(f"Auto-detected start date: {start_date}")

        trade_dates = self.get_trading_dates(start_date, end_date)

        if not trade_dates:
            print(
                f"No trading dates found between {start_date} (exclusive) and {end_date} (inclusive)"
            )
            return

        print(
            f"Downloading {len(trade_dates)} trading days from {trade_dates[0]} to {trade_dates[-1]}..."
        )

        for trade_date in tqdm(trade_dates, desc="Fetching data"):
            start_time = time.time()
            success = self.fetch_daily_data(trade_date)
            elapsed = time.time() - start_time
            sleep_time = self.min_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"Download complete! Data saved to {self.output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Tushare data")
    parser.add_argument("--token", type=str, default=None, help="Tushare API token")
    parser.add_argument(
        "--start", type=str, default="20150101", help="Start date (YYYYMMDD)"
    )
    parser.add_argument(
        "--end", type=str, default="20990101", help="End date (YYYYMMDD)"
    )
    args = parser.parse_args()

    api_token = args.token if args.token else get_api_token()
    if not api_token:
        print(
            "Error: Tushare API token not found. Please provide it via --token or config file."
        )
        exit(1)

    downloader = TushareDataDownloader(api_token=api_token)
    downloader.download(start_date=args.start, end_date=args.end)
