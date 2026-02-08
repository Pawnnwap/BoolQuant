import pandas as pd
import numpy as np
import os
from typing import Optional

class DataPreloader:

    def __init__(
        self,
        data_path: Optional[str] = None,
        market_type: str = None,
        start_date: str = None,
        end_date: str = None,
        data_source: str = "yfinance",
    ):
        from . import get_default_data_path

        self.data_source = data_source
        if data_source == "tushare":
            self.data_path = (
                data_path if data_path is not None else get_default_data_path("tushare")
            )
        else:
            self.data_path = (
                data_path
                if data_path is not None
                else get_default_data_path("yfinance")
            )
        self.market_type = market_type
        self.start_date = start_date
        self.end_date = end_date
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = None
        self.all_dates = None

    def load_and_preprocess(self):

        from .yf_data_download import update_yf_data
        from .load_tushare_data import process_parquet_directory

        print(f"📊 Preloading market data from {self.data_source}...")

        if self.data_source == "tushare":
            if not os.path.exists(self.data_path):
                from . import TushareDataDownloader

                print(
                    f"Tushare data not found at {self.data_path}. Please download Tushare data first."
                )
                raise FileNotFoundError(
                    f"Tushare data not found at {self.data_path}. Run tushare_data_download.py first."
                )

            start_str = (
                self.start_date.replace("-", "") if self.start_date else "20150101"
            )
            end_str = self.end_date.replace("-", "") if self.end_date else "20991231"

            df_main = process_parquet_directory(
                os.path.dirname(self.data_path), start_str, end_str
            )

        else:
            if not os.path.exists(self.data_path):
                print(
                    f"Data file not found at {self.data_path}. Starting auto-download..."
                )
                update_yf_data(data_path=self.data_path)

            df_main = pd.read_parquet(self.data_path).sort_index()

            if self.start_date:
                df_main = df_main[df_main.index >= self.start_date]
            if self.end_date:
                df_main = df_main[df_main.index <= self.end_date]

        self.open = df_main["open"].round(3)
        self.high = df_main["high"].round(3)
        self.low = df_main["low"].round(3)
        self.close = df_main["close"].round(3)
        self.volume = df_main["volume"].round()
        self.all_dates = sorted(df_main.index.unique())

        print(f"✅ Data loaded successfully. Shape: {self.close.shape}")
        print(f"📅 Date range: {self.close.index.min()} to {self.close.index.max()}")
        print(f"📅 Total trading days: {len(self.all_dates)}")
