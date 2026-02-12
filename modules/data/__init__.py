YF_DATA_PATH = "df_main.parquet"
TUSHARE_DATA_PATH = "tushare_data"


def get_default_data_path(source: str = "yfinance") -> str:
    if source == "tushare":
        return TUSHARE_DATA_PATH
    return YF_DATA_PATH


def __getattr__(name):
    if name == "update_yf_data":
        from .yf_data_download import update_yf_data

        return update_yf_data
    if name == "get_latest_cached_date":
        from .yf_data_download import get_latest_cached_date

        return get_latest_cached_date
    if name == "YF_DATA_PATH":
        return YF_DATA_PATH
    if name == "TUSHARE_DATA_PATH":
        return TUSHARE_DATA_PATH
    if name == "get_default_data_path":
        return get_default_data_path
    if name == "DEFAULT_DATA_PATH":
        return YF_DATA_PATH
    if name == "DataPreloader":
        from .data_preloader import DataPreloader

        return DataPreloader
    if name == "load_tushare_data":
        from .load_tushare_data import process_parquet_directory

        return process_parquet_directory
    if name == "TushareDataDownloader":
        from .tushare_data_download import TushareDataDownloader

        return TushareDataDownloader
    raise AttributeError(f"module 'modules.data' has no attribute '{name}'")
