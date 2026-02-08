import pandas as pd
import re
from typing import Dict
from modules.evaluation.condition_evaluator import SafeConditionEvaluator
from modules.data.data_preloader import DataPreloader
from modules.backtest.hard_masks import get_hard_mask_factory

class StockPoolCalculator:
    def __init__(self, data_preloader=None):
        self.data = data_preloader
        self.evaluator = (
            SafeConditionEvaluator(data_preloader) if data_preloader else None
        )

    def _transform_condition(self, condition: str) -> str:
        transformations = [
            (r"\brolling_mean\s*\(\s*(\d+)\s*\)\b", r"rolling(\1).mean()"),
            (r"\bma\s*\(\s*(\d+)\s*\)\b", r"rolling(\1).mean()"),
            (r"\bmoving_avg\s*\(\s*(\d+)\s*\)\b", r"rolling(\1).mean()"),
            (r"\bsma\s*\(\s*(\d+)\s*\)\b", r"rolling(\1).mean()"),
            (r"\brolling_std\s*\(\s*(\d+)\s*\)\b", r"rolling(\1).std()"),
            (r"\brolling_min\s*\(\s*(\d+)\s*\)\b", r"rolling(\1).min()"),
            (r"\brolling_max\s*\(\s*(\d+)\s*\)\b", r"rolling(\1).max()"),
            (r"\brolling_sum\s*\(\s*(\d+)\s*\)\b", r"rolling(\1).sum()"),
            (r"\bema\s*\(\s*(\d+)\s*\)\b", r"ewm(span=\1).mean()"),
            (r"\bexponential_ma\s*\(\s*(\d+)\s*\)\b", r"ewm(span=\1).mean()"),
        ]
        transformed = condition
        for pattern, replacement in transformations:
            transformed = re.sub(pattern, replacement, transformed)
        return transformed

    def load_data(self, data_path: str = "df_main.parquet"):
        preloader = DataPreloader(data_path=data_path)
        preloader.load_and_preprocess()
        self.data = preloader
        self.evaluator = SafeConditionEvaluator(preloader)
        return self

    def calculate_pool(
        self,
        condition: str,
        target_date: str = None,
        prefix: str = None,
        amount_rank_direction: str = "top",
        amount_rank_n: int = 50,
        amount_rank_d: int = 20,
    ) -> Dict:
        if self.data is None or self.evaluator is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if target_date:
            if target_date not in self.data.all_dates:

                raise ValueError(f"Date {target_date} not found in data.")
            target_date_val = target_date
        else:

            target_date_val = self.data.all_dates[-1]

        condition = self._transform_condition(condition)
        result = self.evaluator.evaluate_condition(condition)

        if isinstance(result, pd.DataFrame):
            try:
                day_series = result.loc[target_date_val]
            except KeyError:
                raise ValueError(f"Condition result missing for date {target_date_val}")
        else:
            raise ValueError("Evaluator returned unexpected format.")

        selected_stocks = day_series[day_series == True].index.tolist()

        if prefix:
            try:
                factory = get_hard_mask_factory(
                    "ashare", prefix, restrict_tomorrow_open=False
                )
                mask_3d = factory(self.data)

                date_idx = self.data.all_dates.index(target_date_val)
                hard_mask_row = mask_3d[date_idx]

                valid_stocks = []
                for stock in selected_stocks:
                    if stock in self.data.close.columns:
                        col_idx = self.data.close.columns.get_loc(stock)
                        if hard_mask_row[col_idx]:
                            valid_stocks.append(stock)
            except Exception as e:
                print(f"Warning: Hard mask application failed: {e}")
                valid_stocks = selected_stocks
        else:
            valid_stocks = selected_stocks

        if len(valid_stocks) > amount_rank_n:
            try:
                typical_price = (self.data.high + self.data.low + self.data.close) / 3
                amount = typical_price * self.data.volume
                amount_median = amount.rolling(window=amount_rank_d).median()

                amount_medians = amount_median.loc[target_date_val]

                stock_amounts = []
                for stock in valid_stocks:
                    if stock in amount_medians.index:
                        stock_amounts.append((stock, amount_medians[stock]))

                if amount_rank_direction == "top":
                    stock_amounts.sort(key=lambda x: x[1], reverse=True)
                else:
                    stock_amounts.sort(key=lambda x: x[1])
                valid_stocks = [s[0] for s in stock_amounts[:amount_rank_n]]
            except Exception:
                pass

        stock_data = {}
        for stock in valid_stocks:
            if stock in self.data.close.columns:
                stock_data[stock] = {
                    "open": float(self.data.open.loc[target_date_val, stock]),
                    "high": float(self.data.high.loc[target_date_val, stock]),
                    "low": float(self.data.low.loc[target_date_val, stock]),
                    "close": float(self.data.close.loc[target_date_val, stock]),
                    "volume": int(self.data.volume.loc[target_date_val, stock]),
                }

        return {
            "date": str(target_date_val),
            "total_selected": len(valid_stocks),
            "stocks": stock_data,
        }

def calculate_stock_pool(
    condition: str,
    target_date: str = None,
    data_path: str = "df_main.parquet",
    prefix: str = None,
    amount_rank_direction: str = "top",
    amount_rank_n: int = 50,
    amount_rank_d: int = 20,
) -> Dict:
    calculator = StockPoolCalculator()
    calculator.load_data(data_path)

    return calculator.calculate_pool(
        condition=condition,
        target_date=target_date,
        prefix=prefix,
        amount_rank_direction=amount_rank_direction,
        amount_rank_n=amount_rank_n,
        amount_rank_d=amount_rank_d,
    )
