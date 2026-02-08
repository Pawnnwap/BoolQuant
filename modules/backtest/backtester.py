import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Callable, Optional

class Backtester:

    RETURN_TYPES = ["o2c", "o2o", "c2c", "c2o"]

    def __init__(
        self,
        data_preloader,
        return_type: str = "o2c",
        transaction_fee: float = 7/10_000,
        hard_mask_func: Callable = None,
        amount_rank_direction: str = None,
        amount_rank_n: int = None,
        amount_rank_d: int = None,
        train_test_ratio: float = 2.0,
        pnl_save_dir: str = None,
    ):
        self.data = data_preloader
        self.return_type = return_type
        self.transaction_fee = transaction_fee
        self.hard_mask_func = hard_mask_func
        self.amount_rank_direction = amount_rank_direction
        self.amount_rank_n = amount_rank_n
        self.amount_rank_d = amount_rank_d
        self.train_test_ratio = train_test_ratio
        self.pnl_save_dir = pnl_save_dir or "indicator_pnl"

        self._hard_mask_cache = None
        self._returns_cache = None
        self._amount_median_cache = None
        self._train_indices = None
        self._test_indices = None
        self._cutoff_date = None
        self._train_dates = None
        self._test_dates = None

    def setup_train_test_split(self):
        all_dates = self.data.all_dates
        if all_dates is None or len(all_dates) == 0:
            all_dates = sorted(self.data.close.index.unique())

        total_days = len(all_dates)
        train_ratio = self.train_test_ratio / (1 + self.train_test_ratio)
        train_days = int(total_days * train_ratio)
        test_days = total_days - train_days

        if test_days < 1:
            test_days = 1
            train_days = total_days - 1

        self._train_indices = slice(0, train_days)
        self._test_indices = slice(train_days, total_days)
        self._train_dates = all_dates[:train_days]
        self._test_dates = all_dates[train_days:]
        self._cutoff_date = all_dates[train_days]

        print(f"📊 Train/Test Split: {train_days} train days, {test_days} test days")
        print(f"📅 Cutoff date: {self._cutoff_date}")

    def _get_amount_median(self) -> pd.DataFrame:
        if self._amount_median_cache is not None:
            return self._amount_median_cache

        typical_price = (self.data.high + self.data.low + self.data.close) / 3
        amount = typical_price * self.data.volume
        self._amount_median_cache = amount.rolling(window=self.amount_rank_d).median()

        return self._amount_median_cache

    def _get_hard_mask(self) -> np.ndarray:
        if self._hard_mask_cache is None:
            if self.hard_mask_func is not None:
                mask = self.hard_mask_func(self.data)
            else:
                from modules.backtest.hard_masks import create_ashare_hard_mask
                mask = create_ashare_hard_mask(self.data)

            self._hard_mask_cache = np.asarray(mask, dtype=np.bool_)
        return self._hard_mask_cache

    def _get_returns(self) -> np.ndarray:
        if self._returns_cache is not None:
            return self._returns_cache

        close_arr = self.data.close.values
        open_arr = self.data.open.values

        returns = np.empty_like(close_arr, dtype=np.float64)

        close_next = np.roll(close_arr, -1, axis=0)
        close_next[-1, :] = np.nan

        open_next = np.roll(open_arr, -1, axis=0)
        open_next[-1, :] = np.nan

        if self.return_type == "o2c":
            ratio = close_next / open_arr
            returns = np.roll(ratio, -1, axis=0)
            returns[-1, :] = np.nan

        elif self.return_type == "o2o":
            ratio = open_next / open_arr
            returns = np.roll(ratio, -1, axis=0)
            returns[-1, :] = np.nan

        elif self.return_type == "c2c":
            ratio = close_next / close_arr
            returns = np.roll(ratio, -1, axis=0)
            returns[-1, :] = np.nan

        elif self.return_type == "c2o":
            ratio = open_next / close_arr
            returns = np.roll(ratio, -1, axis=0)
            returns[-1, :] = np.nan

        else:
            raise ValueError(f"Invalid return_type: {self.return_type}")

        returns -= 1 + self.transaction_fee
        np.clip(returns, -0.47, 0.47, out=returns)

        self._returns_cache = returns
        return self._returns_cache

    def run_backtest(
        self,
        condition_df: pd.DataFrame,
    ) -> Tuple[pd.Series, float, float, dict]:
        hard_mask = self._get_hard_mask()
        condition_bool = condition_df.values.astype(np.bool_, copy=False)
        final_mask = condition_bool & hard_mask

        returns = self._get_returns()

        if self.amount_rank_n is not None and self.amount_rank_d is not None:
            amount_median = self._get_amount_median()
            mask_with_rank = final_mask.copy()
            for i in range(len(mask_with_rank)):
                valid_indices = np.where(final_mask[i])[0]
                if len(valid_indices) > self.amount_rank_n:
                    amounts = amount_median.iloc[i].values
                    if self.amount_rank_direction == "top":
                        top_indices = valid_indices[
                            np.argsort(amounts[valid_indices])[-self.amount_rank_n:]
                        ]
                        mask_with_rank[i] = False
                        mask_with_rank[i, top_indices] = True
                    else:
                        bottom_indices = valid_indices[
                            np.argsort(amounts[valid_indices])[:self.amount_rank_n]
                        ]
                        mask_with_rank[i] = False
                        mask_with_rank[i, bottom_indices] = True
            selected_returns = np.where(mask_with_rank, returns, np.nan)
        else:
            selected_returns = np.where(final_mask, returns, np.nan)

        try:
            import bottleneck as bn
            daily_pnl_arr = bn.nanmean(selected_returns, axis=1)
        except ImportError:
            with np.errstate(invalid="ignore"):
                daily_pnl_arr = np.nanmean(selected_returns, axis=1)

        daily_pnl_arr = np.nan_to_num(daily_pnl_arr, nan=0.0)

        daily_pnl = pd.Series(daily_pnl_arr, index=self.data.close.index)

        train_pnl = daily_pnl.iloc[self._train_indices]
        test_pnl = daily_pnl.iloc[self._test_indices]

        train_stats = self._calculate_stats(train_pnl)
        test_stats = self._calculate_stats(test_pnl)

        full_stats = {
            **train_stats,
            **{f"test_{k}": v for k, v in test_stats.items()},
            "cutoff_date": str(self._cutoff_date) if self._cutoff_date else None,
            "train_trading_days": int((train_pnl.dropna() != 0).sum()),
            "test_trading_days": int((test_pnl.dropna() != 0).sum()),
        }

        score = train_stats["score"]
        total_return = train_stats["total_return"]

        return daily_pnl, score, total_return, full_stats

    def _calculate_stats(self, daily_pnl: pd.Series) -> dict:
        non_zero_mask = daily_pnl.values != 0.0
        count_non_zero = np.count_nonzero(non_zero_mask)

        std_val = np.std(daily_pnl.values, ddof=0)
        mean_val = np.mean(daily_pnl.values)

        threshold = 0.8 * len(daily_pnl)
        if count_non_zero > threshold and std_val > 1e-10:
            sharpe = mean_val / std_val * np.sqrt(252)
        else:
            sharpe = 0.0

        total_return = np.sum(daily_pnl.values)
        max_drawdown = (
            (daily_pnl.cumsum() - daily_pnl.cumsum().cummax()).min()
            if not daily_pnl.empty
            else 0
        )

        if max_drawdown != 0:
            ann_return = total_return / self._get_years(daily_pnl.index)
            calmar = abs(ann_return / max_drawdown)
        else:
            calmar = 0.0

        score = sharpe * calmar

        win_rate = (
            (daily_pnl[daily_pnl != 0].dropna() > 0).mean()
            if not daily_pnl.empty
            else 0
        )
        avg_win = daily_pnl[daily_pnl > 0].mean() if (daily_pnl > 0).any() else 0
        avg_loss = daily_pnl[daily_pnl < 0].mean() if (daily_pnl < 0).any() else 0

        return {
            "score": score,
            "sharpe": sharpe,
            "calmar": calmar,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "trading_days": int((daily_pnl.dropna() != 0).sum()),
        }

    def _get_years(self, date_index) -> float:
        if len(date_index) < 2:
            return 1.0 / 252
        start_date = date_index[0]
        end_date = date_index[-1]
        days = (end_date - start_date).days
        return max(days / 365.0, 1.0 / 365.0)

    def generate_pnl_plot(
        self,
        daily_pnl: pd.Series,
        title: str,
        save_path: str = None,
        show_cutoff: bool = True,
    ):
        fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

        train_pnl = daily_pnl.iloc[self._train_indices]
        test_pnl = daily_pnl.iloc[self._test_indices]

        cum_train = np.cumsum(train_pnl.values)
        cum_test = np.cumsum(test_pnl.values)

        ax.plot(
            self._train_dates,
            cum_train,
            linewidth=2.5,
            color="#2E86AB",
            label="Train Cumulative PnL",
            zorder=3,
        )

        if len(test_pnl) > 0:
            last_train_value = cum_train[-1] if len(cum_train) > 0 else 0
            cum_test_adjusted = last_train_value + cum_test
            ax.plot(
                self._test_dates,
                cum_test_adjusted,
                linewidth=2.5,
                color="#E74C3C",
                label="Test Cumulative PnL",
                zorder=3,
            )

        ax.axhline(
            y=0, color="#2C3E50", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2
        )

        all_cum = (
            np.concatenate([cum_train, cum_test]) if len(test_pnl) > 0 else cum_train
        )
        all_dates = self._train_dates + self._test_dates

        ax.fill_between(
            all_dates,
            0,
            np.concatenate(
                [cum_train, cum_test_adjusted if len(test_pnl) > 0 else cum_train]
            ),
            where=(
                np.concatenate(
                    [cum_train, cum_test_adjusted if len(test_pnl) > 0 else cum_train]
                )
                >= 0
            ),
            color="#27AE60",
            alpha=0.3,
            interpolate=True,
            zorder=1,
        )
        ax.fill_between(
            all_dates,
            0,
            np.concatenate(
                [cum_train, cum_test_adjusted if len(test_pnl) > 0 else cum_train]
            ),
            where=(
                np.concatenate(
                    [cum_train, cum_test_adjusted if len(test_pnl) > 0 else cum_train]
                )
                < 0
            ),
            color="#E74C3C",
            alpha=0.3,
            interpolate=True,
            zorder=1,
        )

        if show_cutoff and self._cutoff_date:
            ax.axvline(
                x=self._cutoff_date,
                color="#F39C12",
                linestyle="-",
                linewidth=2,
                alpha=0.8,
                zorder=2,
                label=f"Cutoff: {self._cutoff_date}",
            )

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color="#2C3E50")
        ax.set_xlabel("Date", fontsize=12, fontweight="bold", color="#2C3E50")
        ax.set_ylabel("Cumulative PnL", fontsize=12, fontweight="bold", color="#2C3E50")

        ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color="#95A5A6")
        ax.set_facecolor("#FAFBFC")

        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", labelsize=10)

        if save_path:
            fig.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close(fig)
        else:
            plt.show(fig)

        cum_returns = pd.Series(
            np.concatenate(
                [cum_train, cum_test_adjusted if len(test_pnl) > 0 else cum_train]
            ),
            index=all_dates,
        )
        return cum_returns
