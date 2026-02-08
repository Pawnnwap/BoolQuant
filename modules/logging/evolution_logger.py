import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict

class EvolutionLogger:
    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = base_log_dir
        os.makedirs(base_log_dir, exist_ok=True)
        today = datetime.now()
        date_prefix = today.strftime("%m%d")
        run_count = self._get_daily_run_count(date_prefix)
        self.log_dir = os.path.join(
            base_log_dir, f"evolution_logs_{date_prefix}_{run_count:03d}"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "plots"), exist_ok=True)

        self.history_file = os.path.join(self.log_dir, "evolution_history.json")
        self.plots_dir = os.path.join(self.log_dir, "plots")

    def _get_daily_run_count(self, date_prefix: str) -> int:
        if not os.path.exists(self.base_log_dir):
            return 1

        count = 0
        for name in os.listdir(self.base_log_dir):
            if name.startswith(f"evolution_logs_{date_prefix}"):
                count += 1
        return count + 1

    def log_result(
        self,
        condition: str,
        score: float,
        total_return: float,
        daily_pnl: pd.Series,
        generation: int = 0,
        attempt: int = 0,
        full_stats: dict = None,
    ) -> Dict:
        train_stats = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "attempt": attempt,
            "condition": condition[:200] + "..." if len(condition) > 200 else condition,
            "full_condition": condition,
            "score": score,
            "total_return": total_return,
            "sharpe": score,
            "calmar": full_stats.get("calmar", 0) if full_stats else 0,
            "max_drawdown": full_stats.get("max_drawdown", 0) if full_stats else 0,
            "win_rate": (
                (daily_pnl[daily_pnl != 0].dropna() > 0).mean()
                if not daily_pnl.empty
                else 0
            ),
            "avg_win": daily_pnl[daily_pnl > 0].mean() if (daily_pnl > 0).any() else 0,
            "avg_loss": daily_pnl[daily_pnl < 0].mean() if (daily_pnl < 0).any() else 0,
            "trading_days": full_stats.get("train_trading_days", 0)
            if full_stats
            else 0,
        }

        result = {
            "train": train_stats,
        }

        if full_stats:
            test_stats = {}
            for key, value in full_stats.items():
                if key.startswith("test_"):
                    test_stats[key[5:]] = value
            result["test"] = test_stats

        history = self.load_history()
        history.append(result)
        self.save_history(history)

        return result

    def load_history(self) -> List[Dict]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading history: {e}")
        return []

    def save_history(self, history: List[Dict]):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def get_best_conditions(self, top_n: int = 5) -> List[Dict]:
        history = self.load_history()
        if not history:
            return []

        sorted_history = sorted(
            history, key=lambda x: x["train"]["score"], reverse=True
        )
        return sorted_history[:top_n]
