import os
from typing import List, Dict, Callable

class TradingEvolutionEngine:

    def __init__(
        self,
        data_path: str = None,
        service: str = "lmstudio",
        thinker_model: str = None,
        summarizer_model: str = "qwen3",
        return_type: str = "o2c",
        transaction_fee: float = 7/10_000,
        base_log_dir: str = "logs",
        hard_mask_func: Callable = None,
        market_type: str = "ashare",
        amount_rank_direction: str = None,
        amount_rank_n: int = None,
        amount_rank_d: int = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        train_test_ratio: float = 2.0,
        data_source: str = "yfinance",
    ):
        from modules.data import get_default_data_path
        from modules.data.data_preloader import DataPreloader
        from modules.evaluation.condition_evaluator import SafeConditionEvaluator
        from modules.backtest.backtester import Backtester
        from modules.llm.llm_optimizer import LLMOptimizer
        from modules.logging.evolution_logger import EvolutionLogger

        self.data_path = (
            data_path if data_path is not None else get_default_data_path(data_source)
        )
        self.data_source = data_source
        self.summarizer_model = summarizer_model
        self.start_date = start_date
        self.end_date = end_date
        self.train_test_ratio = train_test_ratio

        self.data_preloader = DataPreloader(
            self.data_path,
            market_type=market_type,
            start_date=start_date,
            end_date=end_date,
            data_source=data_source,
        )
        self.condition_evaluator = SafeConditionEvaluator(self.data_preloader)
        self.backtester = Backtester(
            self.data_preloader,
            return_type,
            transaction_fee,
            hard_mask_func,
            amount_rank_direction,
            amount_rank_n,
            amount_rank_d,
            train_test_ratio=train_test_ratio,
            pnl_save_dir=os.path.join(base_log_dir, "indicator_pnl"),
        )
        self.evolution_logger = EvolutionLogger(base_log_dir=base_log_dir)
        self.llm_optimizer = LLMOptimizer(
            service,
            thinker_model,
            summarizer_model,
            self.evolution_logger,
            market_type=market_type,
            return_type=return_type,
        )

        self.data_preloader.load_and_preprocess()
        self.backtester.setup_train_test_split()

        self._daily_run_count = self._get_daily_run_count()

    def _get_daily_run_count(self) -> int:

        pnl_dir = self.backtester.pnl_save_dir
        if not os.path.exists(pnl_dir):
            return 1

        from datetime import datetime

        date_prefix = datetime.now().strftime("%m%d")
        count = 0
        for name in os.listdir(pnl_dir):
            if name.startswith(date_prefix):
                count += 1
        return count + 1

    def run_single_generation(self, generation: int, attempts: int = 3) -> List[Dict]:

        print(f"\n{'=' * 80}")
        print(f"🚀 GENERATION {generation} - {attempts} ATTEMPTS")
        print(f"{'=' * 80}")

        best_results = []
        previous_best = self.evolution_logger.get_best_conditions(3)

        for attempt in range(attempts):
            print(f"\n🎯 Attempt {attempt + 1}/{attempts}")

            try:
                condition_str, daily_pnl, score, total_return, full_stats = (
                    self.llm_optimizer.generate_condition_stepwise(
                        previous_best, self.condition_evaluator, self.backtester
                    )
                )

                if condition_str is None:
                    print(
                        "⚠️ Invalid generation (LHS=RHS or Score diff < 0.5), skipping to next attempt"
                    )
                    continue

                print(f"📋 Final condition: {condition_str}")
                print(
                    f"📊 Performance - Train Score: {score:.4f}, Train Return: {total_return:.2%}"
                )

                plot_path = os.path.join(
                    self.evolution_logger.plots_dir,
                    f"gen{generation}_att{attempt}_score{score:.3f}.png",
                )
                cond_display = (
                    condition_str[:60] + "..."
                    if len(condition_str) > 60
                    else condition_str
                )
                condition_name = f"Gen{generation}_Att{attempt}: {cond_display}"
                self.backtester.generate_pnl_plot(daily_pnl, condition_name, plot_path)

                os.makedirs(self.backtester.pnl_save_dir, exist_ok=True)
                from datetime import datetime

                date_prefix = datetime.now().strftime("%m%d")
                pnl_path = os.path.join(
                    self.backtester.pnl_save_dir,
                    f"{date_prefix}{self._daily_run_count:03d}_gen{generation}_att{attempt}.parquet",
                )
                daily_pnl.to_frame(name="daily_pnl").to_parquet(pnl_path)
                self._daily_run_count += 1

                result = self.evolution_logger.log_result(
                    condition=condition_str,
                    score=score,
                    total_return=total_return,
                    daily_pnl=daily_pnl,
                    generation=generation,
                    attempt=attempt,
                    full_stats=full_stats,
                )
                best_results.append(result)

            except Exception as e:
                print(f"❌ Error in attempt {attempt + 1}: {e}")
                print("⏭️ Skipping to next round...")
                continue

        return best_results

    def run_evolution(self, generations: int = 10, attempts_per_generation: int = 3):

        print(f"🎯 Starting Evolutionary Optimization")
        print(f"📊 Data shape: {self.data_preloader.close.shape}")
        print(f"🤖 Using model: {self.summarizer_model}")
        print(
            f"🔄 Generations: {generations}, Attempts per generation: {attempts_per_generation}"
        )
        print(f"🔒 Dimensional Constraint: +/- same-category only, */ free")

        all_results = []

        for generation in range(generations):
            results = self.run_single_generation(generation, attempts_per_generation)
            all_results.extend(results)

            best_so_far = self.evolution_logger.get_best_conditions(1)
            if best_so_far:
                best = best_so_far[0]
                print(
                    f"\n🏆 BEST SO FAR - Gen{best['train']['generation']}_Att{best['train']['attempt']}:"
                )
                print(
                    f"   Score: {best['train']['score']:.4f}, Return: {best['train']['total_return']:.2%}"
                )
                print(f"   Condition: {best['train']['full_condition']}")

        print(f"\n{'=' * 80}")
        print(f"🎉 EVOLUTION COMPLETE!")
        print(f"{'=' * 80}")

        best_conditions = self.evolution_logger.get_best_conditions(5)
        print(f"\n📊 TOP 5 CONDITIONS:")
        for i, cond in enumerate(best_conditions, 1):
            print(
                f"\n{i}. Gen{cond['train']['generation']}_Att{cond['train']['attempt']} (Score: {cond['train']['score']:.4f})"
            )
            print(
                f"   Return: {cond['train']['total_return']:.2%}, Sharpe: {cond['train']['sharpe']:.2f}"
            )
            print(f"   Condition: {cond['train']['full_condition']}")

        return best_conditions
