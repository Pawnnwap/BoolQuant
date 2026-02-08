import argparse
import traceback
import shutil
import os
import sys

from modules.engine.trading_engine import TradingEvolutionEngine
from modules.backtest.hard_masks import get_hard_mask_factory

SERVICE_CHOICES = ["ollama", "lmstudio"]

OLLAMA_MODELS = ["qwen3"]
LM_STUDIO_MODELS = [
    "qwen3-4b-instruct-2507-gemini-3-pro-preview-distill",
    "qwen3-30b-a3b-instruct-2507",
    "falcon-h1r-7b-i1",
]

DEFAULT_THINKER_MODEL = {
    "ollama": "qwen3",
    "lmstudio": "qwen3-4b-instruct-2507-gemini-3-pro-preview-distill",
}

DEFAULT_SUMMARIZER_MODEL = {
    "ollama": "qwen3",
    "lmstudio": "qwen3-4b-instruct-2507-gemini-3-pro-preview-distill",
}

def get_available_models(service: str) -> list:
    if service == "ollama":
        return OLLAMA_MODELS
    elif service == "lmstudio":
        return LM_STUDIO_MODELS
    return []

def run_once(args):
    print("🚀 Starting Trading Evolution Engine...")
    print(f"📂 Data: {args.data_source}")
    print(f"📁 Log base dir: {args.base_log_dir}")
    print(f"🏪 Market type: {args.market_type}")
    print(f"🔢 Stock prefix: {args.prefix if args.prefix else 'All'}")
    print(f"🤖 Thinker Model: {args.thinker_model}")
    print(f"🤖 Summarizer Model: {args.summarizer_model}")
    print(f"📊 Return Type: {args.return_type}")
    print(f"💰 Transaction Fee: {args.transaction_fee:.5f}")
    if args.amount_rank_direction:
        print(
            f"📊 Amount Ranker: {args.amount_rank_direction} {args.amount_rank_n} stocks (D={args.amount_rank_d})"
        )

    hard_mask_func = get_hard_mask_factory(args.market_type, prefix=args.prefix)

    try:
        engine = TradingEvolutionEngine(
            service=args.service,
            thinker_model=args.thinker_model,
            summarizer_model=args.summarizer_model,
            return_type=args.return_type,
            transaction_fee=args.transaction_fee,
            base_log_dir=args.base_log_dir,
            hard_mask_func=hard_mask_func,
            market_type=args.market_type,
            amount_rank_direction=args.amount_rank_direction,
            amount_rank_n=args.amount_rank_n,
            amount_rank_d=args.amount_rank_d,
            start_date=args.start_date,
            end_date=args.end_date,
            train_test_ratio=args.train_test_ratio,
            data_source=args.data_source,
        )

        best_conditions = engine.run_evolution(args.generations, args.attempts)

        print("\n✅ Evolution completed successfully!")
        if best_conditions:
            print(f"📈 Best score achieved: {best_conditions[0]['train']['score']:.4f}")
            print(f"🎯 Best condition: {best_conditions[0]['train']['full_condition']}")

        return best_conditions

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description="Trading Evolution Engine")

    parser.add_argument(
        "--service",
        type=str,
        default="lmstudio",
        choices=SERVICE_CHOICES,
        help="LLM service to use (ollama, lmstudio)",
    )

    parser.add_argument(
        "--generations", type=int, default=2, help="Number of generations"
    )

    parser.add_argument(
        "--attempts", type=int, default=5, help="Attempts per generation"
    )

    parser.add_argument(
        "--return-type",
        type=str,
        default="o2c",
        choices=["o2c", "o2o", "c2c", "c2o"],
        help="Return type: o2c, o2o, c2c, c2o",
    )

    parser.add_argument(
        "--transaction-fee",
        type=float,
        default=7 / 10_000,
        help="Transaction fee as fraction (e.g., 0.0007 for 0.07%%)",
    )

    parser.add_argument(
        "--runs", type=int, default=1000, help="Number of independent runs"
    )

    parser.add_argument(
        "--base-log-dir",
        type=str,
        default="logs_ashare",
        help="Base directory for logs",
    )

    parser.add_argument(
        "--market-type",
        type=str,
        default="ashare",
        choices=["ashare"],
        help="Market type for hard mask (ashare)",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Stock code prefix filter: 00 = Shenzhen mainboard, 60 = Shanghai mainboard, 30 = ChiNext, 68 = STAR Market",
    )

    parser.add_argument(
        "--clean", action="store_true", help="Clean logs directory before running"
    )

    parser.add_argument(
        "--amount-rank-direction",
        type=str,
        default=None,
        choices=["top", "bottom"],
        help="Amount ranker direction: top (highest amount) or bottom (lowest amount)",
    )

    parser.add_argument(
        "--amount-rank-n",
        type=int,
        default=None,
        help="Number of stocks to select based on amount ranking",
    )

    parser.add_argument(
        "--amount-rank-d",
        type=int,
        default=None,
        help="Rolling window days D to calculate median amount",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date for data filtering (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date for data filtering (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--train-test-ratio",
        type=float,
        default=2.0,
        help="Train / test ratio (len(train_days)/len(test_days))",
    )

    parser.add_argument(
        "--data-source",
        type=str,
        default="yfinance",
        choices=["yfinance", "tushare"],
        help="Data source to use (yfinance or tushare)",
    )

    parser.add_argument(
        "--tushare-api-token",
        type=str,
        default=None,
        help="Tushare API token (required if data-source is tushare)",
    )

    args, unknown = parser.parse_known_args()

    available_models = get_available_models(args.service)
    default_thinker = DEFAULT_THINKER_MODEL.get(args.service, "qwen3")
    default_summarizer = DEFAULT_SUMMARIZER_MODEL.get(
        args.service, "qwen3-4b-instruct-2507-gemini-3-pro-preview-distill"
    )

    parser.add_argument(
        "--thinker-model",
        type=str,
        default=default_thinker,
        help=f"Thinker model name (recommended for {args.service}: "
        + ", ".join(available_models)
        + ")",
    )
    parser.add_argument(
        "--summarizer-model",
        type=str,
        default=default_summarizer,
        help=f"Summarizer model name (recommended for {args.service}: "
        + ", ".join(available_models)
        + ")",
    )

    args = parser.parse_args()

    if args.clean and os.path.exists(args.base_log_dir):
        shutil.rmtree(args.base_log_dir)

    for run_idx in range(1, args.runs + 1):
        print(f"\n🔄 Running evolution {run_idx}/{args.runs}")
        run_once(args)

    print(f"\n{'=' * 80}")
    print(f"ALL {args.runs} RUNS COMPLETED!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
