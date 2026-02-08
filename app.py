import streamlit as st
import subprocess
import sys
import os
import pandas as pd
import glob
import json

st.set_page_config(page_title="BoolQuant UI", layout="wide")

CONFIG_FILE = "config.json"


def load_params():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_params(params):
    with open(CONFIG_FILE, "w") as f:
        json.dump(params, f, indent=2)


def get_available_dates():
    if os.path.exists("df_main.parquet"):
        df = pd.read_parquet("df_main.parquet")
        dates = sorted(df.index.unique())
        return dates
    return []


saved_params = load_params()

if "evolution_process" not in st.session_state:
    st.session_state.evolution_process = None
if "log_dir" not in st.session_state:
    st.session_state.log_dir = None

_, col_center, _ = st.columns([1, 2, 1])
with col_center:
    st.image("assets/logo.png", width=300)

with st.sidebar:
    _, col_side, _ = st.columns([1, 2, 1])
    with col_side:
        st.image("assets/logo.png", width=100)

    st.header("📊 Stock Selection")
    prefix = st.text_input(
        "Stock Prefix",
        value=saved_params.get("prefix", ""),
        placeholder="60, 00, 30, 68 or leave empty for all",
    )
    market_type = st.selectbox("Market Type", ["ashare"], disabled=True)

    data_source = st.selectbox(
        "Data Source",
        ["yfinance", "tushare"],
        index=["yfinance", "tushare"].index(
            saved_params.get("data_source", "yfinance")
        ),
    )
    if data_source == "tushare":
        tushare_api_token = st.text_input(
            "Tushare API Token",
            type="password",
            value=saved_params.get("tushare_api_token", ""),
            help="Get your token from https://tushare.pro",
        )
    else:
        tushare_api_token = ""

    st.header("🤖 LLM Configuration")
    service = st.selectbox(
        "LLM Service",
        ["lmstudio", "ollama"],
        index=["lmstudio", "ollama"].index(saved_params.get("service", "lmstudio")),
    )
    thinker_model = st.text_input(
        "Thinker Model", value=saved_params.get("thinker_model", "qwen3")
    )
    summarizer_model = st.text_input(
        "Summarizer Model", value=saved_params.get("summarizer_model", "qwen3")
    )

    st.header("⚙️ Evolution Settings")
    return_type = st.selectbox(
        "Return Type",
        ["o2c", "o2o", "c2c", "c2o"],
        index=["o2c", "o2o", "c2c", "c2o"].index(
            saved_params.get("return_type", "o2c")
        ),
        help="o2c: Open-to-Close (intraday) | o2o: Open-to-Open (overnight) | c2c: Close-to-Close (full day) | c2o: Close-to-Open (overnight)",
    )
    generations = st.number_input(
        "Generations", min_value=1, value=saved_params.get("generations", 2)
    )
    attempts = st.number_input(
        "Attempts per Generation", min_value=1, value=saved_params.get("attempts", 5)
    )
    runs = st.number_input(
        "Number of Runs", min_value=1, value=saved_params.get("runs", 1)
    )
    transaction_fee = st.number_input(
        "Transaction Fee",
        min_value=0.0,
        value=saved_params.get("transaction_fee", 0.0007),
        format="%.5f",
    )
    clean = st.checkbox("Clean Logs Before Run", value=saved_params.get("clean", False))

    st.header("📁 Output Settings")
    base_log_dir = st.text_input(
        "Log Directory", value=saved_params.get("base_log_dir", "logs_ashare")
    )

    st.header("🔍 Data Filter")
    amount_rank_enabled = st.checkbox(
        "Enable Amount Ranker", value=saved_params.get("amount_rank_enabled", False)
    )
    col_rank1, col_rank2 = st.columns(2)
    with col_rank1:
        amount_rank_direction = st.selectbox(
            "Direction",
            ["top", "bottom"],
            index=["top", "bottom"].index(
                saved_params.get("amount_rank_direction", "top")
            ),
            disabled=not amount_rank_enabled,
        )
    with col_rank2:
        amount_rank_n = st.number_input(
            "N stocks",
            min_value=1,
            value=saved_params.get("amount_rank_n", 10),
            disabled=not amount_rank_enabled,
        )
    amount_rank_d = st.number_input(
        "D days (rolling window)",
        min_value=1,
        value=saved_params.get("amount_rank_d", 20),
        disabled=not amount_rank_enabled,
    )

    st.header("📊 Data Status")
    if os.path.exists("df_main.parquet"):
        df_check = pd.read_parquet("df_main.parquet")
        min_date = (
            df_check.index.min().date()
            if hasattr(df_check.index.min(), "date")
            else df_check.index.min()
        )
        max_date = (
            df_check.index.max().date()
            if hasattr(df_check.index.max(), "date")
            else df_check.index.max()
        )
        st.success(f"Data loaded: {min_date} to {max_date}")
        st.info(f"Stocks: {len(df_check.columns.get_level_values(1).unique())}")

        all_dates = sorted(df_check.index.unique())
        date_options = [d.date() if hasattr(d, "date") else d for d in all_dates]
    else:
        st.warning("No data found - will auto-download")
        date_options = []

    st.header("📅 Date Range")
    if date_options:
        min_d = min(date_options)
        max_d = max(date_options)
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input(
                "Start Date",
                value=saved_params.get("start_date", min_d),
                min_value=min_d,
                max_value=max_d,
                key="start_date_input",
            )
        with col_date2:
            end_date = st.date_input(
                "End Date",
                value=saved_params.get("end_date", max_d),
                min_value=min_d,
                max_value=max_d,
                key="end_date_input",
            )
    else:
        start_date = saved_params.get("start_date", "2020-01-01")
        end_date = saved_params.get("end_date", "2024-12-31")

    st.header("📊 Train / Test Split")
    train_test_ratio = st.number_input(
        "Train / Test Ratio",
        min_value=0.1,
        value=saved_params.get("train_test_ratio", 2.0),
        step=0.1,
        format="%.1f",
    )
    st.caption(
        f"Example: ratio={train_test_ratio:.1f} means {train_test_ratio:.1f}x more train days than test days"
    )

tab1, tab2, tab3 = st.tabs(["🚀 Run Evolution", "📈 Results", "🔍 Stock Selector"])

with tab1:
    st.subheader("🎯 Evolution Control")
    col_run, col_stop = st.columns(2)
    with col_run:
        run_btn = st.button("▶️ Run Evolution", type="primary")
    with col_stop:
        stop_btn = st.button("⏹️ Stop Evolution")

    if st.session_state.evolution_process is not None:
        proc = st.session_state.evolution_process
        if proc.poll() is None:
            st.info("🔄 Evolution in progress...")
        else:
            st.success("✅ Evolution completed!")
            st.session_state.evolution_process = None
            st.rerun()

    if stop_btn and st.session_state.evolution_process is not None:
        st.session_state.evolution_process.terminate()
        st.session_state.evolution_process = None
        st.warning("⏹️ Evolution stopped!")
        st.rerun()

    st.divider()

    if run_btn:
        params_to_save = {
            "prefix": prefix,
            "service": service,
            "thinker_model": thinker_model,
            "summarizer_model": summarizer_model,
            "return_type": return_type,
            "generations": generations,
            "attempts": attempts,
            "transaction_fee": transaction_fee,
            "runs": runs,
            "base_log_dir": base_log_dir,
            "clean": clean,
            "amount_rank_enabled": amount_rank_enabled,
            "amount_rank_direction": amount_rank_direction,
            "amount_rank_n": amount_rank_n,
            "amount_rank_d": amount_rank_d,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "train_test_ratio": train_test_ratio,
            "data_source": data_source,
            "tushare_api_token": tushare_api_token,
        }
        save_params(params_to_save)

        cmd = [sys.executable, "run_no_ui.py"]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        args = ["--prefix", prefix] if prefix else []
        args.extend(
            [
                "--service",
                service,
                "--thinker-model",
                thinker_model,
                "--summarizer-model",
                summarizer_model,
                "--return-type",
                return_type,
                "--generations",
                str(generations),
                "--attempts",
                str(attempts),
                "--transaction-fee",
                str(transaction_fee),
                "--runs",
                str(runs),
                "--base-log-dir",
                base_log_dir,
                "--start-date",
                str(start_date),
                "--end-date",
                str(end_date),
                "--train-test-ratio",
                str(train_test_ratio),
            ]
        )
        if amount_rank_enabled:
            args.extend(
                [
                    "--amount-rank-direction",
                    amount_rank_direction,
                    "--amount-rank-n",
                    str(amount_rank_n),
                    "--amount-rank-d",
                    str(amount_rank_d),
                ]
            )
        if clean:
            args.append("--clean")

        args.extend(
            [
                "--data-source",
                data_source,
            ]
        )
        if data_source == "tushare" and tushare_api_token:
            args.extend(
                [
                    "--tushare-api-token",
                    tushare_api_token,
                ]
            )

        full_cmd = [sys.executable, "run_no_ui.py"] + args

        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        phases = [
            "🧬 Initializing population...",
            "🧠 Thinker generating signals...",
            "📝 Summarizer analyzing context...",
            "📊 Evaluating fitness...",
            "🎯 Selecting best candidates...",
            "🔀 Applying mutations...",
            "⏳ Waiting for next generation...",
        ]

        for i, phase in enumerate(phases):
            status_placeholder.info(phase)
            progress_bar.progress((i + 1) / len(phases))
            import time

            time.sleep(0.3)

        progress_bar.empty()
        proc = subprocess.Popen(
            full_cmd,
            env=env,
        )
        st.session_state.evolution_process = proc
        status_placeholder.success("🚀 Evolution engine started! PID: " + str(proc.pid))

with tab2:
    st.subheader("📈 Evolution Results")

    evolution_running = (
        st.session_state.evolution_process is not None
        and st.session_state.evolution_process.poll() is None
    )

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh Results"):
            st.rerun()

    if evolution_running:
        st.info("ℹ️ Evolution is running in the background.")

    log_dir = base_log_dir if os.path.exists(base_log_dir) else "logs_ashare"
    st.session_state.log_dir = log_dir

    if os.path.exists(log_dir):
        all_histories = []
        all_plots_by_run = {}

        for run_folder in sorted(os.listdir(log_dir)):
            run_path = os.path.join(log_dir, run_folder)
            if os.path.isdir(run_path):
                history_file = os.path.join(run_path, "evolution_history.json")
                if os.path.exists(history_file):
                    with open(history_file, "r") as f:
                        history = json.load(f)
                    if history:
                        for entry in history:
                            entry["run_folder"] = run_folder
                        all_histories.extend(history)

                plots_dir = os.path.join(run_path, "plots")
                if os.path.exists(plots_dir):
                    all_plots_by_run[run_folder] = sorted(
                        glob.glob(os.path.join(plots_dir, "*.png")),
                        key=os.path.getmtime,
                        reverse=True,
                    )

        if all_histories:
            flattened_histories = []
            for entry in all_histories:
                flat_entry = {}
                for key, value in entry.items():
                    if key == "train" and isinstance(value, dict):
                        for train_key, train_value in value.items():
                            flat_entry[f"train_{train_key}"] = train_value
                            if train_key == "timestamp":
                                flat_entry["timestamp"] = train_value
                            if train_key == "generation":
                                flat_entry["generation"] = train_value
                            if train_key == "attempt":
                                flat_entry["attempt"] = train_value
                    elif key == "test" and isinstance(value, dict):
                        for test_key, test_value in value.items():
                            flat_entry[f"test_{test_key}"] = test_value
                    else:
                        flat_entry[key] = value
                flattened_histories.append(flat_entry)

            df_all = pd.DataFrame(flattened_histories)
            if "timestamp" not in df_all.columns:
                st.warning(
                    "No timestamp field found in history files. Check log data integrity."
                )
                st.stop()
            df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
            df_all["date"] = df_all["timestamp"].dt.date

            min_date = df_all["date"].min()
            max_date = df_all["date"].max()

            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input(
                    "Start Date", min_value=min_date, max_value=max_date, value=min_date
                )
            with col_date2:
                end_date = st.date_input(
                    "End Date", min_value=min_date, max_value=max_date, value=max_date
                )

            st.subheader("🎯 Score Filter")
            score_threshold = st.slider(
                "Minimum test/train score ratio",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Only show results where test_score >= this ratio × train_score",
            )

            mask = (df_all["date"] >= start_date) & (df_all["date"] <= end_date)
            df_filtered = df_all[mask].copy()

            if not df_filtered.empty:
                df_filtered["test_train_ratio"] = (
                    df_filtered["test_score"] / df_filtered["train_score"]
                )
                df_filtered = df_filtered[
                    df_filtered["test_train_ratio"] >= score_threshold
                ]
                df_sorted = df_filtered.sort_values("train_score", ascending=False)

                st.divider()

                if df_sorted.empty:
                    st.warning(
                        f"⚠️ No conditions met the criteria (test/train ratio ≥ {score_threshold:.2f}). Try lowering the threshold."
                    )
                else:
                    st.subheader(f"🏆 Top Results ({len(df_filtered)} runs in range)")

                    top_n = st.slider(
                        "Show top N results", 1, min(10, len(df_sorted)), 5
                    )
                    top_results = df_sorted.head(top_n)

                    st.info(
                        f"**📐 Filter**: test_score ≥ {score_threshold:.2f} × train_score | **Total Runs**: {len(df_filtered)}"
                    )

                    col_plots, col_stats = st.columns([2, 1])

                    with col_plots:
                        st.markdown("### 📈 Best Performance Plot")

                        best_plot_shown = False
                        best_row = df_sorted.iloc[0]
                        best_run_folder = best_row.get("run_folder")

                        if best_run_folder and best_run_folder in all_plots_by_run:
                            gen = best_row.get("generation")
                            att = best_row.get("attempt")
                            score = best_row.get("score", 0)

                            if gen is not None and att is not None:
                                matching = [
                                    p
                                    for p in all_plots_by_run[best_run_folder]
                                    if f"gen{gen}_att{att}" in p
                                ]

                                if matching:
                                    st.image(matching[0], width="stretch")
                                    best_plot_shown = True

                        st.markdown("**Full Bool Inequality:**")
                        st.code(
                            best_row.get(
                                "train_condition",
                                best_row.get("condition", "N/A"),
                            )
                        )

                        st.markdown("#### 📊 Train vs Test Comparison")
                        comparison_df = pd.DataFrame(
                            {
                                "Train": [
                                    f"{best_row.get('train_score', 0):.3f}",
                                    f"{best_row.get('train_total_return', 0) * 100:.2f}%",
                                    f"{best_row.get('train_sharpe', 0):.3f}",
                                    f"{best_row.get('train_calmar', 0):.3f}",
                                    f"{best_row.get('train_max_drawdown', 0) * 100:.2f}%",
                                ],
                                "Test": [
                                    f"{best_row.get('test_score', 0):.3f}",
                                    f"{best_row.get('test_total_return', 0) * 100:.2f}%",
                                    f"{best_row.get('test_sharpe', 0):.3f}",
                                    f"{best_row.get('test_calmar', 0):.3f}",
                                    f"{best_row.get('test_max_drawdown', 0) * 100:.2f}%",
                                ],
                            },
                            index=[
                                "Score",
                                "Return",
                                "Sharpe",
                                "Calmar",
                                "Max Drawdown",
                            ],
                        )
                        st.dataframe(comparison_df, width="stretch")

                        if not best_plot_shown:
                            st.info(
                                "No plot available for best result. Plot is generated during backtesting."
                            )

                    st.divider()

                    st.markdown("### 📋 Top Results Table")
                    available_cols = [
                        col
                        for col in [
                            "date",
                            "run_folder",
                            "generation",
                            "attempt",
                            "train_score",
                            "train_total_return",
                            "train_sharpe",
                            "train_calmar",
                            "train_max_drawdown",
                            "test_score",
                            "test_total_return",
                            "test_sharpe",
                            "test_calmar",
                            "test_max_drawdown",
                        ]
                        if col in top_results.columns
                    ]
                    st.dataframe(
                        top_results[available_cols].reset_index(drop=True),
                        width="stretch",
                    )

                    st.markdown("### 📈 Score Evolution")
                    score_evolution = pd.concat(
                        [
                            df_filtered.assign(series="All Runs"),
                            df_sorted.head(top_n).assign(series="Top N"),
                        ]
                    ).sort_values("timestamp")
                    st.line_chart(
                        score_evolution, x="timestamp", y="train_score", color="series"
                    )

                    st.markdown("### 📁 All Plots in Range")

                    all_range_plots = []
                    for run_folder, plots in all_plots_by_run.items():
                        for plot in plots:
                            timestamp = os.path.getmtime(plot)
                            all_range_plots.append((plot, timestamp, run_folder))

                    all_range_plots.sort(key=lambda x: x[1], reverse=True)

                    filtered_plots = []
                    for plot_path, timestamp, run_folder in all_range_plots:
                        has_stats = False
                        for _, row in df_filtered.iterrows():
                            if row.get("run_folder") == run_folder:
                                gen = row.get("generation")
                                att = row.get("attempt")
                                if gen is not None and att is not None:
                                    if f"gen{gen}_att{att}" in plot_path:
                                        has_stats = True
                                        break
                        if has_stats:
                            filtered_plots.append((plot_path, timestamp, run_folder))

                    if "selected_plot_idx" not in st.session_state:
                        st.session_state.selected_plot_idx = None

                    if filtered_plots:
                        selected_idx = st.session_state.selected_plot_idx

                        col_nav1, col_nav2, col_nav3 = st.columns([1, 8, 1])
                        with col_nav1:
                            if (
                                st.button("◀️ Prev")
                                and selected_idx is not None
                                and selected_idx > 0
                            ):
                                st.session_state.selected_plot_idx = selected_idx - 1
                                st.rerun()
                        with col_nav2:
                            st.markdown(
                                f"**Plot {selected_idx + 1 if selected_idx is not None else 1}/{len(filtered_plots)}**"
                            )
                        with col_nav3:
                            if (
                                st.button("Next ▶️")
                                and selected_idx is not None
                                and selected_idx < len(filtered_plots) - 1
                            ):
                                st.session_state.selected_plot_idx = selected_idx + 1
                                st.rerun()

                        if selected_idx is None:
                            st.session_state.selected_plot_idx = 0
                            selected_idx = 0

                        selected_plot_path, _, selected_run_folder = filtered_plots[
                            selected_idx
                        ]

                        st.markdown("#### Currently Viewing")
                        st.image(selected_plot_path, width=500)

                        matching_row = None
                        for _, row in df_filtered.iterrows():
                            if row.get("run_folder") == selected_run_folder:
                                gen = row.get("generation")
                                att = row.get("attempt")
                                if gen is not None and att is not None:
                                    if f"gen{gen}_att{att}" in selected_plot_path:
                                        matching_row = row
                                        break

                        if matching_row is not None:
                            st.markdown("**Inequality Formula:**")
                            condition = matching_row.get(
                                "train_condition",
                                matching_row.get("condition", "N/A"),
                            )
                            st.code(condition)

                            st.markdown("**📊 Performance Stats**")
                            comparison_df = pd.DataFrame(
                                {
                                    "Metric": [
                                        "Score",
                                        "Total Return",
                                        "Sharpe",
                                        "Calmar",
                                        "Max Drawdown",
                                    ],
                                    "Train": [
                                        f"{matching_row.get('train_score', 0):.3f}",
                                        f"{matching_row.get('train_total_return', 0) * 100:.2f}%",
                                        f"{matching_row.get('train_sharpe', 0):.3f}",
                                        f"{matching_row.get('train_calmar', 0):.3f}",
                                        f"{matching_row.get('train_max_drawdown', 0) * 100:.2f}%",
                                    ],
                                    "Test": [
                                        f"{matching_row.get('test_score', 0):.3f}",
                                        f"{matching_row.get('test_total_return', 0) * 100:.2f}%",
                                        f"{matching_row.get('test_sharpe', 0):.3f}",
                                        f"{matching_row.get('test_calmar', 0):.3f}",
                                        f"{matching_row.get('test_max_drawdown', 0) * 100:.2f}%",
                                    ],
                                }
                            )
                            st.dataframe(comparison_df, width="stretch")
                        else:
                            st.info("Stats not available for this plot")

                    else:
                        st.info(f"No runs found between {start_date} and {end_date}.")
            else:
                st.info("No history found. Run evolution first.")
    else:
        st.info(f"Log directory '{log_dir}' not found. Run evolution first.")

with tab3:
    st.subheader("🔍 Stock Selector")
    st.markdown(
        "Calculate which stocks satisfy a given inequality condition on a specific date."
    )

    from modules.analysis.stock_pool_calculator import calculate_stock_pool

    available_vars = ["open", "high", "low", "close", "volume"]
    st.info(f"**Available variables:** {', '.join(available_vars)}")

    col1, col2 = st.columns(2)
    with col1:
        condition_input = st.text_area(
            "Inequality Condition", placeholder="e.g., close > open", height=80
        )
    with col2:
        st.markdown("**Examples:**")
        st.code("close > open")
        st.code("close > open * 1.05")
        st.code("(close - open) / open > 0.02")
        st.markdown("**Shorthand:**")
        st.code("close > ma(5)")
        st.code("volume > ema(10)")

    all_dates = get_available_dates()
    if not all_dates:
        st.warning("No data found. Please load df_main.parquet first.")
    else:
        target_date = st.selectbox(
            "Target Date",
            options=all_dates,
            index=len(all_dates) - 1,
            format_func=lambda x: str(x)[:10] if isinstance(x, str) else str(x)[:10],
        )

        stock_prefix = st.text_input(
            "Stock Prefix (optional)",
            placeholder="60, 00, 30, 68 or leave empty for all",
            help="Applies A-share hard mask filtering",
        )

        col_rank1, col_rank2 = st.columns(2)
        with col_rank1:
            amount_rank_direction = st.selectbox(
                "Amount Rank Direction",
                ["top", "bottom"],
                index=["top", "bottom"].index("top"),
            )
        with col_rank2:
            amount_rank_n = st.number_input("Amount Rank N", min_value=1, value=50)
        amount_rank_d = st.number_input("Amount Rank D (days)", min_value=1, value=20)

        with st.spinner("Calculating stock pool..."):
            if st.button("Calculate Stock Pool", type="primary"):
                if not condition_input.strip():
                    st.warning("Please enter a condition")
                else:
                    try:
                        result = calculate_stock_pool(
                            condition=condition_input,
                            target_date=target_date,
                            prefix=stock_prefix or None,
                            amount_rank_direction=amount_rank_direction,
                            amount_rank_n=amount_rank_n,
                            amount_rank_d=amount_rank_d,
                        )

                        st.divider()
                        st.markdown("### Results")

                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            st.metric("Total Selected Stocks", result["total_selected"])
                        with col_r2:
                            st.metric("Date", result["date"][:10])

                        if result["stocks"]:
                            st.success(
                                f"Found {result['total_selected']} stocks matching the condition"
                            )

                            st.info(
                                "💡 **Tip:** Consider selecting your actual positions from the first 5 stocks in the pool whose open price is between 0.97× and 0.99× preclose."
                            )

                            stock_list = list(result["stocks"].keys())
                            st.markdown("### Stock List")
                            st.write(", ".join(sorted(stock_list)))

                            stock_df = pd.DataFrame.from_dict(
                                result["stocks"], orient="index"
                            )
                            stock_df.index.name = "Stock"
                            st.dataframe(stock_df)

                            csv = stock_df.to_csv().encode("utf-8")
                            st.download_button(
                                "Download CSV", csv, "stock_pool.csv", "text/csv"
                            )
                        else:
                            st.warning("No stocks matched the condition")

                    except Exception as e:
                        st.error(f"Error: {e}")
