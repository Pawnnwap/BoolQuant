import pandas as pd
import re
import os
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    import ollama
except ImportError:
    ollama = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def load_api_key(service: str) -> str:
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml"
    )
    if os.path.exists(config_path):
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config and "llm" in config and service in config["llm"]:
            return config["llm"][service].get("api_key", "")
    return ""

def apply_transform(df, spec):
    t = spec["transform"]
    if t == "shift":
        return df.shift(spec["periods"])
    elif t == "rolling_mean":
        return df.rolling(spec["window"]).mean()
    elif t == "rolling_median":
        return df.rolling(spec["window"]).median()
    elif t == "rolling_max":
        return df.rolling(spec["window"]).max()
    elif t == "rolling_min":
        return df.rolling(spec["window"]).min()
    elif t == "rolling_std":
        return df.rolling(spec["window"]).std()
    else:
        raise ValueError(f"Unknown transform: {t}")

class LLMOptimizer:
    def __init__(
        self,
        service,
        thinker_model: str = "qwen3",
        summarizer_model: str = "qwen3",
        evolution_logger=None,
        market_type: str = "ashare",
        return_type: str = "o2c",
    ):
        self.service = service
        self.thinker_model = thinker_model
        self.summarizer_model = summarizer_model
        self.evolution_logger = evolution_logger
        self.market_type = market_type
        self.return_type = return_type

        if service == "ollama":
            self.client = ollama.Client() if ollama else None
        elif service == "lmstudio":
            self.client = (
                OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
                if OpenAI
                else None
            )
        else:
            raise ValueError(
                f"Unsupported service: {service}. Only 'ollama' and 'lmstudio' are supported."
            )
        self.banned_conditions = [
            "close > open",
            "close < open",
            "close > close.rolling(20).mean()",
        ]

    def validate_model(self, model_name: str) -> Tuple[bool, str]:
        if not model_name or not model_name.strip():
            return False, "Model name cannot be empty"

        model_name = model_name.strip()

        if self.service == "ollama":
            return self._validate_ollama_model(model_name)
        elif self.service == "lmstudio":
            return self._validate_lmstudio_model(model_name)

        return True, ""

    def _validate_ollama_model(self, model_name: str) -> Tuple[bool, str]:
        valid_models = ["qwen3"]
        if model_name not in valid_models:
            return (
                False,
                f"Invalid model '{model_name}' for Ollama. Valid models: {', '.join(valid_models)}",
            )
        return True, ""

    def _validate_lmstudio_model(self, model_name: str) -> Tuple[bool, str]:
        valid_models = [
            "qwen3-4b-instruct-2507-gemini-3-pro-preview-distill",
            "qwen3-30b-a3b-instruct-2507",
            "falcon-h1r-7b",
        ]
        if model_name not in valid_models:
            return (
                False,
                f"Invalid model '{model_name}' for LM Studio. Valid models: {', '.join(valid_models)}",
            )
        return True, ""

    def is_duplicate_lhs(self, lhs_str: str) -> bool:
        if not self.evolution_logger:
            return False
        cleaned_lhs = lhs_str.strip().lower().replace(" ", "").replace("\n", "").replace("**", "")
        history = self.evolution_logger.load_history()
        if not history:
            return False
        for record in history:
            full_condition = record.get("full_condition", "")
            parts = re.split(r"\s*[<>]=?\s*", full_condition)
            if parts:
                historical_lhs = (
                    parts[0].strip().lower().replace(" ", "").replace("\n", "").replace("**", "")
                )
                if cleaned_lhs == historical_lhs:
                    print(
                        f"🔄 Duplicate LHS detected: '{lhs_str}' matches Gen{record['generation']} Att{record['attempt']}"
                    )
                    return True
        return False

    def is_banned_condition(self, condition_str: str) -> bool:
        cleaned_condition = (
            condition_str.strip().lower().replace(" ", "").replace("\n", "").replace("**", "")
        )
        for banned in self.banned_conditions:
            cleaned_banned = banned.strip().lower().replace(" ", "").replace("\n", "").replace("**", "")
            if cleaned_condition == cleaned_banned:
                print(f"🚫 Banned condition detected: matches example pattern")
                return True
        return False

    def is_duplicate_condition(self, condition_str: str) -> bool:
        if not self.evolution_logger:
            return False
        cleaned_condition = (
            condition_str.strip().lower().replace(" ", "").replace("\n", "").replace("**", "")
        )
        history = self.evolution_logger.load_history()
        if not history:
            return False
        for record in history:
            historical_condition = record.get("full_condition", "")
            cleaned_historical = (
                historical_condition.strip().lower().replace(" ", "").replace("\n", "").replace("**", "")
            )
            if cleaned_condition == cleaned_historical:
                print(
                    f"🔄 Duplicate detected: '{condition_str}' matches Gen{record['generation']} Att{record['attempt']}"
                )
                return True
        return False

    def generate_thinker_prompt(self, previous_results: List[Dict] = None) -> str:
        market_descriptions = {"ashare": "Chinese A-shares"}
        return_type_descriptions = {
            "o2c": "open-to-close (next 2-day return from today open to day+1 close)",
            "o2o": "open-to-open (next 2-day return from today open to day+1 open)",
            "c2c": "close-to-close (next 2-day return from today close to day+1 close)",
            "c2o": "close-to-open (overnight return)",
        }

        prompt = f"""
        Quant researcher for {market_descriptions.get(self.market_type, "Chinese A-shares")}.
        Target return calculation: {return_type_descriptions.get(self.return_type, self.return_type)}.

        Create ONE simple trading condition.
        ## 🔒 RULES:
        - SINGLE expression only (NO &, |, ~)
        - EXACTLY ONE comparison operator (<, >, <=, >=) allowed
        - NO boolean masks
        - **ADDITION/SUBTRACTION (+/-): Same category ONLY**
          - Price + Price: open + high, close - low ✓
          - Volume + Volume: volume + volume ✓
          - Price + Volume: volume + open ✗ (PROHIBITED!)
        - **MULTIPLICATION/DIVISION (*/): Different types ALLOWED**
          - Price * Volume: close * volume ✓
          - Volume * Price: volume * open ✓
        ## 📊 Variables (all DataFrames, support .shift(n) with n>0):
        - Price: open, high, low, close
        - Volume: volume
        ## ❌ BANNED:
        - "close > high.shift(1) * (close > low.shift(1))" (multiple comparisons)
        ## 🎯 Task:
        Create intuitive condition using .shift() for temporal patterns.
        Return ONLY condition string prefixed with: FINAL CONDITION:
        """
        if previous_results:
            prompt += "\n## 🔝 Recent ideas (avoid copying):\n"
            for i, res in enumerate(previous_results[-3:], 1):
                prompt += f"{i}. `{res['condition']}` (Score: {res['score']:.3f})\n"
        return prompt

    def generate_lhs_prompt_with_cache_check(
        self, previous_results: List[Dict] = None, n_examples: int = 5
    ) -> str:
        market_descriptions = {"ashare": "Chinese A-shares"}
        return_type_descriptions = {
            "o2c": "open-to-close (next 2-day return from today open to day+1 close)",
            "o2o": "open-to-open (next 2-day return from today open to day+1 open)",
            "c2c": "close-to-close (next 2-day return from today close to day+1 close)",
            "c2o": "close-to-open (overnight return)",
        }

        prompt = f"""
        You are a creative quant researcher exploring trading signals for {market_descriptions.get(self.market_type, "Chinese A-shares")}.
        Target return calculation: {return_type_descriptions.get(self.return_type, self.return_type)}.

        Think about WHAT drives prices. What combinations of price and volume tell us something meaningful?

        ## 🎨 Creative Expression Ideas (think ECONOMICALLY):
        - **Price patterns**: How does current price compare to historical prices?
        - **Volume signals**: High volume on price moves indicates strength
        - **Price-volume relationships**: Do price moves confirm with volume?
        - **Rate of change**: Fast moves vs slow moves
        - **Relative strength**: How does current price compare to historical levels?

        ## 🔒 Technical Rules:
        - SINGLE expression only (NO &, |, ~)
        - NO comparison operators (<, >, <=, >=)
        - NO boolean masks
        - **ADDITION/SUBTRACTION (+/-): Same category ONLY**
          - Price + Price: ✓
          - Volume + Volume: ✓
          - Price + Volume: ✗ (PROHIBITED!)
        - **MULTIPLICATION/DIVISION (*/): Different types ALLOWED**
        - Return ONLY the expression

        ## 📊 Available Variables (all support .shift(n) for n>0):
        - Price: open, high, low, close
        - Volume: volume

        ## 💡 Task:
        Create an expression that captures a meaningful market relationship. Be creative but logical.
        Why would this expression matter? What market behavior does it try to capture?

        Think: "What if I want to capture [market behavior]? How would I express it mathematically?"

        Your LHS expression:
        """

        if previous_results:
            prompt += "\n## 🔝 Recent expressions (use as inspiration, don't copy):\n"
            for i, res in enumerate(previous_results[-3:], 1):
                condition = res.get("condition", "")
                lhs = (
                    re.split(r"\s*[<>]=?\s*", condition)[0].strip().lower()
                    if condition
                    else ""
                )
                score = res.get("score", 0)
                prompt += f"{i}. `({lhs})` (Sharpe: {score:.2f})\n"

        return prompt

    def generate_rhs_candidates(
        self, lhs: str
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        N_all = [1, 5, 10, 20, 60]
        candidates = []

        for N in N_all:
            candidates.append({"transform": "shift", "periods": N})
            if N != 1:
                candidates.append({"transform": "rolling_mean", "window": N})
                candidates.append({"transform": "rolling_median", "window": N})
                candidates.append({"transform": "rolling_max", "window": N})
                candidates.append({"transform": "rolling_min", "window": N})
                candidates.append({"transform": "rolling_std", "window": N})

        return candidates, [], []

    def generate_summarizer_prompt(self, thinker_output: str) -> str:
        return f"""
        Extract ONLY final condition from reasoning. MUST follow:
        - SINGLE expression (NO logical operators)
        - EXACTLY ONE comparison operator (<, >, <=, >=) allowed
        - Comparison MUST have same type on both sides:
        * Price (open/high/low/close) vs Price
        * Volume (volume) vs Volume
        - Addition (+) and subtraction (-) are allowed for SAME CATEGORY ONLY
        ## 📊 Variables (all DataFrames with .shift(n)):
        - Price: open, high, low, close
        - Volume: volume
        Reasoning:
        {thinker_output}
        Return ONLY condition string in lowercase.
        Final condition:
        """

    def generate_condition(self, previous_results: List[Dict] = None) -> str:
        thinker_prompt = self.generate_thinker_prompt(previous_results)
        max_retries = 10
        retry_reason = None
        conditions_to_avoid = []

        for attempt in range(max_retries):
            full_prompt = thinker_prompt
            if retry_reason:
                full_prompt += f"\n⚠️ PREVIOUS FAILURE: {retry_reason}"
            if conditions_to_avoid:
                full_prompt += f"\n⚠️ AVOID: {', '.join(conditions_to_avoid[-3:])}"

            try:
                print(
                    f"🧠 [Thinker] Generating reasoning (attempt {attempt + 1}/{max_retries})..."
                )
                if self.service == "ollama" and self.client:
                    thinker_response = self.client.chat(
                        model=self.thinker_model,
                        messages=[{"role": "user", "content": full_prompt}],
                        options={
                            "temperature": min(1.5 + attempt * 0.1, 2),
                            "num_predict": 2048,
                        },
                    )
                    thinker_text = (
                        thinker_response["message"]["content"].strip().lower()
                    )
                else:
                    thinker_response = self.client.chat.completions.create(
                        model=self.thinker_model,
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=1 + attempt * 0.1,
                        max_completion_tokens=16384,
                        stream=False,
                    )
                    thinker_text = (
                        thinker_response.choices[0].message.content.strip().lower()
                    )
                thinker_text = re.sub(
                    r"<think>.*?</think>", "", thinker_text, flags=re.DOTALL
                )

                print("📝 [Summarizer] Extracting condition...")
                if self.service == "ollama" and self.client:
                    summarizer_response = self.client.chat(
                        model=self.summarizer_model,
                        messages=[
                            {
                                "role": "user",
                                "content": self.generate_summarizer_prompt(
                                    thinker_text
                                ),
                            }
                        ],
                        options={"temperature": 0.0, "num_predict": 256},
                    )
                    condition_str = (
                        summarizer_response["message"]["content"].strip().lower()
                    )
                else:
                    summarizer_response = self.client.chat.completions.create(
                        model=self.summarizer_model,
                        messages=[
                            {
                                "role": "user",
                                "content": self.generate_summarizer_prompt(
                                    thinker_text
                                ),
                            }
                        ],
                        temperature=0.0,
                        max_completion_tokens=256,
                        stream=False,
                    )
                    condition_str = (
                        summarizer_response.choices[0].message.content.strip().lower()
                    )
                condition_str = re.split(r"\n", condition_str)[0].strip().lower()
                condition_str = re.sub(r"^```.*\n?", "", condition_str)
                condition_str = re.sub(r"\n```.*", "", condition_str)
                condition_str = condition_str.strip('`" \t\n')

                if self.is_banned_condition(condition_str):
                    reason = f"Banned pattern: {condition_str}"
                    print(f"🔄 {reason}")
                    conditions_to_avoid.append(f"[BANNED]{condition_str}")
                    retry_reason = reason
                    continue

                if self.is_duplicate_condition(condition_str):
                    reason = f"Duplicate: {condition_str}"
                    print(f"🔄 {reason}")
                    conditions_to_avoid.append(f"[DUPLICATE]{condition_str}")
                    retry_reason = reason
                    continue

                return condition_str

            except Exception as e:
                retry_reason = f"Exception: {str(e)}"
                print(f"❌ {retry_reason}")

        fallback = "close > open"
        if self.is_duplicate_condition(fallback):
            fallback = "close > close.shift(1)"
        return fallback

    def generate_lhs(
        self, previous_results: List[Dict] = None, max_retries: int = 10
    ) -> str:
        retry_reason = None
        expressions_to_avoid = []

        for attempt in range(max_retries):
            full_prompt = self.generate_lhs_prompt_with_cache_check(
                previous_results, n_examples=5
            )

            if retry_reason:
                full_prompt += f"\n⚠️ PREVIOUS FAILURE: {retry_reason}"
            if expressions_to_avoid:
                full_prompt += f"\n⚠️ AVOID: {', '.join(expressions_to_avoid[-3:])}"

            try:
                print(
                    f"🧠 [LHS Generator] Step 1: Thinking (attempt {attempt + 1}/{max_retries})..."
                )
                thinker_response = self.llm_think(
                    prompt=full_prompt,
                    model=self.thinker_model,
                    temperature=min(1.5 + attempt * 0.1, 2),
                    max_tokens=32768,
                )

                print(f"📝 [LHS Generator] Step 2: Extracting clean expression...")
                extract_prompt = f"""Extract the FINAL expression from the reasoning below.

                ## ONLY ALLOWED - Variables (use exactly these names):
                        - Price: open, high, low, close
                        - Volume: volume

                ## ONLY ALLOWED - Operations:
                        - Multiply: *
                        - Add: +
                        - Subtract: -
                        - Time shift: .shift(n) where n is a positive integer (e.g., .shift(1))
                        - Rolling window: .rolling(n).mean() or .rolling(n).median() or .rolling(n).max() or .rolling(n).min() or .rolling(n).std()
                        - Absolute value: (expression).abs() - NOTE: Use METHOD FORMAT, NOT function call! Example: (high - low).abs() NOT abs(high - low)

                ## Rules:
                - NO comparisons (<, >, <=, >=, =, !)
                - NO logical operators (&, |, ~)
                - NO boolean masks (limit_up, limit_down, zhaban)
                - NO function calls like abs(), max(), min() - use .abs() as a METHOD instead
                - NO np., pd., or other library prefixes

                ## Reasoning:
                {thinker_response}

                Final expression:"""

                lhs_text = self.llm_summarize(
                    prompt=extract_prompt,
                    model=self.summarizer_model,
                    temperature=0.0,
                    max_tokens=128,
                )
                lhs_str = self.extract_from_response(lhs_text)

                if "shift(-" in lhs_str:
                    reason = f"Future leak detected: shift(-) in LHS"
                    print(f"🚫 {reason}: {lhs_str}")
                    expressions_to_avoid.append(lhs_str)
                    retry_reason = reason
                    continue

                if any(op in lhs_str for op in ["<", ">", "=", "!", "&", "|", "~"]):
                    ops_found = [
                        op
                        for op in ["<", ">", "=", "!", "&", "|", "~"]
                        if op in lhs_str
                    ]
                    reason = f"""Previous attempt failed: Output contained forbidden operators: {ops_found}.
                    How to fix: Return ONLY a mathematical expression with NO comparison operators (<, >, =, !) and NO logical operators (&, |, ~).
                    Use only +, -, *, /, parentheses, and .shift()/.rolling()/.abs() methods.
                    IMPORTANT: Use .abs() as a METHOD call, NOT as a function! Example: (high - low).abs() NOT abs(high - low)
                    Remember: LHS must be a pure mathematical expression WITHOUT any comparison operators."""
                    print(f"🔄 Contains operators: {lhs_str}")
                    expressions_to_avoid.append(lhs_str)
                    retry_reason = reason
                    continue

                if self.is_duplicate_lhs(lhs_str):
                    reason = f"Duplicate LHS: {lhs_str}"
                    print(f"🔄 {reason}")
                    expressions_to_avoid.append(lhs_str)
                    retry_reason = reason
                    continue

                if hasattr(self, "evaluator") and self.evaluator:
                    try:
                        is_valid = False
                        if hasattr(self.evaluator, "validate_lhs_syntax"):
                            is_valid = self.evaluator.validate_lhs_syntax(lhs_str)
                        else:
                            is_valid = self.evaluator.validate_condition_syntax(lhs_str)

                        if not is_valid:
                            reason = """Previous attempt failed: LHS violates type rules (invalid cross-type + or -).
                            How to fix: Use + and - ONLY within same category (Price±Price, Volume±Volume, Returns±Returns).
                            Cross-type operations must use * or / only."""
                            print(f"🔄 Rule violation: {lhs_str}")
                            expressions_to_avoid.append(lhs_str)
                            retry_reason = reason
                            continue
                    except Exception as e:
                        print(f"⚠️ Validation check failed: {e}")

                lhs_str = self.clean_abs_usage(lhs_str)

                print(f"✅ LHS: {lhs_str}")
                return lhs_str.lower()

            except Exception as e:
                retry_reason = f"Exception: {str(e)}"
                print(f"❌ {retry_reason}")

        fallback = "close"
        return fallback

    def generate_condition_stepwise(
        self, previous_results: List[Dict] = None, evaluator=None, backtester=None
    ):

        self.evaluator = evaluator

        print("\n📍 Step 1: Generating LHS...")
        lhs = self.generate_lhs(previous_results)
        print(f"✅ LHS generated: ({lhs})")

        print("🔢 Evaluating LHS dataframe...")
        lhs_df = evaluator.evaluate_expression(lhs)

        lhs_category = (
            evaluator.get_expression_category(lhs) if evaluator else "unknown"
        )
        print(f"📊 LHS category: {lhs_category}")

        print("\n📍 Step 2: Generating RHS candidates from templates...")
        standard_rhs, psych_rhs, cross_sectional_rhs = self.generate_rhs_candidates(lhs)

        all_conditions = []

        for rhs_transform in standard_rhs:
            for op in [">", "<"]:
                all_conditions.append((rhs_transform, op, "standard", None))

        for rhs_transform in psych_rhs:
            for op in [">", "<"]:
                all_conditions.append((rhs_transform, op, "psychological", True))

        for transform in cross_sectional_rhs:
            all_conditions.append((transform, None, "cross_sectional", None))

        print(f"📊 Generated {len(all_conditions)} unique conditions from templates")

        if len(all_conditions) == 0:
            print("⚠️ No valid conditions generated from templates")
            return None, pd.Series(dtype=float), 0.0, 0.0, {}

        print(f"\n📍 Step 3: Backtesting {len(all_conditions)} condition(s)...")
        test_results = []
        print_lock = threading.Lock()

        def process_single_condition(args):
            transform_spec, op, ctype, is_psych = args
            try:
                if ctype == "cross_sectional":
                    condition_str = f"{lhs} [cross-sectional {transform_spec['transform']} {transform_spec.get('q', '')}]"
                    condition_df = apply_transform(lhs_df, transform_spec)
                else:
                    rhs_df = apply_transform(lhs_df, transform_spec)

                    if is_psych:
                        window_or_period = transform_spec.get(
                            "window", transform_spec.get("periods", "")
                        )
                        condition_str = f"({lhs})_{transform_spec['transform']}({window_or_period}) {op} 0"
                        if op == ">":
                            condition_df = rhs_df > 0
                        else:
                            condition_df = rhs_df < 0
                    else:
                        window_or_period = transform_spec.get(
                            "window", transform_spec.get("periods", "")
                        )
                        rhs_expr = (
                            f"({lhs}).{transform_spec['transform']}({window_or_period})"
                        )
                        condition_str = f"({lhs}) {op} ({rhs_expr})"
                        if op == ">":
                            condition_df = lhs_df > rhs_df
                        else:
                            condition_df = lhs_df < rhs_df

                daily_pnl, score, total_return, full_stats = backtester.run_backtest(
                    condition_df
                )

                with print_lock:
                    print(
                        f"   {condition_str} - Train Score: {score:.4f}, Train Sharpe: {full_stats.get('sharpe', 0):.4f}, Train Calmar: {full_stats.get('calmar', 0):.4f}, Train Return: {total_return:.2%}"
                    )

                return (condition_str, op, score, total_return, daily_pnl, full_stats)

            except Exception as e:
                with print_lock:
                    print(f"   ❌ Backtest failed: {e}")
                return None

        max_workers = min(10, len(all_conditions)) if all_conditions else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_condition = {
                executor.submit(process_single_condition, cond): cond
                for cond in all_conditions
            }

            for future in as_completed(future_to_condition):
                result = future.result()
                if result is not None:
                    test_results.append(result)

        if len(test_results) == 0:
            print("⚠️ No successful backtests")
            return None, pd.Series(dtype=float), 0.0, 0.0, {}

        if len(test_results) >= 2:
            best_result = max(test_results, key=lambda x: x[2])
            worst_result = min(test_results, key=lambda x: x[2])
            score_diff = best_result[2] - worst_result[2]

            print(
                f"📊 Best Score: {best_result[2]:.4f}, Worst Score: {worst_result[2]:.4f}, Diff: {score_diff:.4f}"
            )

            if score_diff < 0.5 or best_result[2] < 0.6:
                print(
                    f"🚫 Invalid: Score difference ({score_diff:.4f}) < 0.5 or best score < 0.6"
                )
                return None, pd.Series(dtype=float), 0.0, 0.0, {}
        else:
            best_result = test_results[0]
            if best_result[2] < 0.6:
                print(f"🚫 Invalid: Best score ({best_result[2]:.4f}) < 0.6")
                return None, pd.Series(dtype=float), 0.0, 0.0, {}

        best_condition = best_result[0]
        best_score = best_result[2]
        best_return = best_result[3]
        best_pnl = best_result[4]
        best_full_stats = best_result[5] if len(best_result) > 5 else {}

        print(f"\n🏆 Best condition: {best_condition} (Train Score: {best_score:.4f})")

        if best_result[1] == "<" and best_result[2] != "cross_sectional":
            best_condition = self._swap_operator_to_gt(best_condition)
            print(f"   🔄 Normalized to > format: {best_condition}")

        return best_condition, best_pnl, best_score, best_return, best_full_stats

    def _swap_operator_to_gt(self, condition: str) -> str:
        match = re.search(r"\s*(<|>)\s*", condition)
        if not match:
            return condition

        operator = match.group(1)
        lhs_str = condition[: match.start()].strip().lower()
        rhs_str = condition[match.end() :].strip().lower()

        if operator == ">":
            return condition

        return f"{rhs_str} > {lhs_str}"

    def llm_think(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> str:
        model = model or self.thinker_model
        try:
            if self.service == "ollama" and self.client:
                response = self.client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                return response["message"]["content"].strip().lower()
            elif self.client:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    stream=False,
                )
                return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"❌ llm_think error: {e}")
            raise

    def llm_summarize(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> str:
        model = model or self.summarizer_model
        try:
            if self.service == "ollama" and self.client:
                response = self.client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                return response["message"]["content"].strip().lower()
            elif self.client:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    stream=False,
                )
                return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"❌ llm_summarize error: {e}")
            raise

    def extract_from_response(self, response: str, prefix: str = None) -> str:
        cleaned = re.sub(r"```.*\n?", "", response)
        cleaned = cleaned.strip('`" \t\n')
        cleaned = cleaned.replace("**", "")

        if prefix and f"{prefix}:" in cleaned:
            cleaned = cleaned.split(f"{prefix}:")[-1].split("\n")[0].strip().lower()

        return cleaned

    def clean_abs_usage(self, expression: str) -> str:

        def replace_abs(match):
            inner = match.group(1)
            return f"({inner}).abs()"

        cleaned = re.sub(r"\babs\s*\(([^)]+)\)", replace_abs, expression)

        return cleaned
