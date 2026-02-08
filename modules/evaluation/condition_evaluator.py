import pandas as pd
import numpy as np
import ast
from typing import Dict, List, Tuple
import re

class SafeConditionEvaluator:
    class MockSeries:
        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            return self

        def __add__(self, other):
            return self

        def __sub__(self, other):
            return self

        def __mul__(self, other):
            return self

        def __truediv__(self, other):
            return self

        def __floordiv__(self, other):
            return self

        def __mod__(self, other):
            return self

        def __pow__(self, other):
            return self

        def __lt__(self, other):
            return self

        def __le__(self, other):
            return self

        def __gt__(self, other):
            return self

        def __ge__(self, other):
            return self

        def __eq__(self, other):
            return self

        def __ne__(self, other):
            return self

        def __neg__(self):
            return self

        def __invert__(self):
            return self

        def __and__(self, other):
            return self

        def __or__(self, other):
            return self

        def __bool__(self):
            return True

    def __init__(self, data_preloader, validate_only=False):
        self.validate_only = validate_only
        self.data = None if validate_only else data_preloader
        self.safe_functions = {
            "np": np,
            "pd": pd,
            "talib": __import__("talib"),
            "abs": abs,
            "max": max,
            "min": min,
            "sum": sum,
            "len": len,
            "any": any,
            "all": all,
        }
        self.allowed_operators = {"(", ")", "<", ">", "=", "!", "+", "-", "*", "/"}
        self.disallowed_patterns = [
            "|",
            "or ",
            "OR",
            "Or",
            "oR",
            "&",
            "and",
            "AND",
            "And",
            "aNd",
            "~",
            "not",
            "NOT",
            "Not",
            "nOt",
            "if ",
            "else: ",
            "for ",
            "while ",
            "def ",
            "class ",
            "import",
            "exec",
            "eval",
            "sys",
            "subprocess",
            "__import__",
            "globals",
            "locals",
            "open(",
            "write",
            "delete",
            "print",
        ]
        self.variable_categories = {
            "open": "price",
            "high": "price",
            "low": "price",
            "close": "price",
            "volume": "volume",
        }

    @property
    def available_vars(self):
        if self.validate_only:
            mock = self.MockSeries()
            vars_dict = {
                "open": mock,
                "high": mock,
                "low": mock,
                "close": mock,
                "volume": mock,
            }
            return vars_dict
        vars_dict = {
            "open": self.data.open,
            "high": self.data.high,
            "low": self.data.low,
            "close": self.data.close,
            "volume": self.data.volume,
        }
        return vars_dict

    @property
    def var_names(self):
        return list(self.available_vars.keys())

    def _categorize_node(self, node: ast.AST, context: Dict) -> str:
        if isinstance(node, ast.Name):
            return self.variable_categories.get(node.id, "unknown")
        elif isinstance(node, ast.Attribute):
            return self._categorize_node(node.value, context)
        elif isinstance(node, ast.Constant):
            return "constant"
        elif isinstance(node, ast.UnaryOp):
            return self._categorize_node(node.operand, context)
        elif isinstance(node, ast.BinOp):
            left_cat = self._categorize_node(node.left, context)
            right_cat = self._categorize_node(node.right, context)
            if isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                if left_cat == "constant":
                    return right_cat
                elif right_cat == "constant":
                    return left_cat
                return left_cat
            elif isinstance(node.op, (ast.Add, ast.Sub)):
                if left_cat == "constant":
                    return right_cat
                elif right_cat == "constant":
                    return left_cat
                return left_cat
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                return self._categorize_node(node.func.value, context)
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in ("abs", "len", "sum", "min", "max"):
                    return "constant"
                elif func_name in ("np", "pd"):
                    return "indicator"
                if node.args:
                    return self._categorize_node(node.args[0], context)
            return "indicator"
        elif isinstance(node, ast.Compare):
            return "boolean"
        return "unknown"

    def get_expression_category(self, expression_str: str) -> str:
        try:
            tree = ast.parse(expression_str, mode="eval")
            context = {**self.safe_functions, **self.available_vars}
            return self._categorize_node(tree.body, context)
        except Exception:
            return "unknown"

    def _validate_ast_operations(
        self, node: ast.AST, context: Dict
    ) -> Tuple[bool, str]:
        if isinstance(node, ast.BinOp):
            left_cat = self._categorize_node(node.left, context)
            right_cat = self._categorize_node(node.right, context)
            data_cats = {"price", "volume"}
            if isinstance(node.op, ast.Add):
                if left_cat in data_cats and right_cat in data_cats:
                    if left_cat != right_cat:
                        return (
                            False,
                            f"Addition (+) only allowed for same category, not {left_cat} + {right_cat}",
                        )
            elif isinstance(node.op, ast.Sub):
                if left_cat in data_cats and right_cat in data_cats:
                    if left_cat != right_cat:
                        return (
                            False,
                            f"Subtraction (-) only allowed for same category, not {left_cat} - {right_cat}",
                        )
            left_valid, left_msg = self._validate_ast_operations(node.left, context)
            if not left_valid:
                return False, left_msg
            right_valid, right_msg = self._validate_ast_operations(node.right, context)
            if not right_valid:
                return False, right_msg
        for child in ast.iter_child_nodes(node):
            child_valid, child_msg = self._validate_ast_operations(child, context)
            if not child_valid:
                return False, child_msg
        return True, ""

    def validate_condition_syntax(self, condition_str: str) -> bool:
        condition_str_clean = condition_str.strip().replace('"', "")
        for pattern in self.disallowed_patterns:
            if pattern.lower() in condition_str_clean.lower():
                print(f"❌ Disallowed pattern detected: '{pattern}'")
                return False
        if condition_str_clean.count("(") != condition_str_clean.count(")"):
            print("❌ Unbalanced parentheses in condition")
            return False
        tokens = re.split(r"\s+", condition_str_clean)
        for token in tokens:
            if any(char in token for char in self.disallowed_patterns):
                print(f"❌ Disallowed token detected: '{token}'")
                return False
        if any(op in condition_str_clean for op in ["&", "|", "~"]):
            print(
                "❌ Logical operators detected. Only simple expressions are allowed (no & | ~)."
            )
            return False
        try:
            tree = ast.parse(condition_str_clean, mode="eval")
            context = {**self.safe_functions, **self.available_vars}
            is_valid, error_msg = self._validate_ast_operations(tree, context)
            if not is_valid:
                print(f"❌ Dimensional constraint violation: {error_msg}")
                return False
        except SyntaxError as e:
            print(f"❌ Syntax error in condition: {e}")
            return False
        except Exception as e:
            print(f"❌ Error during AST validation: {e}")
            return False
        try:
            mock_context = {var: self.MockSeries() for var in self.var_names}
            mock_context.update(self.safe_functions)
            compile(condition_str_clean, "<string>", "eval")
            eval(condition_str_clean, {"__builtins__": {}}, mock_context)
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error in condition: {e}")
            return False
        except NameError as e:
            print(f"❌ Unknown variable in condition: {e}")
            if "name '" in str(e):
                var_name = str(e).split("'")[1]
                print(f"  Available variables: {self.var_names}")
                print(f"  Unknown variable: '{var_name}'")
            return False
        except Exception as e:
            print(f"❌ Error during syntax validation: {e}")
            return False

    def validate_condition_structure(self, condition_str: str) -> bool:
        if not re.search(r"[<>=!]=?", condition_str):
            print("❌ Condition must contain at least one comparison operator")
            return False
        logical_ops = re.findall(r"[&|~]", condition_str)
        if logical_ops:
            print(
                "❌ Logical operators (& | ~) are not allowed. Use simple expressions only."
            )
            return False
        return True

    def validate_condition(self, condition_str: str) -> bool:
        print(f"🔍 Validating condition: {condition_str}")
        if not self.validate_condition_syntax(condition_str):
            return False
        if not self.validate_condition_structure(condition_str):
            return False
        print("✅ Condition validation passed")
        return True

    def evaluate_condition(self, condition_str: str) -> pd.DataFrame:
        if not self.validate_condition(condition_str):
            raise ValueError("Condition validation failed")
        safe_context = self.safe_functions.copy()
        safe_context.update(self.available_vars)
        try:
            print("⚙️ Evaluating condition...")
            result = eval(condition_str, {"__builtins__": {}}, safe_context)
            if isinstance(result, pd.Series):
                return pd.DataFrame(result).T
            elif isinstance(result, np.ndarray):
                return pd.DataFrame(
                    result, index=self.data.close.index, columns=self.data.close.columns
                )
            elif isinstance(result, pd.DataFrame):
                return result
            else:
                return pd.DataFrame(
                    np.broadcast_to(
                        result, (len(self.data.close), len(self.data.close.columns))
                    ),
                    index=self.data.close.index,
                    columns=self.data.close.columns,
                )
        except Exception as e:
            print(f"❌ Error evaluating condition: {e}")
            print(f"🔍 Condition string: {condition_str}")
            print(f"📋 Available variables: {list(safe_context.keys())}")
            import traceback

            traceback.print_exc()
            raise

    def evaluate_expression(self, expression_str: str) -> pd.DataFrame:
        if self.validate_only:
            raise RuntimeError("Cannot evaluate expressions in validate_only mode")

        expression_str_clean = expression_str.strip().replace('"', "")

        for pattern in self.disallowed_patterns:
            if pattern.lower() in expression_str_clean.lower():
                raise ValueError(f"Disallowed pattern detected: '{pattern}'")

        if expression_str_clean.count("(") != expression_str_clean.count(")"):
            raise ValueError("Unbalanced parentheses in expression")

        if any(op in expression_str_clean for op in ["&", "|", "~"]):
            raise ValueError("Logical operators not allowed in expression")

        if re.search(r"[<>=!]=?", expression_str_clean):
            raise ValueError(
                "Comparison operators not allowed in expression evaluation"
            )

        try:
            tree = ast.parse(expression_str_clean, mode="eval")
            context = {**self.safe_functions, **self.available_vars}
            is_valid, error_msg = self._validate_ast_operations(tree, context)
            if not is_valid:
                raise ValueError(f"Type constraint violation: {error_msg}")
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}")

        safe_context = self.safe_functions.copy()
        safe_context.update(self.available_vars)

        try:
            result = eval(expression_str_clean, {"__builtins__": {}}, safe_context)

            if isinstance(result, pd.DataFrame):
                return result
            elif isinstance(result, pd.Series):
                return (
                    pd.DataFrame(result).T
                    if result.name in self.data.close.columns
                    else pd.DataFrame(result)
                )
            elif isinstance(result, np.ndarray):
                return pd.DataFrame(
                    result, index=self.data.close.index, columns=self.data.close.columns
                )
            else:
                return pd.DataFrame(
                    np.broadcast_to(
                        result, (len(self.data.close), len(self.data.close.columns))
                    ),
                    index=self.data.close.index,
                    columns=self.data.close.columns,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate expression '{expression_str}': {e}")
