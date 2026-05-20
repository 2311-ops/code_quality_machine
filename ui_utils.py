"""Shared helpers for the Streamlit UI."""

from __future__ import annotations

import ast
import io
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from radon.complexity import cc_visit
    from radon.raw import analyze as raw_analyze
except Exception:  # pragma: no cover - fallback for partial environments
    cc_visit = None
    raw_analyze = None

try:
    import pyflakes.api as pyflakes_api
    import pyflakes.reporter as pyflakes_reporter
except Exception:  # pragma: no cover - fallback for partial environments
    pyflakes_api = None
    pyflakes_reporter = None

from Labeler import _compute_metrics

OUTPUT_DIR = Path("output")

MODEL_RESULTS = {
    "Random Forest": {"acc": 0.9356, "f1": 0.9332},
    "Gradient Boosting": {"acc": 1.0, "f1": 1.0},
    "1D CNN": {"acc": 0.9554, "f1": 0.9551},
    "LSTM": {"acc": 0.4455, "f1": 0.2756},
}

SAMPLE_SNIPPETS = {
    "Clean utility": '''\
def normalize_name(raw_name: str) -> str:
    """Normalize a user-facing name."""
    return raw_name.strip().replace("_", " ").title()


def build_slug(value: str) -> str:
    value = value.strip().lower()
    return value.replace(" ", "-")
''',
    "Messy script": '''\
import os
import sys

def process(x, y, z, a, b, c):
    temp = x + y
    if temp > 10:
        return a + b + c
    else:
        return temp

answer = process(1, 2, 3, 4, 5, 6)
''',
    "Buggy sample": '''\
def compute_score(data):
    if len(data) > 0:
        return data[0] + unknown_value
    return 0
''',
}


@dataclass
class AnalysisResult:
    label: str
    confidence: float
    quality_score: float
    rationale: list[str]
    phase1: dict
    ast: dict
    lint: dict
    parse_error: bool


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_outputs(output_dir: Path = OUTPUT_DIR) -> dict:
    """Load the saved project outputs used by the UI."""
    labeled_path = output_dir / "labeled_dataset.csv"
    features_path = output_dir / "features_dataset.csv"
    meta_path = output_dir / "model_metadata.json"

    labeled_df = pd.read_csv(labeled_path) if labeled_path.exists() else pd.DataFrame()
    features_df = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()
    meta = read_json(meta_path, default={}) or {}

    total_files = len(labeled_df)
    feature_count = max(len(features_df.columns) - 2, 0) if not features_df.empty else 0
    if not features_df.empty:
        feature_count = len([c for c in features_df.columns if c not in {"quality_label", "code"}])

    label_counts = (
        labeled_df["quality_label"].value_counts().to_dict()
        if "quality_label" in labeled_df.columns and not labeled_df.empty
        else {}
    )

    return {
        "labeled_df": labeled_df,
        "features_df": features_df,
        "meta": meta,
        "total_files": total_files,
        "feature_count": feature_count,
        "label_counts": label_counts,
        "best_model_name": max(MODEL_RESULTS, key=lambda name: MODEL_RESULTS[name]["f1"]),
        "best_model_metrics": MODEL_RESULTS[max(MODEL_RESULTS, key=lambda name: MODEL_RESULTS[name]["f1"])],
    }


def get_image_paths(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Return the project screenshots that exist in the output folder."""
    wanted = [
        "label_distribution.png",
        "lint_features_boxplot.png",
        "feature_importance_bar.png",
        "final_model_comparison.png",
        "rf_results.png",
        "gbm_results.png",
        "cnn_results.png",
        "baseline_comparison.png",
        "feature_correlation_heatmap.png",
    ]
    return [output_dir / name for name in wanted if (output_dir / name).exists()]


def get_artifact_paths(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    names = [
        "features_dataset.csv",
        "x_train.csv",
        "x_test.csv",
        "y_train.csv",
        "y_test.csv",
        "encoder.json",
        "scaler.json",
        "tfidf_vocab.json",
        "best_model_cnn.keras",
        "model_metadata.json",
    ]
    return [output_dir / name for name in names if (output_dir / name).exists()]


def tokenize_code(code: str) -> str:
    code = re.sub(r'\"\"\".*?\"\"\"', " ", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", " ", code, flags=re.DOTALL)
    code = re.sub(r'"""', " ", code)
    code = re.sub(r'"[^"]*"', " ", code)
    code = re.sub(r"'[^']*'", " ", code)
    code = re.sub(r"#.*", " ", code)
    code = code.replace("_", " ")
    code = re.sub(r"([a-z])([A-Z])", r"\1 \2", code)
    code = re.sub(r"[^a-zA-Z0-9\s]", " ", code)
    code = re.sub(r"\s+", " ", code).strip().lower()
    return code


def get_tree_depth(node, depth: int = 0) -> int:
    if not isinstance(node, ast.AST):
        return depth
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(get_tree_depth(child, depth + 1) for child in children)


def extract_ast_features(code: str) -> dict:
    base = {
        "ast_functions": 0,
        "ast_classes": 0,
        "ast_imports": 0,
        "ast_max_depth": 0,
        "ast_avg_args": 0.0,
        "ast_lambdas": 0,
        "ast_comprehensions": 0,
        "ast_try_blocks": 0,
        "ast_returns": 0,
        "ast_global_vars": 0,
        "ast_docstring_ratio": 0.0,
        "ast_assert_count": 0,
        "ast_parse_error": 0,
    }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        base["ast_parse_error"] = 1
        return base

    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    lambdas = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
    comps = [n for n in ast.walk(tree) if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))]
    try_blocks = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    global_vars = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))]
    arg_counts = [len(fn.args.args) for fn in functions]
    avg_args = float(np.mean(arg_counts)) if arg_counts else 0.0

    def has_docstring(fn):
        return (
            fn.body
            and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)
        )

    doc_ratio = (sum(1 for fn in functions if has_docstring(fn)) / len(functions)) if functions else 0.0

    return {
        "ast_functions": len(functions),
        "ast_classes": len(classes),
        "ast_imports": len(imports),
        "ast_max_depth": get_tree_depth(tree),
        "ast_avg_args": round(avg_args, 2),
        "ast_lambdas": len(lambdas),
        "ast_comprehensions": len(comps),
        "ast_try_blocks": len(try_blocks),
        "ast_returns": len(returns),
        "ast_global_vars": len(global_vars),
        "ast_docstring_ratio": round(doc_ratio, 3),
        "ast_assert_count": len(asserts),
        "ast_parse_error": 0,
    }


def extract_lint_features(code: str, loc: int = 1) -> dict:
    if pyflakes_api is None or pyflakes_reporter is None:
        return {"lint_errors": 0, "lint_warnings": 0, "lint_total": 0, "lint_per_loc": 0.0}

    error_buf = io.StringIO()
    warning_buf = io.StringIO()

    class SplitReporter(pyflakes_reporter.Reporter):
        def unexpectedError(self, filename, msg):
            error_buf.write(msg + "\n")

        def syntaxError(self, filename, msg, lineno, offset, text):
            error_buf.write(msg + "\n")

        def flake(self, message):
            txt = str(message)
            if any(kw in txt for kw in ("undefined", "import *", "SyntaxError")):
                error_buf.write(txt + "\n")
            else:
                warning_buf.write(txt + "\n")

    try:
        reporter = SplitReporter(warning_buf, error_buf)
        pyflakes_api.check(code, filename="<string>", reporter=reporter)
    except Exception:
        pass

    errors = len([line for line in error_buf.getvalue().splitlines() if line.strip()])
    warnings_count = len([line for line in warning_buf.getvalue().splitlines() if line.strip()])
    total = errors + warnings_count
    loc_safe = max(loc, 1)

    return {
        "lint_errors": errors,
        "lint_warnings": warnings_count,
        "lint_total": total,
        "lint_per_loc": round(total / loc_safe, 4),
    }


def score_code(phase1: dict, ast_features: dict, lint_features: dict) -> tuple[str, float, float, list[str]]:
    """Return a transparent demo label, confidence, quality score, and rationale."""
    mi = phase1.get("mi_score", 0.0)
    avg_cc = phase1.get("avg_cc", 0.0)
    max_cc = phase1.get("max_cc", 0.0)
    lint_total = lint_features.get("lint_total", 0)
    lint_errors = lint_features.get("lint_errors", 0)
    doc_ratio = ast_features.get("ast_docstring_ratio", 0.0)
    depth = ast_features.get("ast_max_depth", 0)
    parse_error = ast_features.get("ast_parse_error", 0) == 1

    if parse_error or lint_errors >= 3 or mi < 40 or avg_cc > 10 or max_cc > 15:
        label = "bad"
    elif mi >= 65 and avg_cc <= 5 and lint_total <= 1 and depth <= 10:
        label = "good"
    else:
        label = "medium"

    quality_score = (
        0.50 * mi
        - 2.50 * avg_cc
        - 1.00 * lint_total
        - 0.50 * depth
        + 10.0 * doc_ratio
        + 2.0 * ast_features.get("ast_functions", 0)
    )
    quality_score = float(np.clip(quality_score, 0, 100))

    if label == "good":
        confidence = 0.68 + min((mi - 60) / 120, 0.20) + min((5 - min(avg_cc, 5)) / 50, 0.08)
    elif label == "bad":
        confidence = 0.72 + min((10 - min(mi, 10)) / 40, 0.12) + min((lint_errors + lint_total) / 60, 0.10)
    else:
        confidence = 0.60 + min(abs(mi - 55) / 120, 0.10)
    confidence = float(np.clip(confidence, 0.55, 0.98))

    rationale = []
    if parse_error:
        rationale.append("The code does not parse cleanly, which strongly hurts the quality score.")
    if lint_errors:
        rationale.append(f"Pyflakes found {lint_errors} lint error(s), which usually signals fragile code.")
    elif lint_total == 0:
        rationale.append("No lint issues were detected, which is a positive signal.")

    if mi >= 65:
        rationale.append(f"Maintainability Index is strong at {mi:.2f}.")
    elif mi < 40:
        rationale.append(f"Maintainability Index is low at {mi:.2f}.")

    if avg_cc <= 5:
        rationale.append(f"Average cyclomatic complexity is low at {avg_cc:.2f}.")
    elif avg_cc > 10:
        rationale.append(f"Average cyclomatic complexity is high at {avg_cc:.2f}.")

    if doc_ratio > 0:
        rationale.append(f"Docstring coverage is {doc_ratio:.2f}, which helps readability.")
    if depth > 10:
        rationale.append(f"AST nesting is fairly deep at {depth}, which can make code harder to follow.")

    if not rationale:
        rationale.append("The score is based on the combined maintainability, structure, and lint signals.")

    return label, confidence, quality_score, rationale


def analyze_code(code: str) -> AnalysisResult:
    phase1 = _compute_metrics(code) or {
        "mi_score": 0.0,
        "avg_cc": 0.0,
        "max_cc": 0,
        "loc": len(code.splitlines()) or 1,
        "lloc": 0,
        "sloc": 0,
        "comments": 0,
        "blank": 0,
        "comment_ratio": 0.0,
    }

    ast_features = extract_ast_features(code)
    lint_features = extract_lint_features(code, loc=phase1.get("loc", 1))
    label, confidence, quality_score, rationale = score_code(phase1, ast_features, lint_features)

    return AnalysisResult(
        label=label,
        confidence=confidence,
        quality_score=quality_score,
        rationale=rationale,
        phase1=phase1,
        ast=ast_features,
        lint=lint_features,
        parse_error=ast_features.get("ast_parse_error", 0) == 1,
    )
