"""Phase 4 inference helpers for raw Python code quality assessment."""

from __future__ import annotations

import ast
import io
import json
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "model_cnn.keras"
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_DIR / "model_metadata.json"
DEFAULT_SCALER_PATH = DEFAULT_OUTPUT_DIR / "scaler.json"
DEFAULT_CORPUS_PATH = DEFAULT_OUTPUT_DIR / "labeled_dataset.csv"

TFIDF_MAX_FEATURES = 200
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_TOKEN_PATTERN = r"\b[a-z][a-z0-9]{1,}\b"

RULES = {
    "mi_score": {
        "threshold": 65,
        "direction": "below",
        "tip": (
            "Maintainability Index is low ({val:.1f}/100). "
            "Consider splitting large functions, adding docstrings, "
            "and reducing deeply nested logic."
        ),
    },
    "avg_cc": {
        "threshold": 5,
        "direction": "above",
        "tip": (
            "Average cyclomatic complexity is high ({val:.1f}). "
            "Break complex functions into smaller, single-purpose helpers."
        ),
    },
    "max_cc": {
        "threshold": 10,
        "direction": "above",
        "tip": (
            "Your most complex function has CC={val:.0f}. "
            "It has too many branches - extract logic into sub-functions."
        ),
    },
    "comment_ratio": {
        "threshold": 0.05,
        "direction": "below",
        "tip": (
            "Comment ratio is very low ({val:.1%}). "
            "Add docstrings to functions and inline comments for non-obvious logic."
        ),
    },
    "ast_docstring_ratio": {
        "threshold": 0.3,
        "direction": "below",
        "tip": (
            "Only {val:.0%} of functions have docstrings. "
            "Document all public functions with purpose, args, and return value."
        ),
    },
    "lint_errors": {
        "threshold": 0,
        "direction": "above",
        "tip": (
            "Found {val:.0f} lint error(s) (undefined names / import issues). "
            "Run pyflakes and fix all errors before shipping."
        ),
    },
    "lint_warnings": {
        "threshold": 3,
        "direction": "above",
        "tip": (
            "{val:.0f} lint warnings detected (unused imports, redefined vars). "
            "Clean up unused imports and fix variable shadowing."
        ),
    },
    "ast_max_depth": {
        "threshold": 6,
        "direction": "above",
        "tip": (
            "Code nesting depth is {val:.0f} levels deep. "
            "Flatten deeply nested blocks using early returns or helper functions."
        ),
    },
    "ast_global_vars": {
        "threshold": 10,
        "direction": "above",
        "tip": (
            "{val:.0f} module-level variables detected. "
            "Prefer encapsulating state inside classes or functions."
        ),
    },
}

LABEL_MESSAGES = {
    "good": (
        "This code is HIGH QUALITY. It has good maintainability, low complexity, "
        "and clean structure. Keep it up!"
    ),
    "medium": (
        "This code is MEDIUM QUALITY. It works but has some areas that could be "
        "cleaner. See the suggestions below."
    ),
    "bad": (
        "This code is LOW QUALITY. It has significant maintainability or "
        "complexity issues. Address the suggestions below before production use."
    ),
}


def tokenize_code(code: str) -> str:
    """Mirror the training-time tokenizer used for TF-IDF."""
    code = re.sub(r'""".*?"""', " ", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", " ", code, flags=re.DOTALL)
    code = re.sub(r'"[^"]*"', " ", code)
    code = re.sub(r"'[^']*'", " ", code)
    code = re.sub(r"#.*", " ", code)
    code = code.replace("_", " ")
    code = re.sub(r"([a-z])([A-Z])", r"\1 \2", code)
    code = re.sub(r"[^a-zA-Z0-9\s]", " ", code)
    code = re.sub(r"\s+", " ", code).strip().lower()
    return code


def get_tree_depth(node: ast.AST, depth: int = 0) -> int:
    """Recursively calculate the depth of an AST tree."""
    if not isinstance(node, ast.AST):
        return depth
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(get_tree_depth(child, depth + 1) for child in children)


def extract_ast_features(code: str) -> dict[str, Any]:
    """Extract the AST metrics used during training."""
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
    comprehensions = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
    ]
    try_blocks = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    global_vars = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
    ]
    arg_counts = [len(fn.args.args) for fn in functions]
    avg_args = float(np.mean(arg_counts)) if arg_counts else 0.0

    def has_docstring(fn: ast.FunctionDef) -> bool:
        return (
            fn.body
            and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)
        )

    doc_ratio = (
        sum(1 for fn in functions if has_docstring(fn)) / len(functions)
        if functions
        else 0.0
    )

    return {
        "ast_functions": len(functions),
        "ast_classes": len(classes),
        "ast_imports": len(imports),
        "ast_max_depth": get_tree_depth(tree),
        "ast_avg_args": avg_args,
        "ast_lambdas": len(lambdas),
        "ast_comprehensions": len(comprehensions),
        "ast_try_blocks": len(try_blocks),
        "ast_returns": len(returns),
        "ast_global_vars": len(global_vars),
        "ast_docstring_ratio": doc_ratio,
        "ast_assert_count": len(asserts),
        "ast_parse_error": 0,
    }


def extract_lint_features(code: str, loc: int = 1) -> dict[str, Any]:
    """Count pyflakes issues, mirroring the training-time lint extractor."""
    try:
        import pyflakes.api as pf_api
        import pyflakes.reporter as pf_rep
    except ImportError:
        return {
            "lint_errors": 0,
            "lint_warnings": 0,
            "lint_total": 0,
            "lint_per_loc": 0.0,
        }

    error_buf = io.StringIO()
    warning_buf = io.StringIO()

    class SplitReporter(pf_rep.Reporter):
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
        pf_api.check(code, filename="<string>", reporter=reporter)
    except Exception:
        pass

    errors = len([line for line in error_buf.getvalue().splitlines() if line.strip()])
    warnings_count = len([line for line in warning_buf.getvalue().splitlines() if line.strip()])
    total = errors + warnings_count
    loc_safe = max(int(loc), 1)

    return {
        "lint_errors": errors,
        "lint_warnings": warnings_count,
        "lint_total": total,
        "lint_per_loc": round(total / loc_safe, 4),
    }


def _compute_static_metrics(code: str) -> dict[str, Any]:
    """Compute the Phase 1 and Phase 2 scalar metrics from raw code."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            mi = float(mi_visit(code, multi=True))
            blocks = cc_visit(code)
            avg_cc = sum(b.complexity for b in blocks) / len(blocks) if blocks else 1.0
            max_cc = max((b.complexity for b in blocks), default=1)
            raw = analyze(code)
    except Exception:
        mi = 0.0
        avg_cc = 0.0
        max_cc = 0
        raw = type("Raw", (), {"loc": 0, "lloc": 0, "sloc": 0, "comments": 0, "blank": 0})()

    loc = int(getattr(raw, "loc", 0) or 0)
    lloc = int(getattr(raw, "lloc", 0) or 0)
    sloc = int(getattr(raw, "sloc", 0) or 0)
    comments = int(getattr(raw, "comments", 0) or 0)
    blank = int(getattr(raw, "blank", 0) or 0)
    comment_ratio = round(comments / lloc, 3) if lloc > 0 else 0.0

    return {
        "mi_score": round(mi, 2),
        "avg_cc": round(avg_cc, 2),
        "max_cc": int(max_cc),
        "loc": loc,
        "lloc": lloc,
        "sloc": sloc,
        "comments": comments,
        "blank": blank,
        "comment_ratio": comment_ratio,
    }


def generate_suggestions(feature_dict: dict[str, Any], label: str) -> dict[str, Any]:
    """Generate the structured quality report used by the web tool."""
    suggestions = []

    for feat, rule in RULES.items():
        val = feature_dict.get(feat)
        if val is None:
            continue

        triggered = (
            (rule["direction"] == "below" and val < rule["threshold"])
            or (rule["direction"] == "above" and val > rule["threshold"])
        )

        if triggered:
            msg = rule["tip"].format(val=val)
            severity = "high" if label == "bad" else "medium"
            suggestions.append(
                {
                    "feature": feat,
                    "value": val,
                    "severity": severity,
                    "message": msg,
                }
            )

    suggestions.sort(key=lambda s: (0 if s["severity"] == "high" else 1, s["feature"]))
    score = max(0, min(100, 100 - len(suggestions) * 10))

    return {
        "label": label,
        "score": score,
        "summary": LABEL_MESSAGES[label],
        "suggestions": suggestions,
        "n_issues": len(suggestions),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_corpus_dataframe(corpus_path: Path) -> pd.DataFrame:
    df = pd.read_csv(corpus_path)
    if "code" not in df.columns:
        raise ValueError(f"{corpus_path} does not contain a code column")
    return df


@lru_cache(maxsize=4)
def _load_tfidf_vectorizer(corpus_path_str: str) -> TfidfVectorizer:
    """Rebuild the TF-IDF vectorizer from the saved corpus."""
    corpus_path = Path(corpus_path_str)
    df = _load_corpus_dataframe(corpus_path)
    tokenized_corpus = [tokenize_code(code) for code in df["code"].fillna("")]

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
        ngram_range=(1, 2),
        token_pattern=TFIDF_TOKEN_PATTERN,
    )
    vectorizer.fit(tokenized_corpus)
    return vectorizer


@lru_cache(maxsize=4)
def _load_artifacts(
    model_path_str: str,
    metadata_path_str: str,
    scaler_path_str: str,
    corpus_path_str: str,
) -> dict[str, Any]:
    """Load and cache model, metadata, scaler parameters, and vectorizer."""
    model_path = Path(model_path_str)
    metadata_path = Path(metadata_path_str)
    scaler_path = Path(scaler_path_str)
    corpus_path = Path(corpus_path_str)

    model = tf.keras.models.load_model(model_path, compile=False)
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    scaler = _load_json(scaler_path)
    vectorizer = _load_tfidf_vectorizer(str(corpus_path))

    feature_names = scaler.get("feature_names")
    if not feature_names:
        raise ValueError(f"{scaler_path} is missing feature_names")

    labels = metadata.get("labels") or ["bad", "good", "medium"]
    n_features = int(metadata.get("n_features", len(feature_names)))

    return {
        "model": model,
        "metadata": metadata,
        "feature_names": feature_names,
        "mean": np.asarray(scaler["mean_"], dtype=np.float32),
        "scale": np.asarray(scaler["scale_"], dtype=np.float32),
        "labels": labels,
        "n_features": n_features,
        "vectorizer": vectorizer,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "scaler_path": str(scaler_path),
        "corpus_path": str(corpus_path),
    }


def _predict_label_from_probs(probs: np.ndarray, labels: list[str]) -> tuple[int, str, float]:
    class_idx = int(np.argmax(probs))
    label = labels[class_idx] if class_idx < len(labels) else str(class_idx)
    confidence = float(probs[class_idx])
    return class_idx, label, confidence


def analyze_python_code_quality(
    code: str,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
) -> dict[str, Any]:
    """Run the full Phase 4 pipeline on raw Python code."""
    if not isinstance(code, str):
        raise TypeError("code must be a string")

    code = code.rstrip("\n")
    artifacts = _load_artifacts(
        str(Path(model_path)),
        str(Path(metadata_path)),
        str(Path(scaler_path)),
        str(Path(corpus_path)),
    )

    static_metrics = _compute_static_metrics(code)
    ast_metrics = extract_ast_features(code)
    lint_metrics = extract_lint_features(code, loc=static_metrics["loc"])

    tokenized = tokenize_code(code)
    tfidf_vector = artifacts["vectorizer"].transform([tokenized])
    tfidf_df = pd.DataFrame(
        tfidf_vector.toarray(),
        columns=[f"tfidf_{t}" for t in artifacts["vectorizer"].get_feature_names_out()],
    )

    raw_feature_dict = {
        **static_metrics,
        **ast_metrics,
        **lint_metrics,
    }
    raw_feature_dict.update(tfidf_df.iloc[0].to_dict())

    feature_names = list(artifacts["feature_names"])
    feature_frame = pd.DataFrame([raw_feature_dict])
    for name in feature_names:
        if name not in feature_frame.columns:
            feature_frame[name] = 0.0
    feature_frame = feature_frame[feature_names].astype(np.float32)

    scaled = (feature_frame.to_numpy() - artifacts["mean"]) / artifacts["scale"]
    probs = artifacts["model"].predict(scaled, verbose=0)[0]
    class_idx, label, confidence = _predict_label_from_probs(probs, artifacts["labels"])

    quality_report = generate_suggestions(raw_feature_dict, label)

    model_metadata = {
        "model_name": artifacts["metadata"].get("model_name", artifacts["model"].name),
        "f1": artifacts["metadata"].get("f1"),
        "accuracy": artifacts["metadata"].get("accuracy"),
        "n_features": artifacts["metadata"].get("n_features", artifacts["n_features"]),
        "n_classes": artifacts["metadata"].get("n_classes", len(artifacts["labels"])),
        "labels": artifacts["metadata"].get("labels", artifacts["labels"]),
    }

    warnings_list = []
    if artifacts["metadata"].get("model_name") and artifacts["metadata"]["model_name"] != artifacts["model"].name:
        warnings_list.append(
            f"Metadata model_name={artifacts['metadata']['model_name']!r} does not match loaded model name={artifacts['model'].name!r}."
        )

    return {
        "input": {
            "characters": len(code),
            "lines": code.count("\n") + 1 if code else 0,
        },
        "model_metadata": model_metadata,
        "model": {
            "name": model_metadata["model_name"],
            "loaded_name": artifacts["model"].name,
            "metadata": model_metadata,
            "path": artifacts["model_path"],
            "metadata_path": artifacts["metadata_path"],
            "scaler_path": artifacts["scaler_path"],
            "corpus_path": artifacts["corpus_path"],
            "n_features": artifacts["n_features"],
            "n_classes": len(artifacts["labels"]),
            "labels": artifacts["labels"],
        },
        "prediction": {
            "class_id": class_idx,
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                artifacts["labels"][i] if i < len(artifacts["labels"]) else str(i): round(float(p), 6)
                for i, p in enumerate(probs)
            },
        },
        "quality_report": quality_report,
        "metrics": {
            **static_metrics,
            **ast_metrics,
            **lint_metrics,
        },
        "warnings": warnings_list,
    }


predict_quality_report = analyze_python_code_quality

