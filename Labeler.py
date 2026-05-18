"""
labeler.py
Auto-labels code quality using Radon static analysis metrics.

Labels:
  good   - high maintainability, low complexity
  medium - borderline
  bad    - low maintainability or high complexity
  error  - could not be parsed (dropped before saving)
"""

import warnings

import pandas as pd
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze

from config import (
    CC_BAD_THRESHOLD,
    CC_GOOD_THRESHOLD,
    MI_BAD_THRESHOLD,
    MI_GOOD_THRESHOLD,
)
from Logger import get_logger

log = get_logger("labeler")

#function to compute static metrics for one code string, returning None if parsing fails
def _compute_metrics(code: str) -> dict | None:
    """
    Return a dict of static metrics for one code string.
    Returns None if the code cannot be parsed.
    """
    try:
        #This temporarily ignores SyntaxWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            #maintainability index (MI) is a composite metric that combines several factors,
            # including cyclomatic complexity, lines of code, and comment density.
            mi = mi_visit(code, multi=True)

            blocks = cc_visit(code)
            avg_cc = sum(b.complexity for b in blocks) / len(blocks) if blocks else 1.0
            max_cc = max((b.complexity for b in blocks), default=1)
            #raw metrics include lines of code (loc), logical lines of code (lloc), 
            # source lines of code (sloc),
            # number of comment lines, and number of blank lines
            raw = analyze(code)

        return {
            "mi_score": round(mi, 2),
            "avg_cc": round(avg_cc, 2),
            "max_cc": max_cc,
            "loc": raw.loc,
            "lloc": raw.lloc,
            "sloc": raw.sloc,
            "comments": raw.comments,
            "blank": raw.blank,
            "comment_ratio": round(raw.comments / raw.lloc if raw.lloc > 0 else 0, 3),
        }
    except Exception:
        return None

#function to assign a quality label based on MI and average CC, using the defined thresholds
def _assign_label(mi: float, avg_cc: float) -> str:
    if mi >= MI_GOOD_THRESHOLD and avg_cc <= CC_GOOD_THRESHOLD:
        return "good"
    if mi < MI_BAD_THRESHOLD or avg_cc > CC_BAD_THRESHOLD:
        return "bad"
    return "medium"

#main function to label a DataFrame of code files, 
# adding metric columns and a quality_label column
def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds metric columns and a quality_label column to df.
    Rows that fail parsing are dropped.
    """
    log.info(f"Labeling {len(df)} files...")

    metrics_rows = []
    kept_indexes = []
    failed = 0
    #iterate over each row in the DataFrame, compute metrics for the code, and keep track of which rows succeed or fail
    for index, row in df.iterrows():
        metrics = _compute_metrics(row["code"])
        if metrics is None:
            failed += 1
            continue
        metrics_rows.append(metrics)
        kept_indexes.append(index)

    if failed:
        log.warning(f"  Dropping {failed} files that could not be parsed")
    # If all files fail to parse, return an empty DataFrame with the expected columns
    if not metrics_rows:
        log.warning("  No files could be parsed; returning empty labeled dataset")
        empty = df.iloc[0:0].copy()
        for column in [
            "mi_score",
            "avg_cc",
            "max_cc",
            #loc = lines of code, lloc = logical lines of code, sloc = source lines of code
            "loc",
            "lloc",
            "sloc",
            "comments",
            "blank",
            "comment_ratio",
            "quality_label",
        ]:
            if column not in empty.columns:
                empty[column] = pd.Series(dtype="object")
        return empty

    metrics_df = pd.DataFrame(metrics_rows, index=kept_indexes)
    df = df.loc[kept_indexes].join(metrics_df)

    df["quality_label"] = df.apply(
        lambda r: _assign_label(r["mi_score"], r["avg_cc"]), axis=1
    )

    dist = df["quality_label"].value_counts().to_dict()
    log.info(f"  Label distribution: {dist}")

    return df
