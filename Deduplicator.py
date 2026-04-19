"""
deduplicator.py
Removes duplicate code files using hash-based deduplication.
Forks and copy-pasted files are very common on GitHub.
"""

import hashlib

import pandas as pd

from Logger import get_logger

log = get_logger("deduplicator")

#function to compute the MD5 hash of a code string, ignoring encoding errors
def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df["content_hash"] = df["code"].apply(_md5)
    df = df.drop_duplicates(subset="content_hash")
    df = df.drop(columns=["content_hash"])
    after = len(df)
    log.info(f"Deduplication: {before} -> {after} files ({before - after} removed)")
    return df
