"""
main.py - Phase 1: Data Collection Pipeline
==============================================
Run:
    python main.py

Outputs:
    output/raw_code.csv        - all scraped files (no labels)
    output/labeled_dataset.csv - deduplicated + quality-labeled
    output/scrape.log          - full run log
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import LABELED_CSV, OUTPUT_DIR, RAW_CSV
from Deduplicator import deduplicate
from Labeler import label_dataframe
from Logger import get_logger
from scraper import connect, scrape_all

log = get_logger("main")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 55)
    log.info("  Phase 1: Code Quality Dataset Collection")
    log.info("=" * 55)
    g = connect()

    log.info("\n[Step 1/4] Scraping GitHub repositories...")
    raw_files = scrape_all(g)

    if not raw_files:
        log.error("No files scraped. Check your token and search config.")
        return

    raw_df = pd.DataFrame(raw_files)
    raw_df.to_csv(RAW_CSV, index=False)
    log.info(f"Raw data saved -> {RAW_CSV}  ({len(raw_df)} rows)")

    log.info("\n[Step 2/4] Deduplicating...")
    clean_df = deduplicate(raw_df)

    log.info("\n[Step 3/4] Computing quality metrics and labels...")
    labeled_df = label_dataframe(clean_df)

    log.info("\n[Step 4/4] Saving labeled dataset...")

    save_cols = [
        "repo",
        "repo_url",
        "stars",
        "file_path",
        "file_size",
        "mi_score",
        "avg_cc",
        "max_cc",
        "loc",
        "lloc",
        "sloc",
        "comments",
        "blank",
        "comment_ratio",
        "quality_label",
        "code",
    ]
    save_cols = [c for c in save_cols if c in labeled_df.columns]
    labeled_df[save_cols].to_csv(LABELED_CSV, index=False)

    log.info("\n" + "=" * 55)
    log.info("  Phase 1 Complete")
    log.info("=" * 55)
    log.info(f"  Total files scraped    : {len(raw_df)}")
    log.info(f"  After deduplication     : {len(clean_df)}")
    log.info(f"  After labeling          : {len(labeled_df)}")

    if "quality_label" in labeled_df.columns and not labeled_df.empty:
        dist = labeled_df["quality_label"].value_counts()
        for label, count in dist.items():
            pct = 100 * count / len(labeled_df)
            log.info(f"    {label:<8}: {count:>4} files  ({pct:.1f}%)")

    log.info(f"\n  Raw CSV    -> {RAW_CSV}")
    log.info(f"  Labeled CSV -> {LABELED_CSV}")
    log.info("  Log file   -> output/scrape.log")


if __name__ == "__main__":
    main()
