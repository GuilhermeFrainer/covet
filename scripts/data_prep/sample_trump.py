# -*- coding: utf-8 -*-
"""Samples a fixed subset of the preprocessed and embedded Trump dataset.

Trump preprocessing and embedding are per-document, so sampling the finished
embeddings file yields the same rows as sampling before preprocessing. See
docs/trump_downsampling.md for the rationale.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.logger_config as logger_config
from src.data import sample_from_lf

INPUT_EMBEDDINGS = PROJECT_ROOT / "data/processed/trump_embeddings.parquet"
TEXT_COLUMNS = ("clean_text", "clean_text_stemmed")
DEFAULT_SAMPLE_SIZE = 25000
RANDOM_STATE = 36201624


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Sample a fixed subset of the Trump dataset."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of documents to sample (default: 25000).",
    )
    args = parser.parse_args()

    dataset_name = f"trump_s{args.n}"
    output_embeddings = (
        PROJECT_ROOT / f"data/processed/{dataset_name}_embeddings.parquet"
    )
    output_processed = PROJECT_ROOT / f"data/interim/{dataset_name}_processed.parquet"

    logger = logger_config.setup_logging("sample_trump", PROJECT_ROOT / "logs")
    logger.info(f"Starting Trump sampling for {dataset_name}...")

    if not INPUT_EMBEDDINGS.exists():
        logger.error(
            f"Input file not found: {INPUT_EMBEDDINGS}. "
            "Preprocess and embed the trump dataset first."
        )
        sys.exit(1)

    lf = pl.scan_parquet(INPUT_EMBEDDINGS)

    logger.info(f"Sampling {args.n} documents with seed {RANDOM_STATE}...")
    sampled_df = sample_from_lf(lf, n=args.n, seed=RANDOM_STATE).collect().sort("index")

    # The full file is already row-aligned; verify the sample kept that.
    if sampled_df["index"].n_unique() != args.n:
        logger.error("Sampled rows do not have unique 'index' values.")
        sys.exit(1)
    for col in TEXT_COLUMNS:
        n_empty = sampled_df.filter(pl.col(col).str.strip_chars() == "").height
        if n_empty:
            logger.error(f"Sample has {n_empty} empty '{col}' rows.")
            sys.exit(1)

    logger.info(f"Saving sampled embeddings to {output_embeddings}...")
    sampled_df.write_parquet(output_embeddings)

    embedding_cols = [c for c in sampled_df.columns if "embedding" in c]
    logger.info(f"Saving sampled processed text to {output_processed}...")
    sampled_df.drop(embedding_cols).write_parquet(output_processed)

    logger.info(f"Trump sampling completed: {sampled_df.height} documents.")


if __name__ == "__main__":
    main()
