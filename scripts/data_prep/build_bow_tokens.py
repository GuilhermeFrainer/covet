# -*- coding: utf-8 -*-
"""Tokenizes a dataset's texts for the STM bag-of-words.

STM's vocabulary must match the vocabulary BERTopic variants draw topic words
from, so tokens come from `src.evaluation.representation_tokens`, the same
function c-TF-IDF and coherence evaluation use. Rows are read from the
embeddings file the neural models train on, so both families see the same
documents and metadata.

`scripts/r_scripts/build_bow.R` reads the output and only splits it on
whitespace. Documents left without tokens are kept here and dropped there,
per text column.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.logger_config as logger_config
from src.evaluation import representation_tokens

TEXT_COLUMNS = ("clean_text", "clean_text_stemmed")
TOKEN_COLUMN_PREFIX = "bow_tokens_"


def build_bow_tokens(df: pl.DataFrame) -> pl.DataFrame:
    """Adds one whitespace-joined token column per text column.

    Args:
        df: Dataset rows with the columns in `TEXT_COLUMNS`.

    Returns:
        `df` without embedding columns, plus `bow_tokens_<text column>`.
    """
    df = df.select([c for c in df.columns if "embedding" not in c])
    return df.with_columns(
        pl.Series(
            f"{TOKEN_COLUMN_PREFIX}{column}",
            [" ".join(tokens) for tokens in representation_tokens(df[column])],
        )
        for column in TEXT_COLUMNS
    )


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. fed, trump_s25000, yelp_s10000.",
    )
    args = parser.parse_args()

    logger = logger_config.setup_logging("build_bow_tokens", PROJECT_ROOT / "logs")
    input_path = PROJECT_ROOT / f"data/processed/{args.dataset}_embeddings.parquet"
    output_path = PROJECT_ROOT / f"data/interim/{args.dataset}_bow_tokens.parquet"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    schema = pl.read_parquet_schema(input_path)
    columns = [c for c in schema if "embedding" not in c]
    df = build_bow_tokens(pl.read_parquet(input_path, columns=columns))

    for column in TEXT_COLUMNS:
        tokens = df[f"{TOKEN_COLUMN_PREFIX}{column}"]
        logger.info(
            f"{column}: {(tokens == '').sum()} of {len(df)} documents have no "
            f"tokens; {(tokens == 'emptydoc').sum()} are only 'emptydoc'."
        )

    df.write_parquet(output_path)
    logger.info(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
