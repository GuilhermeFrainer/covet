"""Summarizes the processed datasets and exports a LaTeX table.

Statistics come in two groups with consistent units:

* Document level (before chunking): raw, dropped and retained documents,
  covariates, words, sentences and words per document. Each retained
  document is counted once, from its original ``text``.
* Chunk level (model input): chunks, and the mean and maximum WordPiece
  tokens per chunk from the ``token_count`` column written by the chunker.

Chunking (``src/processing.py``) splits long documents
into chunks that overlap by two sentences and repeats the document's original
``text`` on every chunk row. Summing per-row text would therefore count long
documents several times, so document-level statistics deduplicate on ``id``.

Usage:
    uv run python scripts/data_prep/summarize_datasets.py
    uv run python scripts/data_prep/summarize_datasets.py --datasets anes trump
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import nltk
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR: Path = PROJECT_ROOT / "data/processed"
CONFIG_DIR: Path = PROJECT_ROOT / "experiments/datasets"
# Next to the other paper tables (scripts/analysis/make_paper_outputs.py).
OUTPUT_PATH: Path = (
    Path.home() / "Downloads" / "covet_paper_outputs" / "tables" / "dataset_summary.tex"
)
TEXT_COL: str = "text"
DOC_ID_COL: str = "id"
TOKEN_COUNT_COL: str = "token_count"

# Datasets reported by default. Trump is reported as the 25k sample, which is
# what the heaviest models train on (see docs/trump_downsampling.md).
DEFAULT_DATASETS: tuple[str, ...] = (
    "anes",
    "fed",
    "gadarian",
    "trump_s25000",
    "yelp_s10000",
)

DATASET_LABELS: dict[str, str] = {
    "anes": "ANES",
    "fed": "Fed",
    "gadarian": "Gadarian",
    "trump": "Trump (full)",
    "trump_s25000": "Trump",
    "yelp_s10000": "Yelp",
}

# Documents before preprocessing. None marks a sample drawn from an already
# preprocessed corpus, whose raw count equals its retained count.
RAW_PATHS: dict[str, Path | None] = {
    "anes": PROJECT_ROOT / "data/interim/anes_2008.parquet",
    "fed": PROJECT_ROOT / "data/interim/fed_communications.parquet",
    "gadarian": PROJECT_ROOT / "data/interim/gadarian.parquet",
    "trump": PROJECT_ROOT / "data/raw/trump_tweets.csv",
    "trump_s25000": None,
    "yelp_s10000": PROJECT_ROOT / "data/interim/yelp_s10000_raw.parquet",
}

# Covariate configs whose name differs from the dataset key.
CONFIG_NAMES: dict[str, str] = {"yelp_s10000": "yelp"}

DOCUMENT_SECTION = "Documents"
CHUNK_SECTION = "Chunks (model input)"

# (statistic key, row label, section, decimals), in table order.
TABLE_ROWS: tuple[tuple[str, str, str, int], ...] = (
    ("raw_docs", "Raw documents", DOCUMENT_SECTION, 0),
    ("dropped_docs", "Dropped", DOCUMENT_SECTION, 0),
    ("documents", "Retained", DOCUMENT_SECTION, 0),
    ("covariates", "Covariates", DOCUMENT_SECTION, 0),
    ("words", "Words", DOCUMENT_SECTION, 0),
    ("sentences", "Sentences", DOCUMENT_SECTION, 0),
    ("words_per_doc", "Words / document", DOCUMENT_SECTION, 1),
    ("chunks", "Chunks", CHUNK_SECTION, 0),
    ("tokens_per_chunk", "Tokens / chunk (mean)", CHUNK_SECTION, 1),
    ("max_tokens_per_chunk", "Tokens / chunk (max)", CHUNK_SECTION, 0),
)


def count_rows(path: Path) -> int:
    """Returns the number of rows in a CSV or Parquet file."""
    lf = pl.scan_csv(path) if path.suffix == ".csv" else pl.scan_parquet(path)
    return lf.select(pl.len()).collect().item()


def count_covariates(dataset_key: str, config_dir: Path = CONFIG_DIR) -> int:
    """Counts the covariates declared in a dataset's YAML config."""
    config_path = config_dir / f"{CONFIG_NAMES.get(dataset_key, dataset_key)}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        covariates = yaml.safe_load(f).get("covariates", {})
    return sum(
        len(covariates.get(group) or [])
        for group in ("numerical", "categorical", "binary")
    )


def count_sentences(text: str) -> int:
    """Counts sentences with the segmenter the chunker uses.

    A non-empty text without terminal punctuation is one sentence, which is
    the common case for short survey answers.
    """
    if not text or not text.strip():
        return 0
    return max(1, len(nltk.sent_tokenize(text)))


def summarize_dataset(
    embeddings_path: Path, raw_docs: int | None, covariates: int
) -> dict[str, Any]:
    """Computes document- and chunk-level statistics for one dataset.

    Args:
        embeddings_path: The dataset's processed ``*_embeddings.parquet``.
        raw_docs: Documents before preprocessing, or None when the dataset is
            a sample of preprocessed documents and nothing was dropped.
        covariates: Number of covariates in the dataset config.

    Returns:
        A mapping from each statistic key in ``TABLE_ROWS`` to its value.
    """
    lf = pl.scan_parquet(embeddings_path)

    chunk_stats = lf.select(
        pl.len().alias("chunks"),
        pl.col(TOKEN_COUNT_COL).mean().alias("tokens_per_chunk"),
        pl.col(TOKEN_COUNT_COL).max().alias("max_tokens_per_chunk"),
    ).collect()

    # Every chunk row repeats its document's original text, so keep one row
    # per document before counting words and sentences.
    documents = (
        lf.select(DOC_ID_COL, TEXT_COL)
        .unique(subset=DOC_ID_COL, keep="first")
        .collect()
    )
    words = documents[TEXT_COL].str.count_matches(r"\S+").sum()
    sentences = sum(count_sentences(text) for text in documents[TEXT_COL])
    n_docs = documents.height
    raw = n_docs if raw_docs is None else raw_docs

    return {
        "raw_docs": raw,
        "dropped_docs": raw - n_docs,
        "documents": n_docs,
        "covariates": covariates,
        "words": words,
        "sentences": sentences,
        "words_per_doc": words / n_docs if n_docs else 0.0,
        "chunks": chunk_stats["chunks"].item(),
        "tokens_per_chunk": chunk_stats["tokens_per_chunk"].item(),
        "max_tokens_per_chunk": chunk_stats["max_tokens_per_chunk"].item(),
    }


def build_table(summaries: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """Lays out formatted statistics, one row per statistic, one column per dataset."""
    return pl.DataFrame(
        {
            "Section": [section for _, _, section, _ in TABLE_ROWS],
            "Statistic": [label for _, label, _, _ in TABLE_ROWS],
            **{
                DATASET_LABELS.get(key, key): [
                    f"{stats[stat]:,.{decimals}f}"
                    for stat, _, _, decimals in TABLE_ROWS
                ]
                for key, stats in summaries.items()
            },
        }
    )


def to_latex(table: pl.DataFrame) -> str:
    """Renders the summary table as a small booktabs LaTeX table."""
    dataset_cols = [c for c in table.columns if c not in ("Section", "Statistic")]
    n_cols = len(dataset_cols) + 1
    lines = [
        r"\begin{table*}[!t]",
        r"\small",
        r"\centering",
        rf"\begin{{tabular}}{{l{'r' * len(dataset_cols)}}}",
        r"\toprule",
        " & ".join(["", *dataset_cols]) + r" \\",
    ]
    section = None
    for row in table.iter_rows(named=True):
        if row["Section"] != section:
            section = row["Section"]
            lines += [
                r"\midrule",
                rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{section}}}}} \\",
            ]
        cells = [row["Statistic"], *(row[c] for c in dataset_cols)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [
        r"\caption{Summary of the datasets. Document statistics count each "
        r"retained document once, before chunking; chunk statistics describe "
        r"the model input, in WordPiece tokens.}",
        r"\label{tab:dataset_summary}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=sorted(DATASET_LABELS),
        help="Datasets to summarize, in column order (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="LaTeX output path (default: "
        "~/Downloads/covet_paper_outputs/tables/dataset_summary.tex).",
    )
    return parser.parse_args()


def main() -> None:
    """Summarizes the selected datasets and writes the LaTeX table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    args = parse_args()

    summaries: dict[str, dict[str, Any]] = {}
    for key in args.datasets:
        path = DATA_DIR / f"{key}_embeddings.parquet"
        if not path.exists():
            logger.error(f"Missing {path}; skipping {key}.")
            continue
        raw_path = RAW_PATHS[key]
        if raw_path is not None and not raw_path.exists():
            logger.error(f"Missing raw source {raw_path}; skipping {key}.")
            continue
        logger.info(f"Summarizing {key}...")
        raw_docs = None if raw_path is None else count_rows(raw_path)
        summaries[key] = summarize_dataset(path, raw_docs, count_covariates(key))

    if not summaries:
        logger.error("No datasets summarized.")
        sys.exit(1)

    table = build_table(summaries)
    # ASCII borders: the Windows console code page cannot encode box drawing.
    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_cell_alignment="RIGHT",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_cols=-1,
        tbl_rows=-1,
        tbl_width_chars=200,
    ):
        print(table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(to_latex(table), encoding="utf-8")
    logger.info(f"LaTeX table saved to {args.output}")


if __name__ == "__main__":
    main()
