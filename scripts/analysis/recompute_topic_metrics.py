"""Recompute topic-quality metrics for saved runs under the current protocol.

Runs scored before `evaluation.EVALUATION_PROTOCOL` were evaluated on padded
topics (short topics filled by repeating their own words, which inflates NPMI
and UMass) and, for BERTopic variants, on raw text that does not contain the
punctuation-stripped words BERTopic builds its topics from. Both are fixed in
`src/evaluation.py` and `src/training.py`.

No retraining is needed: each run's topic words are saved in the qualitative
output files (`output/*.json`) or in its assignment export
(`output/document_assignments/<dataset>/<run_uid>/topics.json`). This script
re-scores them against the run's training texts. Existing results are never
modified: one row per run goes to a sidecar CSV under `results/derived/`, which
the results loaders and merge_results.py do not scan.

Usage:
    uv run python scripts/analysis/recompute_topic_metrics.py --dry-run
    uv run python scripts/analysis/recompute_topic_metrics.py
    uv run python scripts/analysis/recompute_topic_metrics.py --datasets anes fed
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import CountVectorizer

from src import evaluation
from src.data import load_texts
from src.utils import load_config

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSIGNMENTS_DIR = OUTPUT_DIR / "document_assignments"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DEFAULT_OUTPUT = RESULTS_DIR / "derived" / "topic_metrics_recomputed.csv"

IDENTITY_COLUMNS = [
    "run_uid",
    "dataset_name",
    "experiment_id",
    "model_name",
    "random_state",
    "file_timestamp",
    "stopword_removal",
]
TRITOPIC_TYPES = ("tritopic", "fast_tritopic")
# TriTopic's keyword extractor default when a config sets no n-gram range.
TRITOPIC_DEFAULT_NGRAM_RANGE = (1, 2)


def load_result_rows(datasets=None) -> pl.DataFrame:
    """One row per run from top-level result CSVs, merged rows first."""
    frames = []
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        frame = pl.read_csv(path, infer_schema_length=0)
        frames.append(frame.with_columns(pl.lit(path.name).alias("source_file")))
    if not frames:
        return pl.DataFrame()
    rows = pl.concat(frames, how="diagonal").with_columns(
        pl.col("source_file").str.contains("_merged").alias("is_merged")
    )
    if datasets:
        rows = rows.filter(pl.col("dataset_name").is_in(datasets))
    return rows.sort("is_merged", descending=True).unique(
        subset=["dataset_name", "model_name", "file_timestamp"],
        keep="first",
        maintain_order=True,
    )


def load_topic_words() -> dict[tuple, list[list[str]]]:
    """Maps (dataset, model_name, file_timestamp) to raw topic word lists."""
    index: dict[tuple, dict[int, list]] = {}
    for path in sorted(OUTPUT_DIR.glob("*.json")):
        for topic in json.loads(path.read_text(encoding="utf-8")):
            if topic.get("topic_id") == -1:
                continue
            key = (
                topic.get("dataset_name"),
                topic.get("model_id"),
                str(topic.get("file_timestamp")),
            )
            index.setdefault(key, {})[topic["topic_id"]] = topic["representation"]
    return {key: [t[i] for i in sorted(t)] for key, t in index.items()}


def run_manifest(row: dict) -> dict | None:
    """Returns the run's assignment manifest, if it was exported."""
    if not row.get("run_uid"):
        return None
    path = ASSIGNMENTS_DIR / row["dataset_name"] / row["run_uid"] / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def topic_words_for(row: dict, index: dict) -> list[list[str]] | None:
    key = (row["dataset_name"], row["model_name"], str(row["file_timestamp"]))
    if key in index:
        return index[key]
    if row.get("run_uid"):
        path = ASSIGNMENTS_DIR / row["dataset_name"] / row["run_uid"] / "topics.json"
        if path.exists():
            topics = json.loads(path.read_text(encoding="utf-8"))
            return [t["representation"] for t in topics if t["topic_id"] != -1]
    return None


def removes_rep_stopwords(row: dict) -> bool:
    """Whether the run's representation vectorizer removed English stop words."""
    regime = row.get("stopword_removal")
    if regime:
        return regime != "keep_rep_stopwords"
    source = row.get("source_file") or ""
    return "keep_rep_stopwords" not in source and "no_stopword_removal" not in source


def tokenizer_key(row: dict, config: dict, manifest: dict | None) -> tuple:
    """Describes how the run's training texts must be tokenized for scoring.

    BERTopic variants use their representation vectorizer on BERTopic's
    preprocessed text. TriTopic variants use their keyword vectorizer, with the
    n-gram range the run actually used (from its manifest when available).
    """
    model_config = (manifest or {}).get("model_config") or config["model"]
    if model_config.get("type") in TRITOPIC_TYPES:
        ngram = model_config.get("params", {}).get("keyword_ngram_range")
        return ("tritopic", tuple(ngram or TRITOPIC_DEFAULT_NGRAM_RANGE))
    return ("bertopic", removes_rep_stopwords(row))


def tokenize(texts: list[str], key: tuple) -> list[list[str]]:
    if key[0] == "tritopic":
        analyzer = CountVectorizer(
            stop_words="english", ngram_range=key[1]
        ).build_analyzer()
        tokens = (analyzer(text) for text in texts)
    else:
        analyzer = CountVectorizer(
            stop_words="english" if key[1] else None
        ).build_analyzer()
        tokens = (analyzer(evaluation.bertopic_preprocess(text)) for text in texts)
    return [t for t in tokens if t]


def score_run(row: dict, index: dict, corpora: dict, dry_run: bool) -> dict:
    """Returns one output record, with a skip reason when it can't be scored."""
    record = {column: row.get(column) for column in IDENTITY_COLUMNS}
    record["skip_reason"] = None
    raw_topics = topic_words_for(row, index)
    if raw_topics is None:
        return {**record, "skip_reason": "topic words not found"}
    try:
        config = load_config(row["experiment_id"], EXPERIMENTS_DIR)
    except FileNotFoundError:
        return {**record, "skip_reason": "experiment config not found"}
    if dry_run:
        return record

    experiment = config["experiment"]
    sampled = experiment.get("sample_size") is not None
    key = (
        experiment["dataset_path"],
        experiment.get("text_col", "text"),
        experiment.get("sample_size"),
        int(row["random_state"]) if sampled else None,
        tokenizer_key(row, config, run_manifest(row)),
    )
    if key not in corpora:
        texts = tokenize(load_texts(config, int(row["random_state"])), key[-1])
        corpora[key] = (texts, Dictionary(texts))
    texts, dictionary = corpora[key]

    model_output = {
        "topics": [t for t in map(evaluation.select_topic_words, raw_topics) if t]
    }
    record.update(evaluation.topic_diagnostics(model_output, dictionary=dictionary))
    for measure in experiment.get("coherence_metrics", []):
        record[measure] = evaluation.compute_coherence(
            model_output, texts, measure=measure, dictionary=dictionary
        )
    for measure in experiment.get("diversity_metrics", []):
        record[measure] = evaluation.compute_diversity(measure, model_output)
    record["evaluation_protocol"] = evaluation.EVALUATION_PROTOCOL
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", help="Limit to these datasets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which runs can be matched to topic words and configs.",
    )
    args = parser.parse_args()

    rows = load_result_rows(args.datasets)
    index = load_topic_words()
    corpora: dict = {}
    records = []
    for number, row in enumerate(rows.iter_rows(named=True), start=1):
        if not args.dry_run:
            print(f"[{number}/{rows.height}] {row['dataset_name']} {row['model_name']}")
        records.append(score_run(row, index, corpora, args.dry_run))

    scored = [r for r in records if r["skip_reason"] is None]
    skipped = [r for r in records if r["skip_reason"] is not None]
    verb = "Matched" if args.dry_run else "Scored"
    print(f"{verb} {len(scored)} of {len(records)} runs.")
    if skipped:
        reasons = pl.DataFrame(skipped).group_by("skip_reason").len()
        print("Skipped:")
        for reason, count in reasons.iter_rows():
            print(f"  {count:4d}  {reason}")
    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(scored).drop("skip_reason").write_csv(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
