"""Recompute topic-quality metrics in the merged results under the current protocol.

Runs scored before `evaluation.EVALUATION_PROTOCOL` were evaluated on padded
topics (short topics filled by repeating their own words, which inflates NPMI
and UMass) and, for BERTopic variants, on raw text that does not contain the
punctuation-stripped words BERTopic builds its topics from. Both are fixed in
`src/evaluation.py` and `src/training.py`.

No retraining is needed: each run's topic words are saved in the qualitative
output files (`output/*.json`) or in its assignment export
(`output/document_assignments/<dataset>/<run_uid>/topics.json`). This script
re-scores them against the run's training texts and updates the merged result
CSVs in place, so every results consumer (dashboard, get_results, paper
outputs) reads the corrected values without changes.

Safeguards:
    * Only `results/*_merged.csv` files are updated. The script refuses to run
      while unmerged raw CSVs exist for the selected datasets: run
      merge_results.py first.
    * Files are read and written as text, so columns that are not updated are
      reproduced exactly.
    * Each row's topic words are matched by (dataset, model name, file
      timestamp). Before any update, the BERTopic topic count must equal the
      stored `n_topics`, and wherever no topic needed padding, the previous
      protocol applied to those words must reproduce the stored NPMI. Any
      failure leaves the whole file untouched.
    * The original file is zipped into `results/archive/` and the archive is
      verified byte for byte before the file is replaced. The new file is
      written to a temporary path, re-read and validated, then moved into place.
    * Previous values are kept as `<metric>_padded` columns, and every updated
      row records `evaluation_protocol`. Rows already at the current protocol
      are skipped, so re-running is safe.

Usage:
    uv run python scripts/analysis/recompute_topic_metrics.py --dry-run
    uv run python scripts/analysis/recompute_topic_metrics.py
    uv run python scripts/analysis/recompute_topic_metrics.py --datasets anes fed
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl  # noqa: E402
from gensim.corpora.dictionary import Dictionary  # noqa: E402
from octis.evaluation_metrics.coherence_metrics import Coherence  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

from src import evaluation  # noqa: E402
from src.data import load_texts  # noqa: E402
from src.utils import load_config  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive"
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSIGNMENTS_DIR = OUTPUT_DIR / "document_assignments"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

RUN_KEY = ("dataset_name", "model_name", "file_timestamp")
DIAGNOSTIC_COLUMNS = ("n_topics_short", "n_topics_unscored", "n_keywords_oov")
PROTOCOL_COLUMN = "evaluation_protocol"
PREVIOUS_SUFFIX = "_padded"
TOPK = 10
TRITOPIC_TYPES = ("tritopic", "fast_tritopic")
# TriTopic's keyword extractor default when a config sets no n-gram range.
TRITOPIC_DEFAULT_NGRAM_RANGE = (1, 2)
VERIFY_METRIC = "c_npmi"
FILE_DATASET = re.compile(r"^(.+?)_(?:standard|stemmed|no_stopword_removal)_")


class RecomputeError(RuntimeError):
    """A file failed a safeguard and was left untouched."""


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
    key = tuple(str(row[column]) for column in RUN_KEY)
    if key in index:
        return index[key]
    if row.get("run_uid"):
        path = ASSIGNMENTS_DIR / row["dataset_name"] / row["run_uid"] / "topics.json"
        if path.exists():
            topics = json.loads(path.read_text(encoding="utf-8"))
            return [t["representation"] for t in topics if t["topic_id"] != -1]
    return None


def removes_rep_stopwords(row: dict, source_file: str = "") -> bool:
    """Whether the run's representation vectorizer removed English stop words."""
    regime = row.get("stopword_removal")
    if regime:
        return regime != "keep_rep_stopwords"
    return (
        "keep_rep_stopwords" not in source_file
        and "no_stopword_removal" not in source_file
    )


def model_config_for(config: dict, manifest: dict | None) -> dict:
    """The run's own model config when recorded, else the current one."""
    return (manifest or {}).get("model_config") or config["model"]


def tokenizer_key(
    row: dict, model_config: dict, source_file: str = "", legacy: bool = False
) -> tuple:
    """Describes how the run's training texts are tokenized for scoring.

    BERTopic variants use their representation vectorizer; under the current
    protocol it reads BERTopic's preprocessed text, under the previous
    (`legacy`) protocol the raw text. TriTopic variants use their keyword
    vectorizer with the n-gram range the run actually used, unchanged.
    """
    if model_config.get("type") in TRITOPIC_TYPES:
        ngram = model_config.get("params", {}).get("keyword_ngram_range")
        return ("tritopic", tuple(ngram or TRITOPIC_DEFAULT_NGRAM_RANGE), False)
    return ("bertopic", removes_rep_stopwords(row, source_file), not legacy)


def tokenize(texts: list[str], key: tuple) -> list[list[str]]:
    family, option, preprocess = key
    if family == "tritopic":
        analyzer = CountVectorizer(
            stop_words="english", ngram_range=option
        ).build_analyzer()
    else:
        analyzer = CountVectorizer(
            stop_words="english" if option else None
        ).build_analyzer()
    if preprocess:
        tokens = (analyzer(evaluation.bertopic_preprocess(text)) for text in texts)
    else:
        tokens = (analyzer(text) for text in texts)
    return [t for t in tokens if t]


def _float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _format(value) -> str | None:
    """Formats a value the way the result CSVs store it."""
    if value is None:
        return None
    if isinstance(value, float):
        return "NaN" if math.isnan(value) else repr(value)
    return str(value)


class Corpora:
    """Tokenized training texts, cached per dataset sample and tokenizer."""

    def __init__(self):
        self._cache = {}

    def get(self, config: dict, seed: int, key: tuple):
        experiment = config["experiment"]
        sampled = experiment.get("sample_size") is not None
        cache_key = (
            experiment["dataset_path"],
            experiment.get("text_col", "text"),
            experiment.get("sample_size"),
            seed if sampled else None,
            key,
        )
        if cache_key not in self._cache:
            texts = tokenize(load_texts(config, seed), key)
            self._cache[cache_key] = (texts, Dictionary(texts))
        return self._cache[cache_key]


def plan_row(row: dict, index: dict, source_file: str) -> dict:
    """Classifies a row and gathers what is needed to rescore it."""
    if row.get(PROTOCOL_COLUMN) == evaluation.EVALUATION_PROTOCOL:
        return {"status": "current"}
    raw_topics = topic_words_for(row, index)
    if raw_topics is None:
        return {"status": "topic words not found"}
    try:
        config = load_config(row["experiment_id"], EXPERIMENTS_DIR)
    except FileNotFoundError:
        return {"status": "experiment config not found"}
    model_config = model_config_for(config, run_manifest(row))
    topics = [t for t in map(evaluation.select_topic_words, raw_topics) if t]
    is_tritopic = model_config.get("type") in TRITOPIC_TYPES
    stored_topics = _float(row.get("n_topics"))
    # TriTopic rows from before 2026-10 store the requested count in n_topics.
    if (
        not is_tritopic
        and stored_topics is not None
        and stored_topics != len(raw_topics)
    ):
        return {
            "status": "topic count mismatch",
            "detail": f"stored n_topics={row.get('n_topics')}, "
            f"saved topics={len(raw_topics)}",
        }
    return {
        "status": "update",
        "config": config,
        "model_config": model_config,
        "topics": topics,
    }


def legacy_npmi(topics, texts) -> float:
    """NPMI as the previous protocol computed it for unpadded topics."""
    return Coherence(texts=texts, topk=TOPK, measure=VERIFY_METRIC).score(
        {"topics": topics}
    )


def rescore(row: dict, plan: dict, corpora: Corpora, source_file: str) -> dict:
    """New metric values and diagnostics, plus the legacy check outcome."""
    config, topics = plan["config"], plan["topics"]
    seed = int(row["random_state"])
    texts, dictionary = corpora.get(
        config, seed, tokenizer_key(row, plan["model_config"], source_file)
    )
    output = {"topics": topics}
    values = evaluation.topic_diagnostics(output, dictionary=dictionary, topk=TOPK)
    experiment = config["experiment"]
    for measure in experiment.get("coherence_metrics", []):
        values[measure] = evaluation.compute_coherence(
            output, texts, measure=measure, topk=TOPK, dictionary=dictionary
        )
    for measure in experiment.get("diversity_metrics", []):
        values[measure] = evaluation.compute_diversity(measure, output, topk=TOPK)

    verified = None
    stored = _float(row.get(VERIFY_METRIC))
    unpadded = topics and all(len(t) >= TOPK for t in topics)
    if unpadded and stored is not None and math.isfinite(stored):
        legacy_texts, _ = corpora.get(
            config,
            seed,
            tokenizer_key(row, plan["model_config"], source_file, legacy=True),
        )
        recomputed = legacy_npmi(topics, legacy_texts)
        verified = math.isclose(recomputed, stored, rel_tol=1e-9, abs_tol=1e-12)
        if not verified:
            values["_legacy_mismatch"] = (stored, recomputed)
    values["_verified"] = verified
    return values


def file_dataset(name: str) -> str:
    """Dataset of a result file named `<dataset>_<result type>_...`.

    Datasets may contain underscores (e.g. trump_s25000), so the name is cut
    at the result-type marker rather than at the first underscore.
    """
    match = FILE_DATASET.match(name)
    return match.group(1) if match else name.split("_")[0]


def raw_files_for(datasets: set[str] | None) -> list[Path]:
    """Unmerged top-level result CSVs, optionally limited to some datasets."""
    raw = []
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        if "_merged" in path.name:
            continue
        if datasets is None or file_dataset(path.name) in datasets:
            raw.append(path)
    return raw


def merged_files_for(datasets: set[str] | None) -> list[Path]:
    files = sorted(RESULTS_DIR.glob("*_merged.csv"))
    if datasets is None:
        return files
    return [p for p in files if file_dataset(p.name) in datasets]


def archive(path: Path, stamp: str) -> Path:
    """Zips the file into results/archive/ and verifies the archived bytes."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    target = ARCHIVE_DIR / f"{path.stem}_pre_recompute_{stamp}.zip"
    if target.exists():
        raise RecomputeError(f"archive {target} already exists")
    original = path.read_bytes()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(path.name, original)
    with zipfile.ZipFile(target) as bundle:
        if bundle.read(path.name) != original:
            raise RecomputeError(f"archive {target} does not match {path.name}")
    return target


def updated_frame(frame: pl.DataFrame, updates: dict[int, dict]) -> pl.DataFrame:
    """Applies new values by row index; keeps previous metric values."""
    metrics = sorted(
        {
            key
            for values in updates.values()
            for key in values
            if not key.startswith("_") and key not in DIAGNOSTIC_COLUMNS
        }
    )
    columns = {name: frame[name].to_list() for name in frame.columns}
    height = frame.height
    for name in (
        *metrics,
        *(f"{m}{PREVIOUS_SUFFIX}" for m in metrics),
        *DIAGNOSTIC_COLUMNS,
        PROTOCOL_COLUMN,
    ):
        columns.setdefault(name, [None] * height)
    for index, values in updates.items():
        for metric in metrics:
            if metric in values:
                previous = columns[metric][index]
                columns[f"{metric}{PREVIOUS_SUFFIX}"][index] = previous
                columns[metric][index] = _format(values[metric])
        for name in DIAGNOSTIC_COLUMNS:
            columns[name][index] = _format(values.get(name))
        columns[PROTOCOL_COLUMN][index] = evaluation.EVALUATION_PROTOCOL
    return pl.DataFrame(
        {
            name: pl.Series(name, values, dtype=pl.String)
            for name, values in columns.items()
        }
    )


def changed_columns(updates: dict[int, dict]) -> set[str]:
    """Columns the update may modify: rescored metrics and bookkeeping."""
    metrics = {
        key
        for values in updates.values()
        for key in values
        if not key.startswith("_") and key not in DIAGNOSTIC_COLUMNS
    }
    return (
        metrics
        | {f"{m}{PREVIOUS_SUFFIX}" for m in metrics}
        | set(DIAGNOSTIC_COLUMNS)
        | {PROTOCOL_COLUMN}
    )


def validate(
    written: Path, original: pl.DataFrame, expected: pl.DataFrame, changed: set
):
    """Checks the intended content, then that the file on disk matches it."""
    if expected.height != original.height:
        raise RecomputeError(f"{written.name}: row count changed")
    if expected.columns[: original.width] != original.columns:
        raise RecomputeError(f"{written.name}: column order changed")
    for name in original.columns:
        if name not in changed and not expected[name].equals(original[name]):
            raise RecomputeError(f"{written.name}: column {name} would change")
    reread = pl.read_csv(written, infer_schema_length=0)
    if reread.columns != expected.columns or not reread.equals(expected):
        raise RecomputeError(f"{written.name}: file on disk differs from intent")


def process_file(
    path: Path, index: dict, corpora: Corpora, dry_run: bool, stamp: str
) -> dict:
    """Rescores one merged file; writes it only if every safeguard passes."""
    original_bytes = path.read_bytes()
    frame = pl.read_csv(path, infer_schema_length=0)
    missing = [
        c for c in (*RUN_KEY, "experiment_id", "random_state") if c not in frame.columns
    ]
    if missing:
        raise RecomputeError(f"{path.name}: missing columns {missing}")
    if frame.select(RUN_KEY).is_duplicated().any():
        raise RecomputeError(f"{path.name}: duplicate run keys")

    rows = frame.to_dicts()
    plans = [plan_row(row, index, path.name) for row in rows]
    report = {"file": path.name, "rows": len(rows)}
    for plan in plans:
        report[plan["status"]] = report.get(plan["status"], 0) + 1
    mismatched = [
        f"{rows[i]['model_name']}: {plan['detail']}"
        for i, plan in enumerate(plans)
        if plan["status"] == "topic count mismatch"
    ]
    if mismatched:
        raise RecomputeError(
            f"{path.name}: topic words do not match {len(mismatched)} rows, e.g. "
            + "; ".join(mismatched[:3])
        )

    updates = {}
    todo = [i for i, plan in enumerate(plans) if plan["status"] == "update"]
    for number, i in enumerate(todo, start=1):
        print(f"  [{number}/{len(todo)}] {rows[i]['model_name']}", flush=True)
        updates[i] = rescore(rows[i], plans[i], corpora, path.name)
    failures = [
        f"{rows[i]['model_name']} (stored {v['_legacy_mismatch'][0]:.12g}, "
        f"recomputed {v['_legacy_mismatch'][1]:.12g})"
        for i, v in updates.items()
        if v["_verified"] is False
    ]
    report["verified"] = sum(v["_verified"] is True for v in updates.values())
    report["unverifiable"] = sum(v["_verified"] is None for v in updates.values())
    if failures:
        raise RecomputeError(
            f"{path.name}: previous-protocol NPMI not reproduced for "
            f"{len(failures)} rows, e.g. " + "; ".join(failures[:3])
        )
    if not updates or dry_run:
        return report

    expected = updated_frame(frame, updates)
    temporary = path.with_name(f".{path.name}.tmp")
    expected.write_csv(temporary)
    try:
        validate(temporary, frame, expected, changed_columns(updates))
        if path.read_bytes() != original_bytes:
            raise RecomputeError(f"{path.name}: changed on disk during the run")
        archived = archive(path, stamp)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    report["archive"] = archived.name
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", help="Limit to these datasets.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every computation and safeguard, but write nothing.",
    )
    args = parser.parse_args()
    datasets = set(args.datasets) if args.datasets else None
    # Per-topic exclusions are recorded in n_topics_unscored; skip the log noise.
    logging.getLogger("pipeline").setLevel(logging.ERROR)

    raw = raw_files_for(datasets)
    if raw:
        print("Unmerged result files exist; run merge_results.py first:")
        for path in raw[:10]:
            print(f"  {path.name}")
        if len(raw) > 10:
            print(f"  ... and {len(raw) - 10} more")
        sys.exit(1)

    files = merged_files_for(datasets)
    if not files:
        print("No merged result files found.")
        sys.exit(1)
    index = load_topic_words()
    corpora = Corpora()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed = False
    for path in files:
        print(f"{path.name}:", flush=True)
        try:
            report = process_file(path, index, corpora, args.dry_run, stamp)
        except RecomputeError as error:
            failed = True
            print(f"  NOT UPDATED: {error}")
            continue
        details = ", ".join(f"{k}={v}" for k, v in report.items() if k != "file")
        action = "dry run, nothing written" if args.dry_run else "updated"
        print(f"  {action}: {details}")
    if failed:
        print("Some files were left untouched; see the messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
