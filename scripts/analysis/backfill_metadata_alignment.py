"""Backfill topic–metadata alignment (AMI) for runs with exported assignments.

Runs executed before the metric existed can be scored without retraining when
their per-document assignments were exported. Each run's source dataset is
located by the SHA-256 recorded in its manifest (not by path, which is often
a cluster scratch path), and the assignments file is verified against its
recorded checksum. Existing artifacts are never modified: the results go to a
long-format sidecar CSV keyed by ``run_uid``, under ``results/derived/`` so the
results loaders and merge_results.py (which scan only the top level) do not
mistake it for run results. The dashboard joins it by ``run_uid`` explicitly.

Usage:
    uv run python scripts/analysis/backfill_metadata_alignment.py
    uv run python scripts/analysis/backfill_metadata_alignment.py --datasets fed
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl

from src.document_assignments import file_checksum
from src.metadata_alignment import metadata_alignment

ASSIGNMENTS_DIR = PROJECT_ROOT / "output" / "document_assignments"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
# Kept out of the top level of results/: the dashboard loads every top-level CSV
# as run results, and merge_results.py would absorb and archive it as a raw run.
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "derived" / "metadata_alignment_backfill.csv"
)


def covariate_columns(covariates) -> list[str]:
    """Flattens a manifest's covariate spec in pipeline order."""
    if isinstance(covariates, list):
        return covariates
    return (
        covariates.get("numerical", [])
        + covariates.get("categorical", [])
        + covariates.get("binary", [])
    )


def index_local_datasets(data_dir: Path) -> dict[str, Path]:
    """Maps SHA-256 -> local parquet path."""
    return {file_checksum(p): p for p in sorted(data_dir.glob("*.parquet"))}


def score_run(manifest_path: Path, datasets: dict, cache: dict) -> dict:
    """Returns one result record (with a skip reason when it can't be scored)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = {
        "run_uid": manifest.get("run_uid"),
        "dataset_name": manifest.get("dataset_name"),
        "experiment_id": manifest.get("experiment_id"),
        "model_id": manifest.get("model_id"),
        "random_state": manifest.get("random_state"),
        "requested_topic_setting": manifest.get("requested_topic_setting"),
        "stopword_removal": manifest.get("stopword_removal"),
        "skip_reason": None,
    }
    artifact = manifest.get("artifacts", {}).get("assignments.parquet")
    if manifest.get("status") != "success" or artifact is None:
        return {**record, "skip_reason": f"status={manifest.get('status')}"}
    assignments_path = manifest_path.parent / artifact["path"]
    if file_checksum(assignments_path) != artifact["sha256"]:
        return {**record, "skip_reason": "assignments checksum mismatch"}
    source = datasets.get(manifest["input"]["dataset_sha256"])
    if source is None:
        return {**record, "skip_reason": "source dataset not found locally"}

    columns = covariate_columns(manifest["input"]["covariates"])
    key = (source, tuple(columns))
    if key not in cache:
        cache[key] = pl.read_parquet(source, columns=columns).with_row_index(
            "source_row_ordinal"
        )
    assignments = pl.read_parquet(
        assignments_path, columns=["source_row_ordinal", "topic_id"]
    ).with_row_index("order")
    joined = (
        assignments.with_columns(pl.col("source_row_ordinal").cast(pl.UInt32))
        .join(cache[key], on="source_row_ordinal", how="left", validate="1:1")
        .sort("order")
    )
    result = metadata_alignment(joined["topic_id"], joined.select(columns))

    recorded = manifest.get("metadata_alignment", {}).get("meta_ami_mean")
    return {
        **record,
        "meta_ami_mean": result["meta_ami_mean"],
        "meta_ami_by_covariate": result["meta_ami_by_covariate"],
        "matches_run_time_value": None
        if recorded is None
        else bool(np.isclose(recorded, result["meta_ami_mean"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", help="Limit to these datasets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    datasets = index_local_datasets(DATA_DIR)
    manifests = sorted(ASSIGNMENTS_DIR.glob("*/*/manifest.json"))
    if args.datasets:
        manifests = [m for m in manifests if m.parent.parent.name in args.datasets]

    cache, records = {}, []
    for manifest_path in manifests:
        records.append(score_run(manifest_path, datasets, cache))

    rows = []
    for r in records:
        if r["skip_reason"] is not None:
            continue
        base = {k: v for k, v in r.items() if k not in ("meta_ami_by_covariate",)}
        for covariate, ami in r["meta_ami_by_covariate"].items():
            rows.append({**base, "covariate": covariate, "ami": ami})
    scored = [r for r in records if r["skip_reason"] is None]
    skipped = [r for r in records if r["skip_reason"] is not None]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).drop("skip_reason").write_csv(args.output)

    print(f"Scored {len(scored)} runs -> {args.output}")
    reasons = pl.DataFrame(skipped).group_by("skip_reason").len() if skipped else None
    if reasons is not None:
        print("Skipped:")
        for reason, count in reasons.iter_rows():
            print(f"  {count:4d}  {reason}")
    checked = [r for r in scored if r["matches_run_time_value"] is not None]
    if checked:
        agree = sum(r["matches_run_time_value"] for r in checked)
        print(f"Agreement with run-time values: {agree}/{len(checked)}")


if __name__ == "__main__":
    main()
