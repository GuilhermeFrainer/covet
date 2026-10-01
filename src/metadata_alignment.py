"""Topic–metadata alignment: adjusted mutual information per raw covariate.

Alignment is a descriptive outcome, not a quality score to maximize: a model
whose topics simply reproduce the metadata groups scores 1.0.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_mutual_info_score

N_BINS = 5
METHOD = (
    "adjusted mutual information (arithmetic normalization) between hard topic "
    f"labels and each raw covariate; numerics with more than {N_BINS} distinct "
    f"values are split into {N_BINS} quantile bins over the run's documents, "
    "other covariates are used as categories; nulls form their own category; "
    "noise (-1) counts as its own topic"
)


def discretize(values: pl.Series, n_bins: int = N_BINS) -> np.ndarray:
    """Turns one raw covariate into category labels for mutual information."""
    if values.dtype.is_temporal():
        values = values.to_physical()
    if values.dtype.is_numeric() and values.n_unique() > n_bins:
        values = values.cast(pl.Float64).qcut(
            n_bins, labels=[f"q{i}" for i in range(n_bins)], allow_duplicates=True
        )
    return values.cast(pl.String).fill_null("NA").to_numpy()


def metadata_alignment(topics, covariates: pl.DataFrame) -> dict:
    """Computes AMI between topic labels and each covariate column.

    Args:
        topics: Final topic label per document, aligned with covariates rows.
        covariates: Raw covariate values, one column per covariate.

    Returns:
        A dict with ``meta_ami_mean`` (mean over covariates) and
        ``meta_ami_by_covariate`` (covariate name -> AMI).
    """
    topics = np.asarray(topics)
    if len(topics) != covariates.height:
        raise ValueError(
            f"{len(topics)} topic labels for {covariates.height} covariate rows"
        )
    if covariates.width == 0:
        return {"meta_ami_mean": float("nan"), "meta_ami_by_covariate": {}}
    by_covariate = {
        name: float(adjusted_mutual_info_score(discretize(covariates[name]), topics))
        for name in covariates.columns
    }
    return {
        "meta_ami_mean": float(np.mean(list(by_covariate.values()))),
        "meta_ami_by_covariate": by_covariate,
    }


def load_covariate_alignment(manifests_dir, backfill_path) -> pl.DataFrame:
    """Collects per-covariate AMI for every scored run, one row per covariate.

    Run-time values recorded in run manifests take precedence; the backfill
    sidecar supplies runs trained before the metric existed.

    Args:
        manifests_dir: Directory holding ``<dataset>/<run>/manifest.json``.
        backfill_path: Long-format CSV written by
            ``scripts/analysis/backfill_metadata_alignment.py``.

    Returns:
        Columns ``run_uid``, ``covariate``, ``ami``, ``meta_ami_mean`` and
        ``ami_source`` (``run_time`` or ``backfill``).
    """
    schema = {
        "run_uid": pl.String,
        "covariate": pl.String,
        "ami": pl.Float64,
        "meta_ami_mean": pl.Float64,
        "ami_source": pl.String,
    }
    rows = []
    for path in sorted(Path(manifests_dir).glob("*/*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        alignment = manifest.get("metadata_alignment")
        if not manifest.get("run_uid") or not alignment:
            continue
        for covariate, ami in alignment.get("meta_ami_by_covariate", {}).items():
            rows.append(
                {
                    "run_uid": manifest["run_uid"],
                    "covariate": covariate,
                    "ami": ami,
                    "meta_ami_mean": alignment.get("meta_ami_mean"),
                    "ami_source": "run_time",
                }
            )
    run_time = pl.DataFrame(rows, schema=schema)
    backfill_path = Path(backfill_path)
    if not backfill_path.exists():
        return run_time
    backfill = (
        pl.read_csv(backfill_path, infer_schema_length=None)
        .select(
            pl.col("run_uid").cast(pl.String),
            pl.col("covariate").cast(pl.String),
            pl.col("ami").cast(pl.Float64),
            pl.col("meta_ami_mean").cast(pl.Float64),
            pl.lit("backfill").alias("ami_source"),
        )
        .filter(~pl.col("run_uid").is_in(run_time["run_uid"].unique().implode()))
    )
    return pl.concat([run_time, backfill])


def fill_run_alignment(results: pl.DataFrame, alignment: pl.DataFrame) -> pl.DataFrame:
    """Fills missing ``meta_ami_mean`` result values by ``run_uid``.

    Values already in the results are kept; rows without a ``run_uid`` are left
    unchanged.
    """
    if "run_uid" not in results.columns or alignment.is_empty():
        return results
    by_run = alignment.group_by("run_uid").agg(
        pl.col("meta_ami_mean").first().alias("_meta_ami_filled")
    )
    current = (
        pl.col("meta_ami_mean").cast(pl.Float64, strict=False)
        if "meta_ami_mean" in results.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    return (
        results.with_columns(pl.col("run_uid").cast(pl.String))
        .join(by_run, on="run_uid", how="left", maintain_order="left")
        .with_columns(
            pl.coalesce(current, pl.col("_meta_ami_filled")).alias("meta_ami_mean")
        )
        .drop("_meta_ami_filled")
    )
