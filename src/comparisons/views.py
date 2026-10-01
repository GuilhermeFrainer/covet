"""Presentation tables for the RQ1 comparison edges.

Pure functions over the frames returned by
`analysis.compute_registered_edge_comparisons`, so the dashboard only renders.
"Reference" is the baseline an edge compares against and "Proposed" is the
variant. Deltas are improvement-oriented (positive favours the proposed model)
except for outcome metrics such as topic-metadata AMI, where they are
proposed minus reference.
"""

from __future__ import annotations

import polars as pl

STANDARD_CONDITION = "remove_rep_stopwords"

METRIC_LABELS = {
    "c_v": "C_v",
    "c_npmi": "NPMI",
    "u_mass": "UMass",
    "irbo": "IRBO",
    "topic_diversity": "Topic diversity",
    "duration_seconds": "Duration (s)",
    "outliers": "Outliers",
    "meta_ami_mean": "Metadata AMI",
}

_RUN_KEYS = ["Edge ID", "Dataset", "Seed", "Requested topics"]


def model_name(model_id: str, catalog: dict) -> str:
    """Short presentation name of a catalog model."""
    entry = catalog.get(model_id, {})
    return entry.get("short_label") or entry.get("label") or model_id


def edge_names(edges: list[dict], catalog: dict) -> pl.DataFrame:
    """One row per edge: order, chain, reference and proposed model names."""
    rows = [
        {
            "Edge ID": edge["id"],
            "Edge order": order,
            "Chain": edge["chain"],
            "Reference": model_name(edge["reference_model"], catalog),
            "Proposed": model_name(edge["variant_model"], catalog),
        }
        for order, edge in enumerate(edges)
    ]
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        (pl.col("Reference") + " → " + pl.col("Proposed")).alias("Comparison")
    )


def comparison_table(
    edges: list[dict],
    catalog: dict,
    edge_summary: pl.DataFrame,
    edge_datasets: pl.DataFrame,
    metric: str,
    condition: str = STANDARD_CONDITION,
) -> pl.DataFrame:
    """Cross-dataset summary of each edge for a single metric.

    Reference and proposed scores average the dataset-level scores of the
    datasets included in the test.
    """
    names = edge_names(edges, catalog)
    summary = (
        {
            row["Edge ID"]: row
            for row in edge_summary.filter(pl.col("Metric") == metric).to_dicts()
        }
        if not edge_summary.is_empty()
        else {}
    )
    datasets = (
        edge_datasets.filter(
            (pl.col("Metric") == metric) & (pl.col("Condition") == condition)
        )
        if not edge_datasets.is_empty()
        else pl.DataFrame()
    )
    rows = []
    for name in names.to_dicts():
        result = summary.get(name["Edge ID"], {})
        included = [d for d in (result.get("Included datasets") or "").split(", ") if d]
        scores = (
            datasets.filter(
                (pl.col("Edge ID") == name["Edge ID"])
                & pl.col("Dataset").is_in(included)
            )
            if included and not datasets.is_empty()
            else pl.DataFrame()
        )
        wins = result.get("Wins")
        rows.append(
            {
                "Chain": name["Chain"],
                "Reference": name["Reference"],
                "Proposed": name["Proposed"],
                "Reference score": scores["Baseline score"].mean()
                if not scores.is_empty()
                else None,
                "Proposed score": scores["Ablation score"].mean()
                if not scores.is_empty()
                else None,
                "Mean Δ": result.get("Mean dataset delta"),
                "Median Δ": result.get("Median dataset delta"),
                "W/T/L": (
                    f"{wins}/{result.get('Ties')}/{result.get('Losses')}"
                    if wins is not None
                    else "—"
                ),
                "Rank-biserial r": result.get("Rank-biserial effect"),
                "Exact p": result.get("Raw exact p"),
                "Holm p": result.get("Holm adjusted p"),
                "Datasets": result.get("Dataset coverage") or "0/5",
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def run_values(
    edge_runs: pl.DataFrame, metric: str, condition: str = STANDARD_CONDITION
) -> pl.DataFrame:
    """Matched run values for one metric, with realized topics and outliers.

    Columns: Edge ID, Dataset, Seed, Requested topics, Reference, Proposed, Δ,
    and the reference/proposed realized topic counts and outlier counts (null
    when not recorded).
    """
    if edge_runs.is_empty():
        return pl.DataFrame()
    runs = edge_runs.filter(pl.col("Condition") == condition)

    def values(name: str, ref: str, prop: str, delta: str | None = None):
        frame = runs.filter(pl.col("Metric") == name).select(
            *_RUN_KEYS,
            pl.col("Baseline score").alias(ref),
            pl.col("Ablation score").alias(prop),
            *([pl.col("Improvement delta").alias(delta)] if delta else []),
        )
        return frame.unique(subset=_RUN_KEYS, keep="first")

    result = values(metric, "Reference", "Proposed", "Δ")
    for name, ref, prop in (
        ("n_topics", "Reference topics", "Proposed topics"),
        ("outliers", "Reference outliers", "Proposed outliers"),
    ):
        result = result.join(values(name, ref, prop), on=_RUN_KEYS, how="left")
    return result.sort(_RUN_KEYS)


def dataset_table(
    edges: list[dict],
    catalog: dict,
    edge_runs: pl.DataFrame,
    metric: str,
    dataset: str,
    requested_topics=None,
    condition: str = STANDARD_CONDITION,
) -> pl.DataFrame:
    """Descriptive per-edge summary for one dataset (no cross-dataset test).

    Averages the matched runs in scope; W/T/L counts runs, not datasets.
    """
    runs = run_values(edge_runs, metric, condition)
    if runs.is_empty():
        return pl.DataFrame()
    runs = runs.filter(pl.col("Dataset") == dataset)
    if requested_topics:
        runs = runs.filter(pl.col("Requested topics").is_in(list(requested_topics)))
    stats = runs.group_by("Edge ID").agg(
        pl.col("Reference").mean().alias("Reference mean"),
        pl.col("Reference").std().alias("Reference SD"),
        pl.col("Proposed").mean().alias("Proposed mean"),
        pl.col("Proposed").std().alias("Proposed SD"),
        pl.col("Δ").mean().alias("Mean Δ"),
        pl.format(
            "{}/{}/{}",
            (pl.col("Δ") > 0).sum(),
            (pl.col("Δ") == 0).sum(),
            (pl.col("Δ") < 0).sum(),
        ).alias("Run W/T/L"),
        pl.col("Reference topics").mean().alias("Reference topics"),
        pl.col("Proposed topics").mean().alias("Proposed topics"),
        pl.col("Reference outliers").mean().alias("Reference outliers"),
        pl.col("Proposed outliers").mean().alias("Proposed outliers"),
        pl.len().alias("Runs"),
    )
    return (
        edge_names(edges, catalog)
        .join(stats, on="Edge ID", how="left")
        .sort("Edge order")
        .drop("Edge ID", "Edge order", "Comparison")
    )


def delta_by_topic_count(
    edges: list[dict],
    catalog: dict,
    edge_runs: pl.DataFrame,
    metric: str,
    dataset: str,
    condition: str = STANDARD_CONDITION,
) -> pl.DataFrame:
    """Mean and SD of the seed-level Δ per edge and requested topic count."""
    runs = run_values(edge_runs, metric, condition)
    if runs.is_empty():
        return pl.DataFrame()
    stats = (
        runs.filter(pl.col("Dataset") == dataset)
        .group_by("Edge ID", "Requested topics")
        .agg(
            pl.col("Δ").mean().alias("Mean Δ"),
            pl.col("Δ").std().fill_null(0.0).alias("SD Δ"),
            pl.len().alias("Seeds"),
        )
    )
    return (
        stats.join(edge_names(edges, catalog), on="Edge ID")
        .with_columns(
            (pl.col("Mean Δ") - pl.col("SD Δ")).alias("Low"),
            (pl.col("Mean Δ") + pl.col("SD Δ")).alias("High"),
        )
        .sort("Edge order", "Requested topics")
    )


def seed_table(
    edges: list[dict],
    catalog: dict,
    edge_runs: pl.DataFrame,
    metric: str,
    dataset: str,
    requested_topics: int,
    condition: str = STANDARD_CONDITION,
) -> pl.DataFrame:
    """Per-seed values of every edge for one dataset and topic count."""
    runs = run_values(edge_runs, metric, condition)
    if runs.is_empty():
        return pl.DataFrame()
    runs = runs.filter(
        (pl.col("Dataset") == dataset)
        & (pl.col("Requested topics") == requested_topics)
    )
    return (
        edge_names(edges, catalog)
        .join(runs, on="Edge ID")
        .sort("Edge order", "Seed")
        .drop("Edge ID", "Edge order", "Comparison", "Dataset", "Requested topics")
    )


def heatmap_cells(
    edges: list[dict],
    catalog: dict,
    edge_runs: pl.DataFrame,
    metric: str,
    condition: str = STANDARD_CONDITION,
) -> pl.DataFrame:
    """Seed-averaged Δ per edge × dataset × requested topic count."""
    runs = run_values(edge_runs, metric, condition)
    if runs.is_empty():
        return pl.DataFrame()
    cells = runs.group_by("Edge ID", "Dataset", "Requested topics").agg(
        pl.col("Δ").mean().alias("Mean Δ"),
        pl.col("Reference").mean().alias("Reference"),
        pl.col("Proposed").mean().alias("Proposed"),
        pl.col("Reference topics").mean().alias("Reference topics"),
        pl.col("Proposed topics").mean().alias("Proposed topics"),
        pl.len().alias("Seeds"),
    )
    return (
        cells.join(edge_names(edges, catalog), on="Edge ID")
        .with_columns(
            pl.format("{} · k={}", "Dataset", "Requested topics").alias("Cell")
        )
        .sort("Edge order", "Dataset", "Requested topics")
    )
