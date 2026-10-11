"""Matched ablation-versus-baseline summaries for the dashboard."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

import polars as pl
import yaml

# REQUESTED_TOPICS is re-exported for modules that import it from here.
from src.topic_grids import DEFAULT_TOPIC_GRID, REQUESTED_TOPICS, topic_grid  # noqa: F401

SEEDS = (36201624, 62613654, 57116123)
BENCHMARK_DATASETS = ("anes", "fed", "gadarian", "trump", "yelp")
QUALITY_METRICS = ("c_v", "c_npmi", "u_mass", "irbo", "topic_diversity")
# Topic–metadata AMI is tested as a change (variant minus reference), not as an
# improvement: models using covariates as input can raise it by construction.
INFERENTIAL_METRICS = (
    *QUALITY_METRICS, "duration_seconds", "outliers", "meta_ami_mean"
)
METRIC_DIRECTIONS = {
    **{name: "maximize" for name in QUALITY_METRICS},
    "duration_seconds": "minimize",
    "outliers": "minimize",
    "meta_ami_mean": "outcome",
    "n_topics": "outcome",
}
RQ1_EDGE_REGISTRY = Path(__file__).resolve().parents[2] / "config" / "rq1_presentation_edges.yaml"
# Corpora that can stand in for the benchmark's Trump dataset. They are the
# same corpus, so exactly one may count in a cross-dataset comparison.
TRUMP_VARIANTS = {"trump": "Full Trump", "trump_s25000": "Trump 25k sample"}


def use_trump_variant(df: pl.DataFrame, variant: str = "trump") -> pl.DataFrame:
    """Makes `variant` the benchmark's Trump dataset and drops the others.

    Rows of `variant` are relabelled `trump` in `dataset_label`, so every
    comparison treats them as the fifth benchmark dataset; rows of the other
    Trump variants are removed.
    """
    if variant not in TRUMP_VARIANTS:
        raise ValueError(f"Unknown Trump variant: {variant}")
    others = [name for name in TRUMP_VARIANTS if name != variant]
    return df.filter(~pl.col("dataset_label").is_in(others)).with_columns(
        pl.when(pl.col("dataset_label") == variant)
        .then(pl.lit("trump"))
        .otherwise(pl.col("dataset_label"))
        .alias("dataset_label")
    )


def load_rq1_edges(path=RQ1_EDGE_REGISTRY):
    """Load and validate the explicit RQ1 display/comparison edges."""
    with Path(path).open(encoding="utf-8") as stream:
        registry = yaml.safe_load(stream)
    if not isinstance(registry, dict) or set(registry) != {"schema_version", "edges"}:
        raise ValueError("Invalid RQ1 edge registry structure")
    if registry["schema_version"] != 1 or not isinstance(registry["edges"], list):
        raise ValueError("Unsupported RQ1 edge registry schema")
    required = {
        "id", "reference_model", "variant_model", "chain", "label",
        "intended_change", "classification", "main_figure",
    }
    ids = set()
    for edge in registry["edges"]:
        if not isinstance(edge, dict) or set(edge) != required:
            raise ValueError("Each RQ1 edge must have exactly the registered fields")
        if edge["id"] in ids:
            raise ValueError(f"Duplicate RQ1 edge id: {edge['id']}")
        ids.add(edge["id"])
        if edge["reference_model"] == edge["variant_model"]:
            raise ValueError(f"Self-comparison in RQ1 edge: {edge['id']}")
        if edge["classification"] not in {"strict_candidate", "architectural"}:
            raise ValueError(f"Invalid comparison classification: {edge['id']}")
        if not isinstance(edge["main_figure"], bool):
            raise ValueError(f"main_figure must be boolean: {edge['id']}")
    return registry["edges"]


def _as_int(value):
    try:
        number = float(value)
        return int(number) if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _as_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _exact_pratt_signed_rank(differences):
    """Return a two-sided exhaustive signed-rank p-value and diagnostics."""
    if not differences or any(not math.isfinite(value) for value in differences):
        return None, None, None, None, None, None, None
    indexed = sorted((abs(value), index) for index, value in enumerate(differences))
    ranks = [0.0] * len(differences)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2
        for _, index in indexed[i:j]:
            ranks[index] = average_rank
        i = j
    nonzero = [index for index, value in enumerate(differences) if value != 0]
    observed = sum(ranks[index] if differences[index] > 0 else -ranks[index] for index in nonzero)
    positive_rank_sum = sum(ranks[index] for index in nonzero if differences[index] > 0)
    negative_rank_sum = sum(ranks[index] for index in nonzero if differences[index] < 0)
    rank_total = positive_rank_sum + negative_rank_sum
    rank_biserial = (positive_rank_sum - negative_rank_sum) / rank_total if rank_total else None
    tied_rank_groups = sum(1 for _, group in _group_tied_ranks(indexed) if group > 1)
    if not nonzero:
        return 1.0, observed, 0, len(differences), tied_rank_groups, 1, None
    extreme = 0
    total = 1 << len(nonzero)
    tolerance = 1e-12
    for signs in product((-1, 1), repeat=len(nonzero)):
        statistic = sum(sign * ranks[index] for sign, index in zip(signs, nonzero))
        if abs(statistic) + tolerance >= abs(observed):
            extreme += 1
    return extreme / total, observed, len(nonzero), len(differences) - len(nonzero), tied_rank_groups, total, rank_biserial


def _group_tied_ranks(indexed):
    """Yield tied groups from a sorted (magnitude, observation) list."""
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        yield indexed[i][0], j - i
        i = j


def _holm(p_values):
    """Holm step-down adjusted p-values, preserving the input order."""
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _deduplicate_runs(df):
    """Prefer merged rows and collapse raw/merged copies by run_uid or run cell."""
    rows = df.to_dicts()
    chosen = {}
    ambiguous = set()
    identities = {}
    for row in rows:
        if row.get("role") != "ablation" and row.get("role") != "baseline":
            continue
        model = row.get("catalog_id")
        dataset = row.get("dataset_label") or row.get("dataset_name")
        condition = row.get("condition") or "unknown"
        seed = _as_int(row.get("random_state", row.get("seed")))
        if seed is None:
            seed = _as_int(row.get("seed"))
        topics = _as_int(row.get("nr_topics"))
        if topics is None:
            topics = _as_int(row.get("n_clusters"))
        if not model or not dataset or seed is None or topics is None:
            continue
        key = (model, dataset, condition, seed, topics)
        source = str(row.get("source_file") or "")
        rank = (0 if "_merged" in source else 1, str(row.get("file_timestamp") or ""))
        identity = row.get("run_uid") or (
            row.get("experiment_id"), row.get("model_name"),
            row.get("file_timestamp"), row.get("resolved_config_hash"),
        )
        if key in identities and identity != identities[key]:
            ambiguous.add(key)
        else:
            identities[key] = identity
        previous = chosen.get(key)
        if previous is None or rank < previous[0]:
            chosen[key] = (rank, row)
    return {key: value[1] for key, value in chosen.items()}, ambiguous


def compute_ablation_comparisons(
    df: pl.DataFrame,
    catalog: dict,
    summary_model_ids=None,
    requested_topics=None,
    grid: str = DEFAULT_TOPIC_GRID,
):
    """Build dataset-level deltas and cross-dataset tests for registered pairs.

    The dashboard intentionally presents all preprocessing conditions separately.
    Inferential tests use the documented standard condition and the seed x
    requested-topic grid of each dataset under `grid` (see `topic_grids`), or
    only its counts in `requested_topics` (an exploratory slice). A dataset
    whose grid has none of the sliced counts is left out.
    """
    cells, ambiguous = _deduplicate_runs(df)
    dataset_rows = []
    run_rows = []
    summary_rows = []
    ablations = [(name, entry) for name, entry in catalog.items() if entry["role"] == "ablation"]
    inference_ablations = [
        (name, entry)
        for name, entry in ablations
        if summary_model_ids is None or name in summary_model_ids
    ]
    inference_ids = {name for name, _ in inference_ablations}

    family_records = {metric: [] for metric in INFERENTIAL_METRICS}
    standard_seen = set()
    for ablation_id, entry in ablations:
        baseline_id = entry["baseline_id"]
        datasets = BENCHMARK_DATASETS
        conditions = sorted({key[2] for key in cells if key[0] in {ablation_id, baseline_id}})
        for condition in conditions:
            if condition in {"standard", "remove_rep_stopwords"} and ablation_id in inference_ids:
                standard_seen.add(ablation_id)
            tested = (
                condition in {"standard", "remove_rep_stopwords"}
                and ablation_id in inference_ids
            )
            pair_dataset_deltas = {metric: [] for metric in INFERENTIAL_METRICS}
            for dataset in datasets:
                counts = topic_grid(dataset, grid)
                if requested_topics:
                    counts = tuple(c for c in counts if c in requested_topics)
                if not counts:
                    if tested:
                        for metric in INFERENTIAL_METRICS:
                            pair_dataset_deltas[metric].append(None)
                    continue
                expected = {(seed, count) for seed in SEEDS for count in counts}
                metric_diffs = {metric: [] for metric in (*INFERENTIAL_METRICS, "n_topics")}
                baseline_scores = {metric: [] for metric in metric_diffs}
                variant_scores = {metric: [] for metric in metric_diffs}
                matched = 0
                failures = []
                for seed, count in sorted(expected):
                    left_key = (baseline_id, dataset, condition, seed, count)
                    right_key = (ablation_id, dataset, condition, seed, count)
                    left = cells.get(left_key)
                    right = cells.get(right_key)
                    if left_key in ambiguous or right_key in ambiguous:
                        failures.append("ambiguous duplicate run cell")
                        continue
                    if left is None or right is None:
                        continue
                    if left.get("run_status") not in (None, "success") or right.get("run_status") not in (None, "success"):
                        failures.append("failed run")
                        continue
                    left_n = _as_int(left.get("n_observations"))
                    right_n = _as_int(right.get("n_observations"))
                    if left_n is not None and right_n is not None and left_n != right_n:
                        failures.append("sample-size mismatch")
                        continue
                    matched += 1
                    for metric in metric_diffs:
                        baseline_value = _as_float(left.get(metric))
                        variant_value = _as_float(right.get(metric))
                        if baseline_value is None or variant_value is None:
                            continue
                        baseline_scores[metric].append(baseline_value)
                        variant_scores[metric].append(variant_value)
                        direction = METRIC_DIRECTIONS[metric]
                        if direction == "maximize":
                            delta = variant_value - baseline_value
                        elif direction == "minimize":
                            delta = baseline_value - variant_value
                        else:
                            delta = variant_value - baseline_value
                        metric_diffs[metric].append(delta)
                        run_rows.append({
                            "Ablation": entry["label"], "Model ID": ablation_id,
                            "Reference baseline": catalog[baseline_id]["label"],
                            "Metric": metric, "Dataset": dataset,
                            "Condition": condition, "Seed": seed,
                            "Requested topics": count,
                            "Baseline score": baseline_value,
                            "Ablation score": variant_value,
                            "Improvement delta": delta,
                        })
                expected_count = len(expected)
                status = "complete" if matched == expected_count else "incomplete"
                if failures:
                    status += "; " + ", ".join(sorted(set(failures)))
                for metric, deltas in metric_diffs.items():
                    metric_status = status if len(deltas) == expected_count else f"incomplete metric: {len(deltas)}/{expected_count} valid values"
                    dataset_rows.append({
                        "Ablation": entry["label"], "Model ID": ablation_id,
                        "Reference baseline": catalog[baseline_id]["label"],
                        "Baseline ID": baseline_id, "Metric": metric,
                        "Direction": METRIC_DIRECTIONS[metric], "Dataset": dataset,
                        "Condition": condition,
                        "Baseline score": sum(baseline_scores[metric]) / len(baseline_scores[metric]) if baseline_scores[metric] else None,
                        "Ablation score": sum(variant_scores[metric]) / len(variant_scores[metric]) if variant_scores[metric] else None,
                        "Improvement delta": sum(deltas) / len(deltas) if deltas else None,
                        "Matched cells": len(deltas), "Expected cells": expected_count,
                        "Coverage": f"{len(deltas)}/{expected_count}", "Status": metric_status,
                        "Provenance": "sample/config parity not fully verifiable from merged result rows",
                    })
                if tested:
                    for metric in INFERENTIAL_METRICS:
                        if matched == expected_count and len(metric_diffs[metric]) == expected_count:
                            pair_dataset_deltas[metric].append(metric_diffs[metric])
                        else:
                            pair_dataset_deltas[metric].append(None)

            if tested:
                for metric in INFERENTIAL_METRICS:
                    dataset_deltas = pair_dataset_deltas[metric]
                    # Keep dataset-level deltas in rows above; inference is cross-dataset only.
                    flattened = [sum(values) / len(values) for values in dataset_deltas if values is not None]
                    complete_datasets = [
                        dataset for dataset, values in zip(datasets, dataset_deltas)
                        if values is not None
                    ]
                    dataset_coverage = f"{len(complete_datasets)}/{len(BENCHMARK_DATASETS)}"
                    complete = len(complete_datasets) >= 2
                    if complete:
                        (
                            p_value, statistic, nonzero, zero_count, tied_rank_groups,
                            sign_flip_assignments, rank_biserial,
                        ) = _exact_pratt_signed_rank(flattened)
                        ordered = sorted(flattened)
                        summary = {
                            "Ablation": entry["label"], "Model ID": ablation_id,
                            "Reference baseline": catalog[baseline_id]["label"],
                            "Baseline ID": baseline_id, "Metric": metric,
                            "Datasets": len(flattened),
                            "Dataset coverage": dataset_coverage,
                            "Included datasets": ", ".join(complete_datasets),
                            "Missing/incomplete datasets": ", ".join(
                                dataset for dataset in BENCHMARK_DATASETS
                                if dataset not in complete_datasets
                            ),
                            "Mean dataset delta": sum(flattened) / len(flattened),
                            "Median dataset delta": ordered[len(ordered)//2] if len(ordered) % 2 else (ordered[len(ordered)//2-1] + ordered[len(ordered)//2]) / 2,
                            "Wins": sum(value > 0 for value in flattened),
                            "Ties": sum(value == 0 for value in flattened),
                            "Losses": sum(value < 0 for value in flattened),
                            "Signed-rank statistic": statistic,
                            "Nonzero datasets": nonzero, "Zero datasets": zero_count,
                            "Tied-rank groups": tied_rank_groups,
                            "Sign-flip assignments": sign_flip_assignments,
                            "Rank-biserial effect": rank_biserial,
                            "Raw exact p": p_value,
                            "Holm adjusted p": None,
                            "Inference status": (
                                f"tested across {len(complete_datasets)} complete datasets"
                                if len(complete_datasets) == len(BENCHMARK_DATASETS)
                                else f"partial benchmark: tested across {len(complete_datasets)} complete datasets"
                            ),
                        }
                        family_records[metric].append(summary)
                    else:
                        summary_rows.append({
                            "Ablation": entry["label"], "Model ID": ablation_id,
                            "Reference baseline": catalog[baseline_id]["label"],
                            "Baseline ID": baseline_id, "Metric": metric,
                            "Datasets": len(flattened),
                            "Dataset coverage": dataset_coverage,
                            "Included datasets": ", ".join(complete_datasets),
                            "Missing/incomplete datasets": ", ".join(
                                dataset for dataset in BENCHMARK_DATASETS
                                if dataset not in complete_datasets
                            ),
                            "Mean dataset delta": sum(flattened) / len(flattened) if flattened else None,
                            "Median dataset delta": None, "Wins": None, "Ties": None,
                            "Losses": None, "Signed-rank statistic": None,
                            "Nonzero datasets": None, "Zero datasets": None,
                            "Tied-rank groups": None, "Sign-flip assignments": None,
                            "Rank-biserial effect": None, "Raw exact p": None,
                            "Holm adjusted p": None,
                            "Inference status": (
                                "descriptive only: fewer than two datasets have a "
                                "complete seed x topic-count grid"
                            ),
                        })

    # Keep catalog pairs with no standard-condition rows visible in the summary.
    for ablation_id, entry in inference_ablations:
        if ablation_id in standard_seen:
            continue
        for metric in INFERENTIAL_METRICS:
            summary_rows.append({
                "Ablation": entry["label"], "Model ID": ablation_id,
                "Reference baseline": catalog[entry["baseline_id"]]["label"],
                "Baseline ID": entry["baseline_id"], "Metric": metric,
                "Datasets": 0, "Mean dataset delta": None,
                "Median dataset delta": None, "Wins": None, "Ties": None,
                "Losses": None, "Signed-rank statistic": None,
                "Nonzero datasets": None, "Zero datasets": None,
                "Tied-rank groups": None, "Sign-flip assignments": None,
                "Rank-biserial effect": None, "Raw exact p": None,
                "Holm adjusted p": None,
                "Inference status": "blocked: no standard-condition matched results found",
            })

    # Correct over comparisons that can be tested; incomplete comparisons remain
    # visible as blocked rows and do not prevent analysis of available datasets.
    for metric in INFERENTIAL_METRICS:
        records = family_records[metric]
        if records:
            adjusted = _holm([record["Raw exact p"] for record in records])
            for record, adjusted_p in zip(records, adjusted):
                record["Holm adjusted p"] = adjusted_p
                record["Inference status"] += (
                    f"; Holm adjusted across {len(records)} estimable comparisons for this metric"
                )
                summary_rows.append(record)
    dataset_frame = (
        pl.DataFrame(dataset_rows, infer_schema_length=None)
        if dataset_rows
        else pl.DataFrame()
    )
    summary_frame = (
        pl.DataFrame(summary_rows, infer_schema_length=None)
        if summary_rows
        else pl.DataFrame()
    )
    run_frame = (
        pl.DataFrame(run_rows, infer_schema_length=None)
        if run_rows
        else pl.DataFrame()
    )
    return dataset_frame, summary_frame, run_frame


def compute_registered_edge_comparisons(
    df: pl.DataFrame,
    catalog: dict,
    edges: list[dict],
    requested_topics=None,
    grid: str = DEFAULT_TOPIC_GRID,
):
    """Compute results for explicit directed edges, including adjacent variants.

    `requested_topics` and `grid` select the topic counts as in
    `compute_ablation_comparisons`.
    """
    dataset_rows = []
    summary_rows = []
    run_rows = []
    for order, edge in enumerate(edges):
        reference_id = edge["reference_model"]
        variant_id = edge["variant_model"]
        if reference_id not in catalog or variant_id not in catalog:
            raise ValueError(f"Unknown model in RQ1 edge: {edge['id']}")
        if catalog[variant_id]["role"] != "ablation":
            raise ValueError(f"RQ1 edge variant is not a catalog ablation: {edge['id']}")

        edge_catalog = dict(catalog)
        edge_catalog[variant_id] = dict(catalog[variant_id], baseline_id=reference_id)
        dataset_frame, summary_frame, run_frame = compute_ablation_comparisons(
            df,
            edge_catalog,
            summary_model_ids={variant_id},
            requested_topics=requested_topics,
            grid=grid,
        )
        if not dataset_frame.is_empty():
            dataset_rows.extend(
                row for row in dataset_frame.filter(
                    (pl.col("Model ID") == variant_id)
                    & (pl.col("Baseline ID") == reference_id)
                ).to_dicts()
            )
        if not summary_frame.is_empty():
            summary_rows.extend(
                row for row in summary_frame.filter(
                    pl.col("Model ID") == variant_id
                ).to_dicts()
            )
        if not run_frame.is_empty():
            run_rows.extend(
                row for row in run_frame.filter(
                    pl.col("Model ID") == variant_id
                ).to_dicts()
            )

        metadata = {
            "Edge order": order,
            "Edge ID": edge["id"],
            "Chain": edge["chain"],
            "Comparison": edge["label"],
            "Intended change": edge["intended_change"],
            "Comparison type": edge["classification"],
        }
        for rows in (dataset_rows, summary_rows, run_rows):
            # Only annotate rows belonging to the edge just computed.
            for row in reversed(rows):
                if row.get("Model ID") != variant_id:
                    break
                row.update(metadata)

    # Apply Holm within metric over the registered RQ1 edges that can be tested.
    tested_by_metric = {}
    for row in summary_rows:
        if row.get("Raw exact p") is not None:
            tested_by_metric.setdefault(row["Metric"], []).append(row)
    for metric_rows in tested_by_metric.values():
        adjusted = _holm([row["Raw exact p"] for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["Holm adjusted p"] = adjusted_p
            row["Inference status"] = row["Inference status"].split(
                "; Holm adjusted across ", maxsplit=1
            )[0]
            row["Inference status"] += (
                f"; exploratory Holm across {len(metric_rows)} estimable RQ1 edges"
            )

    def frame(rows):
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()

    return frame(dataset_rows), frame(summary_rows), frame(run_rows)
