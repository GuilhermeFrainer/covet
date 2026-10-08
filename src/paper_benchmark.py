"""Free-for-all benchmark table (T2): mean scores and Friedman ranks.

Each model's score on a dataset is the mean over its runs (3 seeds x 5
requested topic counts; STM has one run per topic count). Models are ranked
within each dataset (1 = best, ties averaged), and the average ranks are
compared with the Iman-Davenport correction of the Friedman test, with
Kendall's W as the effect size. Only models with complete runs on every
benchmark dataset are ranked, so the number of datasets stays fixed; the
others are reported with their means and marked as unranked. Pairwise
differences are not tested here: they come from the planned comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.stats import f as f_distribution
from scipy.stats import rankdata

from src.comparisons.analysis import (
    BENCHMARK_DATASETS,
    METRIC_DIRECTIONS,
    REQUESTED_TOPICS,
)
from src.comparisons.views import STANDARD_CONDITION
from src.paper_outputs import (
    PRELIMINARY_NOTE,
    _latex_escape,
    _latex_metric,
    paper_label,
)

BENCHMARK_CONFIG = (
    Path(__file__).resolve().parents[1] / "config" / "paper_benchmark.yaml"
)
# Columns holding the requested topic count, by model family: HDBSCAN models
# store it in `nr_topics`, spectral and K-Means models in `n_clusters`, and
# TriTopic in `n_topics` (where it overwrote the realized count).
REQUESTED_TOPIC_COLUMNS = ("requested_topics", "nr_topics", "n_clusters", "n_topics")
# Models whose `n_topics` holds the requested, not the realized, topic count.
REQUESTED_COUNT_MODELS = {"tritopic", "fast_tritopic"}


def load_benchmark_models(path: Path = BENCHMARK_CONFIG) -> list[dict]:
    """Benchmark models in table order, each with its `sources`."""
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    models = []
    for entry in config["models"]:
        models.append(
            {"id": entry["id"], "sources": entry.get("sources") or [entry["id"]]}
        )
    return models


def _requested_topics(results: pl.DataFrame) -> pl.Expr:
    columns = [c for c in REQUESTED_TOPIC_COLUMNS if c in results.columns]
    return pl.coalesce(
        [pl.col(c).cast(pl.Float64, strict=False).cast(pl.Int64) for c in columns]
    ).alias("requested")


def dataset_scores(results: pl.DataFrame, models: list[dict], metrics) -> pl.DataFrame:
    """Per model and dataset: run count, completeness and mean scores.

    A model is complete on a dataset when every requested topic count is
    present for each of its seeds and every metric is scored. For each
    dataset, the first complete source in the model's `sources` is used.
    """
    standard = results.filter(pl.col("condition") == STANDARD_CONDITION).with_columns(
        _requested_topics(results),
        pl.col("random_state").cast(pl.Int64, strict=False).alias("seed"),
        pl.col("source_file").str.contains("_merged").alias("merged"),
    )
    rows = []
    for model in models:
        for dataset in BENCHMARK_DATASETS:
            chosen = None
            for source in model["sources"]:
                runs = (
                    standard.filter(
                        (pl.col("catalog_id") == source)
                        & (pl.col("dataset_label") == dataset)
                        & pl.col("requested").is_in(list(REQUESTED_TOPICS))
                    )
                    .sort("merged", descending=True)
                    .unique(subset=["seed", "requested"], keep="first")
                )
                if runs.is_empty():
                    continue
                candidate = _summarize(runs, model["id"], source, dataset, metrics)
                if chosen is None or (candidate["Complete"] and not chosen["Complete"]):
                    chosen = candidate
                if chosen["Complete"]:
                    break
            if chosen is not None:
                rows.append(chosen)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def _summarize(runs: pl.DataFrame, model_id, source, dataset, metrics) -> dict:
    seeds = runs["seed"].n_unique()
    scored = all(
        metric in runs.columns
        and runs[metric].cast(pl.Float64, strict=False).null_count() == 0
        for metric in metrics
    )
    row = {
        "Model ID": model_id,
        "Source": source,
        "Dataset": dataset,
        "Runs": runs.height,
        "Complete": scored and runs.height == seeds * len(REQUESTED_TOPICS),
    }
    for metric in metrics:
        row[metric] = (
            runs[metric].cast(pl.Float64, strict=False).mean()
            if metric in runs.columns
            else None
        )
    topics = runs["n_topics"].cast(pl.Float64, strict=False).mean()
    row["n_topics"] = None if source in REQUESTED_COUNT_MODELS else topics
    return row


def friedman(scores: np.ndarray, direction: str = "maximize") -> dict:
    """Average ranks and Iman-Davenport test for a datasets x models array.

    Rank 1 is the best model on a dataset; ties receive average ranks.
    """
    n_datasets, n_models = scores.shape
    oriented = -scores if direction == "maximize" else scores
    ranks = np.vstack([rankdata(row) for row in oriented])
    mean_ranks = ranks.mean(axis=0)
    result = {"Mean ranks": mean_ranks, "N": n_datasets, "k": n_models}
    if n_datasets < 2 or n_models < 2:
        return {**result, "Chi2": None, "F": None, "p": None, "W": None}
    chi2 = (
        12
        * n_datasets
        / (n_models * (n_models + 1))
        * float(np.sum(mean_ranks**2) - n_models * (n_models + 1) ** 2 / 4)
    )
    w = chi2 / (n_datasets * (n_models - 1))
    denominator = n_datasets * (n_models - 1) - chi2
    if denominator <= 0:  # every dataset ranks the models identically
        f_stat, p_value = float("inf"), 0.0
    else:
        f_stat = (n_datasets - 1) * chi2 / denominator
        p_value = float(
            f_distribution.sf(f_stat, n_models - 1, (n_models - 1) * (n_datasets - 1))
        )
    return {**result, "Chi2": chi2, "F": f_stat, "p": p_value, "W": w}


def benchmark_table(
    scores: pl.DataFrame, models: list[dict], metrics
) -> tuple[pl.DataFrame, dict]:
    """One row per model, and the Friedman statistics per metric.

    Row columns: complete datasets, mean score and average rank per metric
    (rank only for models complete on every benchmark dataset), and the mean
    realized topic count.
    """
    complete = scores.filter(pl.col("Complete")) if not scores.is_empty() else scores
    counts = (
        {
            r["Model ID"]: r["len"]
            for r in complete.group_by("Model ID").len().to_dicts()
        }
        if not complete.is_empty()
        else {}
    )
    ranked = [
        m["id"] for m in models if counts.get(m["id"], 0) == len(BENCHMARK_DATASETS)
    ]
    stats, ranks = {}, {}
    for metric in metrics:
        if len(ranked) >= 2:
            wide = (
                complete.filter(pl.col("Model ID").is_in(ranked))
                .pivot(on="Model ID", index="Dataset", values=metric)
                .sort("Dataset")
                .select(ranked)
            )
            stats[metric] = friedman(wide.to_numpy(), METRIC_DIRECTIONS[metric])
            ranks[metric] = dict(zip(ranked, stats[metric]["Mean ranks"]))
        else:
            stats[metric] = None
            ranks[metric] = {}
    rows = []
    for model in models:
        model_id = model["id"]
        own = complete.filter(pl.col("Model ID") == model_id) if counts else complete
        row = {
            "Model ID": model_id,
            "Datasets": counts.get(model_id, 0),
            "Ranked": model_id in ranked,
            "Sources": ", ".join(sorted(set(own["Source"].to_list())))
            if counts
            else "",
        }
        for metric in metrics:
            row[metric] = own[metric].mean() if own.height else None
            row[f"{metric} rank"] = ranks[metric].get(model_id)
        topics = own["n_topics"].drop_nulls() if own.height else []
        row["n_topics"] = topics.mean() if len(topics) else None
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None), stats


def _score(value) -> str:
    """Three decimals, without the leading zero below 1 (".509", "-.033")."""
    if value is None:
        return "--"
    text = f"{value:.3f}"
    if abs(value) < 1:
        text = text.replace("0.", ".", 1)
    return f"${text}$"


def benchmark_table_latex(
    table: pl.DataFrame,
    stats: dict,
    catalog: dict,
    metrics,
    preliminary: bool = False,
    label: str = "tab:benchmark",
    note: str = "",
) -> str:
    """Booktabs table: mean score (average rank) per metric, then test rows.

    Models without complete runs on any dataset are left out; models complete
    on only some datasets are marked and named in the caption.
    """
    header = " & ".join(
        [r"\textbf{Model}"] + [_latex_metric(m) for m in metrics] + ["$K$"]
    )
    best = {
        metric: min(
            (
                r[f"{metric} rank"]
                for r in table.to_dicts()
                if r[f"{metric} rank"] is not None
            ),
            default=None,
        )
        for metric in metrics
    }
    body, unranked, requested_only = [], [], False
    previous_group = None
    for row in table.to_dicts():
        if not row["Datasets"]:
            continue
        group = catalog.get(row["Model ID"], {}).get("role") == "ablation"
        if previous_group is not None and group != previous_group:
            body.append(r"\midrule")
        previous_group = group
        name = paper_label(row["Model ID"], catalog)
        if not row["Ranked"]:
            unranked.append(f"{name}: {row['Datasets']}")
            name += r"$^\dagger$"
        cells = [name]
        for metric in metrics:
            cell = _score(row[metric])
            rank = row[f"{metric} rank"]
            if rank is not None:
                rank_text = f"{rank:.1f}"
                if best[metric] is not None and abs(rank - best[metric]) < 1e-9:
                    rank_text = rf"\textbf{{{rank_text}}}"
                cell += rf"\,{{\scriptsize({rank_text})}}"
            cells.append(cell)
        if row["n_topics"] is None:
            cells.append("--")
            requested_only = requested_only or any(
                s in REQUESTED_COUNT_MODELS for s in row["Sources"].split(", ")
            )
        else:
            cells.append("--" if row["n_topics"] is None else f"{row['n_topics']:.0f}")
        body.append(" & ".join(cells) + r" \\")
    test_rows = [
        ("$p$", lambda s: f"{s['p']:.3f}" if s["p"] >= 0.001 else "$<0.001$"),
        ("$W$", lambda s: f"{s['W']:.2f}"),
    ]
    tests = []
    for name, render in test_rows:
        cells = [name]
        for metric in metrics:
            s = stats.get(metric)
            cells.append("--" if not s or s["p"] is None else render(s))
        cells.append("")
        tests.append(" & ".join(cells) + r" \\")
    tested = next((s for s in stats.values() if s), None)
    shown = {r["Model ID"] for r in table.to_dicts() if r["Datasets"]}
    caption = (
        r"Free-for-all comparison. Mean score across datasets (each dataset "
        r"averages 3 seeds $\times$ 5 requested topic counts"
        + ("; STM: one run per topic count" if "stm" in shown else "")
        + r") and, in parentheses, average rank (1 = best, best in bold)"
    )
    if tested:
        caption += (
            f" among the {tested['k']} models with complete runs on all "
            f"{tested['N']} datasets"
        )
    caption += (
        r". $p$: Iman--Davenport test of "
        r"equal average ranks; $W$: Kendall's coefficient of concordance. "
        r"Pairwise differences are not tested here. $K$: mean realized number "
        r"of topics."
    )
    if unranked:
        caption += (
            r" $^\dagger$Not ranked: complete runs on fewer datasets ("
            + ", ".join(unranked)
            + "); mean over those only."
        )
    if requested_only:
        caption += r" TriTopic's realized topic count was not recorded."
    if note:
        caption += " " + _latex_escape(note)
    if preliminary:
        caption += r" \textbf{Preliminary.}"
    lines = [
        "% Generated by scripts/analysis/make_paper_outputs.py",
        *([f"% {PRELIMINARY_NOTE}"] if preliminary else []),
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\begin{{tabular}}{{l{'c' * len(metrics)}r}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\midrule",
        *tests,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"
