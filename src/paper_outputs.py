"""Paper tables and figures for the RQ1 ablations.

All numbers come from `comparisons.analysis.compute_ablation_comparisons`, so
they match the dashboard: per dataset, the proposed model is paired with its
catalog reference by seed and requested topic count, and the differences are
averaged. Deltas are improvement-oriented (positive favours the proposed
model); topic counts, noise share and topic-metadata AMI are reported as
proposed minus reference.

Outputs are marked preliminary while any result row was scored with an older
evaluation protocol than `evaluation.EVALUATION_PROTOCOL`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.comparisons.analysis import BENCHMARK_DATASETS, compute_ablation_comparisons
from src.comparisons.views import METRIC_LABELS, STANDARD_CONDITION, model_name
from src.evaluation import EVALUATION_PROTOCOL
from src.experiment_tracker import classify_result_condition
from src.metadata_alignment import fill_run_alignment, load_covariate_alignment
from src.model_catalog import annotate_models, load_catalog
from src.results_analysis import canonical_dataset_expr

HDBSCAN_VARIANTS = (
    "append_umap",
    "aligned_umap",
    "feature_stacking_hdbscan",
    "mv_hdbscan",
)
TABLE_METRICS = ("c_npmi", "c_v", "irbo", "topic_diversity")
WEIGHTED_APPEND = (
    ("append_umap_w000", 0.0),
    ("append_umap_w005", 0.05),
    ("append_umap_w010", 0.1),
    ("append_umap_w020", 0.2),
    ("append_umap_w030", 0.3),
    ("append_umap_w050", 0.5),
)
DATASET_MARKERS = dict(zip(BENCHMARK_DATASETS, ("o", "s", "^", "D", "X")))
DATASET_COLORS = dict(
    zip(BENCHMARK_DATASETS, ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"))
)
PRELIMINARY_NOTE = "Preliminary: coherence scored with the pre-2026-10 evaluation."
# Metric names in the paper's notation.
LATEX_METRIC_LABELS = {
    "c_npmi": r"$C_\text{NPMI}$",
    "c_v": r"$C_V$",
    "u_mass": r"$C_\text{UMass}$",
    "irbo": "IRBO",
    "topic_diversity": "Diversity",
    "meta_ami_mean": "AMI",
}


def load_results(project_root: Path) -> pl.DataFrame:
    """Loads top-level result CSVs the way the dashboard does for comparisons.

    Adds `source_file`, `dataset_label` and `condition`, annotates catalog
    roles, and fills topic-metadata AMI from run manifests and the backfill.
    """
    results_dir = project_root / "results"
    frames = []
    for path in sorted(results_dir.glob("*.csv")):
        frame = pl.read_csv(path, infer_schema_length=None)
        if frame.is_empty() or "dataset_name" not in frame.columns:
            continue
        first = frame.row(0, named=True)
        condition = classify_result_condition(
            path.name,
            str(first.get("experiment_id") or ""),
            first.get("stopword_removal"),
        )
        frames.append(
            frame.with_columns(
                canonical_dataset_expr(pl.col("dataset_name")),
                pl.lit(path.stem).alias("source_file"),
                pl.lit(condition).alias("condition"),
            ).with_columns(
                # Legacy rows name the embeddings file, as the dashboard handles.
                pl.col("dataset_name")
                .str.replace("_embeddings$", "")
                .alias("dataset_label")
            )
        )
    results = pl.concat(frames, how="diagonal_relaxed")
    if "model_name" in results.columns:
        results = results.with_columns(
            pl.col("model_name").str.replace("^stemmed_", "")
        )
    results = annotate_models(results, load_catalog())
    alignment = load_covariate_alignment(
        project_root / "output" / "document_assignments",
        results_dir / "derived" / "metadata_alignment_backfill.csv",
    )
    return fill_run_alignment(results, alignment)


def is_preliminary(results: pl.DataFrame) -> bool:
    """True when any standard-condition row predates the current protocol."""
    standard = results.filter(pl.col("condition") == STANDARD_CONDITION)
    if "evaluation_protocol" not in standard.columns:
        return True
    return (
        standard.filter(
            pl.col("evaluation_protocol").is_null()
            | (pl.col("evaluation_protocol") != EVALUATION_PROTOCOL)
        ).height
        > 0
    )


def documents_per_dataset(results: pl.DataFrame) -> dict[str, int]:
    """Corpus size per dataset, used to turn outlier counts into shares."""
    sizes = (
        results.filter(pl.col("condition") == STANDARD_CONDITION)
        .group_by("dataset_label")
        .agg(pl.col("n_observations").cast(pl.Int64, strict=False).max())
    )
    return {row[0]: row[1] for row in sizes.iter_rows() if row[1]}


def compare(results: pl.DataFrame, catalog: dict, variants) -> tuple:
    """Dataset-level and cross-dataset comparisons for `variants`.

    Holm adjustment runs within each metric across `variants`.
    """
    datasets, summary, _ = compute_ablation_comparisons(
        results, catalog, summary_model_ids=set(variants)
    )
    datasets = datasets.filter(
        pl.col("Model ID").is_in(list(variants))
        & (pl.col("Condition") == STANDARD_CONDITION)
    )
    summary = summary.filter(pl.col("Model ID").is_in(list(variants)))
    return datasets, summary


def _complete(datasets: pl.DataFrame) -> pl.DataFrame:
    """Dataset rows with every expected seed x topic-count cell matched."""
    return datasets.filter(
        pl.col("Improvement delta").is_not_null()
        & (pl.col("Matched cells") == pl.col("Expected cells"))
    )


def _raw_change(datasets: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Proposed minus reference per complete dataset for an outcome metric."""
    return (
        _complete(datasets)
        .filter(pl.col("Metric") == metric)
        .select(
            "Model ID",
            "Dataset",
            (pl.col("Ablation score") - pl.col("Baseline score")).alias("change"),
        )
    )


def ablation_table(
    datasets: pl.DataFrame,
    summary: pl.DataFrame,
    catalog: dict,
    variants=HDBSCAN_VARIANTS,
    metrics=TABLE_METRICS,
    documents: dict[str, int] | None = None,
) -> pl.DataFrame:
    """One row per variant: cross-dataset results for each metric.

    Columns per metric: mean dataset Δ, W/T/L, rank-biserial r and Holm p.
    Also the mean change in realized topics, in noise share (percentage
    points, when `documents` is given) and the tested datasets.
    """
    by_key = {(r["Model ID"], r["Metric"]): r for r in summary.to_dicts()}
    topics = _raw_change(datasets, "n_topics")
    outliers = _raw_change(datasets, "outliers")
    if documents:
        outliers = outliers.with_columns(
            (
                pl.col("change")
                / pl.col("Dataset").replace_strict(documents, default=None)
                * 100
            ).alias("change")
        )
    rows = []
    for variant in variants:
        row = {"Model ID": variant, "Proposed": model_name(variant, catalog)}
        for metric in metrics:
            result = by_key.get((variant, metric), {})
            wins = result.get("Wins")
            row[f"{metric} Δ"] = result.get("Mean dataset delta")
            row[f"{metric} W/T/L"] = (
                f"{wins}/{result.get('Ties')}/{result.get('Losses')}"
                if wins is not None
                else None
            )
            row[f"{metric} r"] = result.get("Rank-biserial effect")
            row[f"{metric} Holm p"] = result.get("Holm adjusted p")
            row[f"{metric} datasets"] = result.get("Included datasets")
        row["Δ topics"] = topics.filter(pl.col("Model ID") == variant)["change"].mean()
        row["Δ noise"] = (
            outliers.filter(pl.col("Model ID") == variant)["change"].mean()
            if documents
            else None
        )
        tested = by_key.get((variant, metrics[0]), {})
        row["Datasets"] = tested.get("Datasets") or 0
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def _latex_escape(text: str) -> str:
    for char, escaped in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(char, escaped)
    return text


def _latex_metric(metric: str) -> str:
    return LATEX_METRIC_LABELS.get(metric, _latex_escape(metric))


def _signed(value, digits: int = 3) -> str:
    return "--" if value is None else f"${value:+.{digits}f}$"


def ablation_table_latex(
    table: pl.DataFrame,
    catalog: dict,
    reference: str = "baseline",
    metrics=TABLE_METRICS,
    preliminary: bool = False,
    label: str = "tab:hdbscan_ablation",
) -> str:
    """Booktabs `table*`: mean Δ with W/T/L per metric, then topics and noise."""
    header = " & ".join(
        [r"\textbf{Proposed}", r"\textbf{$n$}"]
        + [rf"\textbf{{{_latex_metric(m)}}}" for m in metrics]
        + [r"\textbf{$\Delta$ topics}", r"\textbf{$\Delta$ noise (pp)}"]
    )
    body = []
    for row in table.to_dicts():
        cells = [_latex_escape(row["Proposed"]), str(row["Datasets"])]
        for metric in metrics:
            wtl = row[f"{metric} W/T/L"]
            cells.append(
                _signed(row[f"{metric} Δ"])
                + (rf"\,{{\scriptsize({wtl})}}" if wtl else "")
            )
        cells.append(_signed(row["Δ topics"], 1))
        cells.append(_signed(row["Δ noise"], 1))
        body.append(" & ".join(cells) + r" \\")
    caption = (
        "Change relative to "
        + _latex_escape(model_name(reference, catalog))
        + r", averaged over datasets (each dataset averages 3 seeds $\times$ 5 "
        r"requested topic counts). Positive values favour the proposed model; "
        r"W/T/L counts datasets. $\Delta$ topics and $\Delta$ noise (share of "
        r"documents assigned to the noise cluster, in percentage points) are "
        r"proposed minus reference. $n$: datasets with complete runs."
    )
    if preliminary:
        caption += r" \textbf{Preliminary.}"
    columns = "l" + "c" * (len(metrics) + 3)
    lines = [
        "% Generated by scripts/analysis/make_paper_outputs.py",
        *([f"% {PRELIMINARY_NOTE}"] if preliminary else []),
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def ablation_stats_latex(
    table: pl.DataFrame,
    metrics=TABLE_METRICS,
    preliminary: bool = False,
    label: str = "tab:hdbscan_ablation_stats",
) -> str:
    """Rank-biserial r and Holm-adjusted exact p for each variant and metric."""
    header = " & ".join(
        [r"\textbf{Proposed}"] + [rf"\textbf{{{_latex_metric(m)}}}" for m in metrics]
    )
    body = []
    for row in table.to_dicts():
        cells = [_latex_escape(row["Proposed"])]
        for metric in metrics:
            r, p = row[f"{metric} r"], row[f"{metric} Holm p"]
            cells.append(
                "--"
                if r is None
                else f"${r:+.2f}$ ({p:.3f})"
                if p is not None
                else f"${r:+.2f}$"
            )
        body.append(" & ".join(cells) + r" \\")
    tested = max((row["Datasets"] for row in table.to_dicts()), default=0)
    floor = (
        f"With {tested} datasets the smallest attainable two-sided $p$ is "
        f"${2 / 2**tested:.4g}$, so the tests"
        if tested
        else "The tests"
    )
    caption = (
        r"Rank-biserial correlation $r$ and, in parentheses, Holm-adjusted exact "
        r"Wilcoxon signed-rank $p$ across datasets. "
        + floor
        + r" summarize cross-dataset consistency rather than confirm effects."
    )
    if preliminary:
        caption += r" \textbf{Preliminary.}"
    lines = [
        "% Generated by scripts/analysis/make_paper_outputs.py",
        *([f"% {PRELIMINARY_NOTE}"] if preliminary else []),
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{l{'c' * len(metrics)}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def tradeoff_points(
    datasets: pl.DataFrame, catalog: dict, x_metric: str, y_metric: str
) -> pl.DataFrame:
    """Dataset-level Δ of two metrics per variant (complete datasets only)."""
    complete = _complete(datasets).select(
        "Model ID", "Dataset", "Metric", "Improvement delta"
    )

    def metric(name, alias):
        return complete.filter(pl.col("Metric") == name).select(
            "Model ID", "Dataset", pl.col("Improvement delta").alias(alias)
        )

    points = metric(x_metric, "x").join(
        metric(y_metric, "y"), on=["Model ID", "Dataset"]
    )
    return points.with_columns(
        pl.col("Model ID")
        .map_elements(lambda m: model_name(m, catalog), return_dtype=pl.String)
        .alias("Proposed")
    ).sort("Model ID", "Dataset")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42})
    return plt


def plot_tradeoff(
    points: pl.DataFrame,
    x_metric: str,
    y_metric: str,
    path_stem: Path,
    variants=HDBSCAN_VARIANTS,
    catalog: dict | None = None,
    preliminary: bool = False,
    formats=("pdf", "png"),
) -> list[Path]:
    """Scatter of Δ x vs Δ y: colour = proposed model, marker = dataset."""
    plt = _pyplot()
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    colors = dict(zip(variants, palette))
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--", zorder=0)
    ax.axvline(0, color="#888888", linewidth=0.7, linestyle="--", zorder=0)
    for row in points.to_dicts():
        ax.scatter(
            row["x"],
            row["y"],
            color=colors.get(row["Model ID"], "#444444"),
            marker=DATASET_MARKERS.get(row["Dataset"], "o"),
            s=28,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )
    x_label = METRIC_LABELS.get(x_metric, x_metric)
    y_label = METRIC_LABELS.get(y_metric, y_metric)
    ax.set_xlabel(f"Δ {x_label} (proposed − reference)")
    ax.set_ylabel(f"Δ {y_label}")
    ax.grid(alpha=0.2)
    model_handles = [
        plt.Line2D(
            [],
            [],
            color=colors[v],
            marker="o",
            linestyle="",
            label=model_name(v, catalog or {}),
        )
        for v in variants
        if v in set(points["Model ID"].to_list())
    ]
    dataset_handles = [
        plt.Line2D(
            [], [], color="#444444", marker=DATASET_MARKERS[d], linestyle="", label=d
        )
        for d in BENCHMARK_DATASETS
        if d in set(points["Dataset"].to_list())
    ]
    ax.legend(
        handles=model_handles + dataset_handles,
        fontsize=6,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    if preliminary:
        ax.set_title("Preliminary (pre-fix coherence)", fontsize=7, color="#B22222")
    return _save(fig, plt, path_stem, formats)


def dose_response(
    datasets: pl.DataFrame, documents: dict[str, int] | None = None
) -> pl.DataFrame:
    """Δ per weighted-Append weight and dataset (complete datasets only).

    Quality metrics are improvement-oriented; `n_topics`, `meta_ami_mean` and
    the noise share (from `outliers`, in percentage points) are proposed minus
    reference.
    """
    weights = dict(WEIGHTED_APPEND)
    complete = _complete(datasets).filter(pl.col("Model ID").is_in(list(weights)))
    rows = []
    for row in complete.to_dicts():
        metric, value = row["Metric"], row["Improvement delta"]
        if metric == "outliers":
            if not documents or row["Dataset"] not in documents:
                continue
            metric = "noise_share"
            value = (
                (row["Ablation score"] - row["Baseline score"])
                / documents[row["Dataset"]]
                * 100
            )
        rows.append(
            {
                "w": weights[row["Model ID"]],
                "Dataset": row["Dataset"],
                "Metric": metric,
                "Δ": value,
            }
        )
    return pl.DataFrame(rows).sort("Metric", "Dataset", "w") if rows else pl.DataFrame()


DOSE_PANELS = (
    ("c_npmi", "Δ NPMI"),
    ("topic_diversity", "Δ Topic diversity"),
    ("noise_share", "Δ noise share (pp)"),
    ("meta_ami_mean", "Δ topic–metadata AMI"),
)


def plot_dose_response(
    curve: pl.DataFrame,
    path_stem: Path,
    preliminary: bool = False,
    formats=("pdf", "png"),
) -> list[Path]:
    """2 × 2 panels of Δ against the metadata weight w, one line per dataset."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2), sharex=True)
    for ax, (metric, title) in zip(axes.flat, DOSE_PANELS):
        ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--", zorder=0)
        for dataset in BENCHMARK_DATASETS:
            if curve.is_empty():
                break
            line = curve.filter(
                (pl.col("Metric") == metric) & (pl.col("Dataset") == dataset)
            )
            if line.is_empty():
                continue
            ax.plot(
                line["w"].to_list(),
                line["Δ"].to_list(),
                marker=DATASET_MARKERS[dataset],
                color=DATASET_COLORS[dataset],
                markersize=4,
                linewidth=1.2,
                label=dataset,
            )
        ax.set_title(title)
        ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("Metadata weight $w$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    if preliminary:
        fig.suptitle("Preliminary (pre-fix coherence)", fontsize=8, color="#B22222")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return _save(fig, plt, path_stem, formats)


def _save(fig, plt, path_stem: Path, formats) -> list[Path]:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = path_stem.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        paths.append(path)
    plt.close(fig)
    return paths


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
