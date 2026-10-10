"""Paper tables and figures for the RQ1 ablations."""

import polars as pl
import pytest

from src import paper_outputs
from src.comparisons.analysis import REQUESTED_TOPICS, SEEDS
from src.evaluation import EVALUATION_PROTOCOL

CATALOG = {
    "baseline": {
        "role": "baseline",
        "family": "hdbscan",
        "label": "UMAP + HDBSCAN",
        "short_label": "UMAP + HDBSCAN",
        "latex_label": r"$\text{BERTopic}_\text{H}$",
    },
    "append_umap": {
        "role": "ablation",
        "family": "hdbscan",
        "label": "Append",
        "short_label": "Append UMAP",
        "latex_label": r"$\text{\systemshort}_\text{Ap}$",
        "baseline_id": "baseline",
    },
    "append_umap_w010": {
        "role": "ablation",
        "family": "hdbscan",
        "label": "Weighted 0.1",
        "baseline_id": "baseline",
    },
}
DATASETS = ("anes", "fed")
DOCUMENTS = {"anes": 200, "fed": 400}


def _results(protocol=None) -> pl.DataFrame:
    """Append beats the baseline by 0.02 NPMI and loses 0.1 diversity."""
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for k in REQUESTED_TOPICS:
                for model, bonus, topics, outliers in (
                    ("baseline", 0.0, 20, 40),
                    ("append_umap", 0.02, 12, 0),
                    ("append_umap_w010", 0.01, 20, 20),
                ):
                    rows.append(
                        {
                            "catalog_id": model,
                            "role": CATALOG[model]["role"],
                            "dataset_label": dataset,
                            "condition": "remove_rep_stopwords",
                            "random_state": seed,
                            "nr_topics": k,
                            "c_npmi": 0.1 + bonus,
                            "topic_diversity": 0.9 - bonus * 5,
                            "n_topics": topics,
                            "outliers": outliers,
                            "n_observations": DOCUMENTS[dataset],
                            "meta_ami_mean": 0.05 + bonus * 10,
                            "evaluation_protocol": protocol,
                        }
                    )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def comparison():
    return paper_outputs.compare(_results(), CATALOG, ("append_umap",))


def test_ablation_table_reports_deltas_topics_and_noise(comparison):
    datasets, summary = comparison
    table = paper_outputs.ablation_table(
        datasets,
        summary,
        CATALOG,
        variants=("append_umap",),
        metrics=("c_npmi", "topic_diversity"),
        documents=DOCUMENTS,
    )
    row = table.row(0, named=True)
    assert row["Proposed"] == "Append UMAP"
    assert row["c_npmi Δ"] == pytest.approx(0.02)
    assert row["c_npmi W/T/L"] == "2/0/0"
    assert row["topic_diversity Δ"] == pytest.approx(-0.1)
    assert row["Δ topics"] == pytest.approx(-8)
    # 40 noise documents fewer: 20 pp of 200 (anes) and 10 pp of 400 (fed).
    assert row["Δ noise"] == pytest.approx(-15)
    assert row["Datasets"] == 2
    assert row["Reference ID"] == "baseline" and row["Family"] == "hdbscan"
    assert row["Δ AMI"] == pytest.approx(0.2)
    assert row["AMI datasets"] == 2


def test_latex_tables_use_paper_notation(comparison):
    datasets, summary = comparison
    metrics = ("c_npmi", "c_v")
    table = paper_outputs.ablation_table(
        datasets, summary, CATALOG, variants=("append_umap",), metrics=metrics
    )
    main = paper_outputs.ablation_table_latex(
        table, CATALOG, metrics=metrics, preliminary=True
    )
    assert r"$C_\text{NPMI}$" in main and "$C_V$" in main
    assert "C_v" not in main
    assert r"\textbf{Preliminary.}" in main and main.startswith("% Generated")
    assert r"\textit{vs.} $\text{BERTopic}_\text{H}$" in main
    row = r"$\text{\systemshort}_\text{Ap}$ & $+0.020$\,{\scriptsize(2/0/0)}"
    assert row in main
    assert "$n$" not in main
    stats = paper_outputs.ablation_stats_latex(table, CATALOG, metrics=metrics)
    assert "Wilcoxon" not in stats and "smallest attainable" not in stats
    assert "Preliminary" not in stats


def test_preliminary_until_every_row_uses_the_current_protocol():
    assert paper_outputs.is_preliminary(_results())
    assert not paper_outputs.is_preliminary(_results(EVALUATION_PROTOCOL))
    mixed = pl.concat([_results(EVALUATION_PROTOCOL), _results().head(1)])
    assert paper_outputs.is_preliminary(mixed)
    # The stale row is a baseline run: it flags outputs using the baseline only.
    assert mixed.tail(1)["catalog_id"].item() == "baseline"
    assert paper_outputs.is_preliminary(mixed, ["baseline", "append_umap"])
    assert not paper_outputs.is_preliminary(mixed, ["append_umap_w010"])


def test_dose_response_maps_weights_and_noise_share():
    datasets, _ = paper_outputs.compare(_results(), CATALOG, ("append_umap_w010",))
    curve = paper_outputs.dose_response(datasets, DOCUMENTS)
    assert set(curve["w"].to_list()) == {0.1}
    noise = curve.filter(pl.col("Metric") == "noise_share").sort("Dataset")
    assert noise["Δ"].to_list() == pytest.approx([-10.0, -5.0])
    npmi = curve.filter(pl.col("Metric") == "c_npmi")
    assert npmi["Δ"].to_list() == pytest.approx([0.01, 0.01])


def test_figures_are_written(comparison, tmp_path):
    datasets, _ = comparison
    points = paper_outputs.tradeoff_points(
        datasets, CATALOG, "c_npmi", "topic_diversity"
    )
    assert points.height == 2
    written = paper_outputs.plot_tradeoff(
        points,
        "c_npmi",
        "topic_diversity",
        tmp_path / "tradeoff",
        variants=("append_umap",),
        catalog=CATALOG,
        preliminary=True,
        formats=("png",),
    )
    weighted, _ = paper_outputs.compare(_results(), CATALOG, ("append_umap_w010",))
    written += paper_outputs.plot_dose_response(
        paper_outputs.dose_response(weighted, DOCUMENTS),
        tmp_path / "dose",
        formats=("pdf",),
    )
    assert all(path.exists() and path.stat().st_size > 0 for path in written)


def test_load_results_annotates_condition_and_dataset(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    pl.DataFrame(
        {
            "experiment_id": ["fed_standard_baseline"],
            "model_name": ["baseline_1_seed36201624"],
            "dataset_name": ["fed_embeddings"],
            "stopword_removal": ["remove_rep_stopwords"],
            "random_state": [36201624],
        }
    ).write_csv(results / "fed_standard_merged.csv")
    loaded = paper_outputs.load_results(tmp_path)
    row = loaded.row(0, named=True)
    assert row["dataset_label"] == "fed"
    assert row["condition"] == "remove_rep_stopwords"
    assert row["catalog_id"] == "baseline"


def test_partial_ami_is_marked_and_spectral_noise_omitted(comparison):
    datasets, summary = comparison
    table = paper_outputs.ablation_table(
        datasets, summary, CATALOG, variants=("append_umap",), metrics=("c_npmi",)
    ).with_columns(pl.lit(1).alias("AMI datasets"))
    main = paper_outputs.ablation_table_latex(table, CATALOG, metrics=("c_npmi",))
    assert r"$+0.20$$^\dagger$" in main
    assert "AMI available for fewer datasets" in main
    spectral = table.with_columns(pl.lit("spectral").alias("Family"))
    lines = paper_outputs.ablation_table_latex(
        spectral, CATALOG, metrics=("c_npmi",)
    ).splitlines()
    row = next(line for line in lines if line.startswith(r"$\text{\systemshort}"))
    assert row.split(" & ")[-2] == "--"


def test_caption_note_names_the_trump_variant(comparison):
    datasets, summary = comparison
    table = paper_outputs.ablation_table(
        datasets, summary, CATALOG, variants=("append_umap",), metrics=("c_npmi",)
    )
    note = "Trump results use the Trump 25k sample."
    main = paper_outputs.ablation_table_latex(
        table, CATALOG, metrics=("c_npmi",), note=note
    )
    stats = paper_outputs.ablation_stats_latex(
        table, CATALOG, metrics=("c_npmi",), note=note
    )
    assert note in main and note in stats


def test_figure_labels_use_paper_notation():
    assert paper_outputs.figure_label("append_umap", CATALOG) == (
        r"$\mathrm{COVET}_\mathrm{Ap}$"
    )
    assert paper_outputs.figure_label("baseline", CATALOG) == (
        r"$\mathrm{BERTopic}_\mathrm{H}$"
    )
    # No latex_label: the plain name.
    assert paper_outputs.figure_label("append_umap_w010", CATALOG) == "Weighted 0.1"


def test_tradeoff_plots_only_the_given_variants(comparison, tmp_path):
    datasets, _ = comparison
    points = paper_outputs.tradeoff_points(datasets, CATALOG, "c_npmi", "irbo")
    extra = points.with_columns(pl.lit("mv_spectral").alias("Model ID"))
    written = paper_outputs.plot_tradeoff(
        pl.concat([points, extra]),
        "c_npmi",
        "irbo",
        tmp_path / "tradeoff",
        variants=("append_umap",),
        catalog=CATALOG,
        formats=("png",),
    )
    assert written[0].exists()


def test_weight_note_names_the_tables_weighted_append():
    catalog = {"append_umap_w010": {"latex_label": r"$X^{w}$"}}
    assert paper_outputs.weight_note(["append_umap_w010"], catalog) == (
        r" $X^{w}$ uses $w = 0.1$."
    )
    assert paper_outputs.weight_note(["baseline"], catalog) == ""


def test_heatmap_averages_seeds_per_topic_count(tmp_path):
    runs = paper_outputs.matched_runs(_results(), CATALOG, ("append_umap",))
    cells = paper_outputs.heatmap_cells(runs, "c_npmi")
    assert cells.height == len(DATASETS) * len(REQUESTED_TOPICS)
    assert cells["Mean Δ"].to_list() == pytest.approx([0.02] * cells.height)
    assert set(cells["Seeds"].to_list()) == {len(SEEDS)}
    written = paper_outputs.plot_heatmap(
        cells,
        "c_npmi",
        tmp_path / "heatmap",
        variants=("append_umap", "append_umap_w010"),
        catalog=CATALOG,
        preliminary=True,
    )
    assert [path.suffix for path in written] == [".pdf"]
    assert written[0].stat().st_size > 0
