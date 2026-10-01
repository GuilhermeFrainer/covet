"""Paper tables and figures for the RQ1 ablations."""

import polars as pl
import pytest

from src import paper_outputs
from src.comparisons.analysis import REQUESTED_TOPICS, SEEDS
from src.evaluation import EVALUATION_PROTOCOL

CATALOG = {
    "baseline": {
        "role": "baseline",
        "label": "UMAP + HDBSCAN",
        "short_label": "UMAP + HDBSCAN",
    },
    "append_umap": {
        "role": "ablation",
        "label": "Append",
        "short_label": "Append UMAP",
        "baseline_id": "baseline",
    },
    "append_umap_w010": {
        "role": "ablation",
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
    assert r"$+0.020$\,{\scriptsize(2/0/0)}" in main
    stats = paper_outputs.ablation_stats_latex(table, metrics=metrics)
    assert "With 2 datasets the smallest attainable two-sided $p$ is $0.5$" in stats
    assert "Preliminary" not in stats


def test_preliminary_until_every_row_uses_the_current_protocol():
    assert paper_outputs.is_preliminary(_results())
    assert not paper_outputs.is_preliminary(_results(EVALUATION_PROTOCOL))
    mixed = pl.concat([_results(EVALUATION_PROTOCOL), _results().head(1)])
    assert paper_outputs.is_preliminary(mixed)


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
