"""Noise-coverage table (T3)."""

import polars as pl
import pytest

from src import paper_noise
from src.comparisons.analysis import BENCHMARK_DATASETS, REQUESTED_TOPICS, SEEDS

CATALOG = {
    "baseline": {"latex_label": r"$\text{BERTopic}_\text{H}$"},
    "append_umap": {"latex_label": r"$\text{\systemshort}_\text{Ap}$"},
}


def _results(model, outliers_by_seed, datasets=BENCHMARK_DATASETS, documents=200):
    rows = []
    for dataset in datasets:
        for seed, outliers in zip(SEEDS, outliers_by_seed):
            for k in REQUESTED_TOPICS:
                rows.append(
                    {
                        "catalog_id": model,
                        "dataset_label": dataset,
                        "condition": "remove_rep_stopwords",
                        "source_file": f"{dataset}_standard_merged",
                        "random_state": seed,
                        "nr_topics": k,
                        "outliers": outliers,
                        "n_observations": documents,
                    }
                )
    return rows


def test_noise_share_is_averaged_over_runs():
    results = pl.DataFrame(
        _results("baseline", (20, 40, 60))
        + _results("append_umap", (0, 0, 0), BENCHMARK_DATASETS[:-1])
    )
    coverage = paper_noise.noise_coverage(results, ("baseline", "append_umap"))
    baseline = coverage.filter(pl.col("Model ID") == "baseline").row(0, named=True)
    assert baseline["Runs"] == 15 and baseline["Complete"]
    assert baseline["Noise mean"] == pytest.approx(20.0)
    assert baseline["Noise SD"] == pytest.approx(pl.Series([10.0, 20, 30] * 5).std())
    assert coverage.filter(pl.col("Model ID") == "append_umap").height == 4


def test_noise_table_marks_missing_datasets_and_incomplete_runs():
    results = pl.DataFrame(
        _results("baseline", (20, 40, 60))
        + _results("append_umap", (0, 0, 0), BENCHMARK_DATASETS[:-1])
    ).filter(
        # One missing run makes the baseline incomplete on the first dataset.
        ~(
            (pl.col("catalog_id") == "baseline")
            & (pl.col("dataset_label") == BENCHMARK_DATASETS[0])
            & (pl.col("nr_topics") == 10)
            & (pl.col("random_state") == SEEDS[0])
        )
    )
    coverage = paper_noise.noise_coverage(results, ("baseline", "append_umap"))
    latex = paper_noise.noise_table_latex(
        coverage, CATALOG, ("baseline", "append_umap"), note="A note."
    )
    lines = latex.splitlines()
    baseline = next(line for line in lines if line.startswith(r"$\text{BERTopic}"))
    assert r"$^\dagger$" in baseline.split(" & ")[1]
    append = next(line for line in lines if line.startswith(r"$\text{\systemshort}"))
    assert append.split(" & ")[-2] == "--"
    assert append.split(" & ")[-1].startswith(r"0.0$^\ddagger$")
    assert "Incomplete runs" in latex and "Over fewer datasets" in latex
    assert "A note." in latex and "Preliminary" not in latex


def test_noise_coverage_uses_the_datasets_grid():
    rows = [
        {**row, "nr_topics": k}
        for row in _results("baseline", (20, 40, 60), ("gadarian",))
        if row["nr_topics"] == 10
        for k in (4, 6, 8, 10, 12)
    ]
    common = paper_noise.noise_coverage(pl.DataFrame(rows), ("baseline",))
    assert common.row(0, named=True)["Runs"] == 3
    assert not common.row(0, named=True)["Complete"]
    dataset = paper_noise.noise_coverage(
        pl.DataFrame(rows), ("baseline",), grid="dataset"
    )
    assert dataset.row(0, named=True)["Runs"] == 15
    assert dataset.row(0, named=True)["Complete"]
