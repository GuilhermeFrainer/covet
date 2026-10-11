"""Topic-count slicing of the RQ1 comparisons and their presentation tables."""

import polars as pl
import pytest

from src.comparisons import views
from src.comparisons.analysis import (
    REQUESTED_TOPICS,
    SEEDS,
    compute_registered_edge_comparisons,
)

CATALOG = {
    "base": {"role": "baseline", "label": "Base model", "short_label": "Base"},
    "var": {
        "role": "ablation",
        "label": "Variant model",
        "short_label": "Variant",
        "baseline_id": "base",
    },
}
EDGES = [
    {
        "id": "base_var",
        "reference_model": "base",
        "variant_model": "var",
        "chain": "Chain A",
        "label": "Base → Variant",
        "intended_change": "Adds metadata.",
        "classification": "architectural",
        "main_figure": True,
    }
]
DATASETS = ("anes", "fed", "gadarian")


def _results() -> pl.DataFrame:
    """Variant beats the reference by k / 1000 on NPMI, everywhere."""
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for k in REQUESTED_TOPICS:
                for model, bonus, topics in (
                    ("base", 0.0, k - 1),
                    ("var", k / 1000, 7),
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
                            "n_topics": topics,
                            "outliers": 5,
                        }
                    )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def full_grid():
    return compute_registered_edge_comparisons(_results(), CATALOG, EDGES)


@pytest.fixture(scope="module")
def k20_slice():
    return compute_registered_edge_comparisons(
        _results(), CATALOG, EDGES, requested_topics=(20,)
    )


def test_topic_slice_restricts_the_grid(full_grid, k20_slice):
    # Benchmark datasets without results keep rows with no delta.
    with_data = (
        pl.col("Metric").eq("c_npmi") & pl.col("Improvement delta").is_not_null()
    )
    datasets, summary, runs = k20_slice
    npmi = datasets.filter(with_data)
    assert npmi["Expected cells"].unique().to_list() == [len(SEEDS)]
    assert npmi["Improvement delta"].to_list() == pytest.approx([0.02] * 3)
    assert runs["Requested topics"].unique().to_list() == [20]
    row = summary.filter(pl.col("Metric") == "c_npmi").row(0, named=True)
    assert row["Datasets"] == 3 and row["Wins"] == 3
    full = full_grid[0].filter(with_data)
    assert full["Expected cells"].unique().to_list() == [15]
    assert full["Improvement delta"].to_list() == pytest.approx([0.03] * 3)


def test_comparison_table_splits_reference_and_proposed(full_grid):
    datasets, summary, _ = full_grid
    table = views.comparison_table(EDGES, CATALOG, summary, datasets, "c_npmi")
    row = table.row(0, named=True)
    assert "Intended change" not in table.columns and "Type" not in table.columns
    assert (row["Reference"], row["Proposed"]) == ("Base", "Variant")
    assert row["Reference score"] == pytest.approx(0.1)
    assert row["Proposed score"] == pytest.approx(0.13)
    assert row["W/T/L"] == "3/0/0"
    assert row["Datasets"] == "3/5"


def test_dataset_table_counts_runs_and_reports_realized_topics(full_grid):
    runs = full_grid[2]
    table = views.dataset_table(EDGES, CATALOG, runs, "c_npmi", "fed")
    row = table.row(0, named=True)
    assert row["Runs"] == 15 and row["Run W/T/L"] == "15/0/0"
    assert row["Mean Δ"] == pytest.approx(0.03)
    assert row["Reference topics"] == pytest.approx(29)
    assert row["Proposed topics"] == pytest.approx(7)
    sliced = views.dataset_table(
        EDGES, CATALOG, runs, "c_npmi", "fed", requested_topics=(50,)
    )
    assert sliced.row(0, named=True)["Runs"] == 3


def test_seed_table_and_trend(full_grid):
    runs = full_grid[2]
    seeds = views.seed_table(EDGES, CATALOG, runs, "c_npmi", "anes", 30)
    assert sorted(seeds["Seed"].to_list()) == sorted(SEEDS)
    assert seeds["Δ"].to_list() == pytest.approx([0.03] * 3)
    trend = views.delta_by_topic_count(EDGES, CATALOG, runs, "c_npmi", "anes")
    assert trend["Requested topics"].to_list() == list(REQUESTED_TOPICS)
    assert trend["Mean Δ"].to_list() == pytest.approx(
        [k / 1000 for k in REQUESTED_TOPICS]
    )
    assert trend["SD Δ"].to_list() == pytest.approx([0.0] * 5)


def test_heatmap_has_one_cell_per_dataset_and_topic_count(full_grid):
    cells = views.heatmap_cells(EDGES, CATALOG, full_grid[2], "c_npmi")
    assert cells.height == len(DATASETS) * len(REQUESTED_TOPICS)
    assert set(cells["Seeds"].to_list()) == {len(SEEDS)}
    assert "fed · k=40" in cells["Cell"].to_list()
    assert cells["Comparison"].unique().to_list() == ["Base → Variant"]


def test_use_trump_variant_counts_exactly_one_trump():
    from src.comparisons.analysis import use_trump_variant

    rows = pl.DataFrame(
        {"dataset_label": ["fed", "trump", "trump_s25000"], "value": [1, 2, 3]}
    )
    full = use_trump_variant(rows, "trump")
    assert full["dataset_label"].to_list() == ["fed", "trump"]
    assert full["value"].to_list() == [1, 2]
    sample = use_trump_variant(rows, "trump_s25000")
    assert sample["dataset_label"].to_list() == ["fed", "trump"]
    assert sample["value"].to_list() == [1, 3]
    with pytest.raises(ValueError):
        use_trump_variant(rows, "trump_s10")


def _two_grid_results() -> pl.DataFrame:
    """`_results` plus Gadarian and ANES runs on their own topic-count grids."""
    rows = _results().to_dicts()
    for dataset, counts in (("gadarian", (4, 6, 8, 12)), ("anes", (8, 11, 14, 17))):
        for seed in SEEDS:
            for k in counts:
                for model, bonus in (("base", 0.0), ("var", k / 1000)):
                    rows.append(
                        {
                            **rows[0],
                            "catalog_id": model,
                            "role": CATALOG[model]["role"],
                            "dataset_label": dataset,
                            "random_state": seed,
                            "nr_topics": k,
                            "c_npmi": 0.1 + bonus,
                        }
                    )
    return pl.DataFrame(rows)


def _npmi_deltas(grid, requested_topics=None):
    datasets, _, _ = compute_registered_edge_comparisons(
        _two_grid_results(),
        CATALOG,
        EDGES,
        requested_topics=requested_topics,
        grid=grid,
    )
    rows = datasets.filter(
        pl.col("Metric").eq("c_npmi") & pl.col("Improvement delta").is_not_null()
    )
    return dict(zip(rows["Dataset"], rows["Improvement delta"]))


def test_dataset_grid_matches_each_dataset_on_its_own_topic_counts():
    # The variant's NPMI bonus is k / 1000, so Δ is the mean requested count.
    assert _npmi_deltas("common") == pytest.approx(
        {"anes": 0.03, "fed": 0.03, "gadarian": 0.03}
    )
    assert _npmi_deltas("dataset") == pytest.approx(
        {"anes": 0.014, "fed": 0.03, "gadarian": 0.008}
    )


def test_topic_slice_off_a_datasets_grid_leaves_the_dataset_out():
    assert _npmi_deltas("dataset", requested_topics=(30,)) == pytest.approx(
        {"fed": 0.03}
    )
    assert _npmi_deltas("dataset", requested_topics=(8,)) == pytest.approx(
        {"anes": 0.008, "gadarian": 0.008}
    )
