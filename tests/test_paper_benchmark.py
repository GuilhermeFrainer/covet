"""Free-for-all benchmark table (T2)."""

import json

import numpy as np
import polars as pl
import pytest
from scipy.stats import friedmanchisquare

from src import paper_benchmark
from src.comparisons.analysis import BENCHMARK_DATASETS, REQUESTED_TOPICS, SEEDS

CATALOG = {
    "baseline": {"role": "baseline", "latex_label": r"$\text{BERTopic}_1$"},
    "tritopic": {"role": "external_baseline", "short_label": "TriTopic"},
    "append_umap": {
        "role": "ablation",
        "latex_label": r"$\text{\systemshort}_\text{Ap}$",
    },
}
MODELS = [
    {"id": "baseline", "sources": ["baseline"]},
    {"id": "tritopic", "sources": ["tritopic", "fast_tritopic"]},
    {"id": "append_umap", "sources": ["append_umap"]},
]


def _runs(model, datasets=BENCHMARK_DATASETS, score=0.1, column="nr_topics"):
    rows = []
    for i, dataset in enumerate(datasets):
        for seed in SEEDS:
            for k in REQUESTED_TOPICS:
                row = {
                    "run_uid": f"{model}-{dataset}-{seed}-{k}",
                    "catalog_id": model,
                    "dataset_label": dataset,
                    "condition": "remove_rep_stopwords",
                    "source_file": f"{dataset}_standard_merged",
                    "random_state": seed,
                    "c_npmi": score + 0.01 * i,
                    "n_topics": k - 1,
                }
                # TriTopic's `n_topics` holds the requested count instead.
                row[column] = k
                rows.append(row)
    return rows


def _results():
    """Baseline > Append on every dataset; TriTopic (k in `n_topics`) in between.

    FastTriTopic stands in for TriTopic on the last dataset, and Append has no
    runs on the first dataset.
    """
    rows = _runs("baseline", score=0.3)
    rows += _runs("tritopic", BENCHMARK_DATASETS[:-1], score=0.2, column="n_topics")
    rows += _runs("fast_tritopic", BENCHMARK_DATASETS[-1:], 0.2, column="n_topics")
    rows += _runs("append_umap", BENCHMARK_DATASETS[1:], score=0.1)
    return pl.DataFrame(rows)


def test_friedman_matches_scipy_and_handles_perfect_agreement():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(5, 4))
    result = paper_benchmark.friedman(scores)
    chi2, _ = friedmanchisquare(*scores.T)
    assert result["Chi2"] == pytest.approx(chi2)
    assert result["W"] == pytest.approx(chi2 / (5 * 3))
    agreed = paper_benchmark.friedman(np.tile([3.0, 2.0, 1.0], (5, 1)))
    assert agreed["Mean ranks"].tolist() == [1.0, 2.0, 3.0]
    assert agreed["W"] == pytest.approx(1.0) and agreed["p"] == 0.0
    lower_is_better = paper_benchmark.friedman(
        np.tile([3.0, 2.0, 1.0], (5, 1)), "minimize"
    )
    assert lower_is_better["Mean ranks"].tolist() == [3.0, 2.0, 1.0]


def test_dataset_scores_use_fallback_sources_and_flag_incomplete_runs():
    results = _results()
    # Drop one run of the baseline on the first dataset.
    results = results.filter(
        ~(
            (pl.col("catalog_id") == "baseline")
            & (pl.col("dataset_label") == BENCHMARK_DATASETS[0])
            & (pl.col("nr_topics") == 50)
            & (pl.col("random_state") == SEEDS[0])
        )
    )
    scores = paper_benchmark.dataset_scores(results, MODELS, ("c_npmi",))
    tritopic = scores.filter(pl.col("Model ID") == "tritopic").sort("Dataset")
    assert tritopic["Complete"].all() and tritopic.height == 5
    assert set(tritopic["Source"]) == {"tritopic", "fast_tritopic"}
    # TriTopic's n_topics is the requested count, so it is not reported.
    assert tritopic["n_topics"].null_count() == 5
    baseline = scores.filter(pl.col("Model ID") == "baseline")
    first = baseline.filter(pl.col("Dataset") == BENCHMARK_DATASETS[0]).row(
        0, named=True
    )
    assert not first["Complete"] and first["Runs"] == 14


def test_benchmark_table_ranks_only_models_complete_everywhere():
    scores = paper_benchmark.dataset_scores(_results(), MODELS, ("c_npmi",))
    table, stats = paper_benchmark.benchmark_table(scores, MODELS, ("c_npmi",))
    rows = {row["Model ID"]: row for row in table.to_dicts()}
    assert rows["baseline"]["c_npmi rank"] == 1.0
    assert rows["tritopic"]["c_npmi rank"] == 2.0
    assert not rows["append_umap"]["Ranked"] and rows["append_umap"]["Datasets"] == 4
    assert rows["append_umap"]["c_npmi rank"] is None
    assert rows["baseline"]["n_topics"] == pytest.approx(29.0)
    assert stats["c_npmi"]["k"] == 2 and stats["c_npmi"]["N"] == 5

    latex = paper_benchmark.benchmark_table_latex(
        table, stats, CATALOG, ("c_npmi",), preliminary=True, note="A note."
    )
    assert r"$\text{BERTopic}_1$ & $.320$\,{\scriptsize(\textbf{1.0})} & 29" in latex
    assert r"$\text{\systemshort}_\text{Ap}$$^\dagger$ & $.115$ &" in latex
    assert r"fewer datasets ($\text{\systemshort}_\text{Ap}$: 4)" in latex
    assert "TriTopic's realized topic count" in latex
    assert r"\textbf{Preliminary.}" in latex and "A note." in latex
    assert "among the 2 models with complete runs on all 5 datasets" in latex


def test_models_without_results_are_left_out_of_the_latex_table():
    models = [*MODELS, {"id": "stm", "sources": ["stm"]}]
    scores = paper_benchmark.dataset_scores(_results(), models, ("c_npmi",))
    table, stats = paper_benchmark.benchmark_table(scores, models, ("c_npmi",))
    assert table.filter(pl.col("Model ID") == "stm")["Datasets"].item() == 0
    latex = paper_benchmark.benchmark_table_latex(table, stats, CATALOG, ("c_npmi",))
    assert "stm" not in latex.lower()


def test_scores_drop_the_leading_zero():
    assert paper_benchmark._score(0.509) == "$.509$"
    assert paper_benchmark._score(-0.033) == "$-.033$"
    assert paper_benchmark._score(-7.574) == "$-7.574$"
    assert paper_benchmark._score(None) == "--"


def test_tritopic_realized_topics_come_from_exported_topics(tmp_path):
    results = _results()
    tritopic = results.filter(pl.col("catalog_id").str.ends_with("tritopic"))
    # Every run lost one topic and has a noise topic, which is not counted.
    for row in tritopic.iter_rows(named=True):
        run_dir = tmp_path / row["dataset_label"] / row["run_uid"]
        run_dir.mkdir(parents=True)
        topics = [{"topic_id": i} for i in range(-1, row["n_topics"] - 1)]
        (run_dir / "topics.json").write_text(json.dumps(topics), encoding="utf-8")
    realized = paper_benchmark.realized_topic_counts(
        [*tritopic["run_uid"].to_list(), None, "missing"], tmp_path
    )
    assert len(realized) == tritopic.height
    scores = paper_benchmark.dataset_scores(results, MODELS, ("c_npmi",), realized)
    counts = scores.filter(pl.col("Model ID") == "tritopic")["n_topics"]
    # Requested 10..50 (mean 30), realized one fewer.
    assert counts.to_list() == pytest.approx([29.0] * 5)
    # Without the exports the count stays unknown.
    partial = dict(list(realized.items())[1:])
    scores = paper_benchmark.dataset_scores(results, MODELS, ("c_npmi",), partial)
    assert scores.filter(pl.col("Model ID") == "tritopic")["n_topics"].null_count() == 1


def test_benchmark_config_names_catalog_models():
    from src.model_catalog import load_catalog

    catalog = load_catalog()
    for model in paper_benchmark.load_benchmark_models():
        assert model["id"] in catalog
        assert all(source in catalog for source in model["sources"])
