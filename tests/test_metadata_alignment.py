import json
from datetime import date

import numpy as np
import polars as pl
import pytest

from src.comparisons.analysis import compute_ablation_comparisons
from src.metadata_alignment import (
    N_BINS,
    discretize,
    fill_run_alignment,
    load_covariate_alignment,
    metadata_alignment,
)


def test_discretize_bins_continuous_numerics_into_quantiles():
    labels = discretize(pl.Series(np.arange(100, dtype=float)))
    assert len(set(labels)) == N_BINS
    assert np.unique(labels, return_counts=True)[1].tolist() == [20] * N_BINS


def test_discretize_keeps_few_valued_numerics_and_categories():
    assert discretize(pl.Series([0, 1, 1, 0])).tolist() == ["0", "1", "1", "0"]
    assert discretize(pl.Series(["a", None, "b"])).tolist() == ["a", "NA", "b"]
    assert discretize(pl.Series([True, False])).tolist() == ["true", "false"]


def test_discretize_bins_dates():
    days = pl.Series([date(2020, 1, d) for d in range(1, 31)])
    assert len(set(discretize(days))) == N_BINS


def test_topics_equal_to_a_covariate_score_one():
    kind = ["A", "B", "C"] * 20
    topics = [{"A": 0, "B": 1, "C": 2}[k] for k in kind]
    result = metadata_alignment(topics, pl.DataFrame({"kind": kind}))
    assert result["meta_ami_by_covariate"]["kind"] == pytest.approx(1.0)


def test_unrelated_topics_score_near_zero():
    rng = np.random.default_rng(0)
    covariates = pl.DataFrame(
        {"kind": rng.choice(["A", "B"], 2000), "rate": rng.normal(size=2000)}
    )
    result = metadata_alignment(rng.integers(0, 30, 2000), covariates)
    assert abs(result["meta_ami_mean"]) < 0.01


def test_noise_counts_as_its_own_topic():
    kind = ["A"] * 10 + ["B"] * 10
    topics = [0] * 10 + [-1] * 10
    result = metadata_alignment(topics, pl.DataFrame({"kind": kind}))
    assert result["meta_ami_mean"] == pytest.approx(1.0)


def test_mean_is_over_covariates_and_rows_must_align():
    covariates = pl.DataFrame({"a": ["x", "y"] * 10, "b": ["u"] * 10 + ["v"] * 10})
    topics = [0, 1] * 10
    result = metadata_alignment(topics, covariates)
    expected = np.mean(list(result["meta_ami_by_covariate"].values()))
    assert result["meta_ami_mean"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="topic labels"):
        metadata_alignment(topics[:-1], covariates)


def _write_manifest(root, run_uid, by_covariate):
    run_dir = root / "anes" / run_uid
    run_dir.mkdir(parents=True)
    alignment = {
        "meta_ami_mean": float(np.mean(list(by_covariate.values()))),
        "meta_ami_by_covariate": by_covariate,
    }
    manifest = {"run_uid": run_uid, "metadata_alignment": alignment}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_covariate_alignment_prefers_run_time_values(tmp_path):
    _write_manifest(tmp_path / "runs", "new", {"age": 0.2, "party": 0.4})
    _write_manifest(tmp_path / "runs", "both", {"age": 0.1})
    backfill = tmp_path / "backfill.csv"
    pl.DataFrame(
        {
            "run_uid": ["old", "old", "both"],
            "meta_ami_mean": [0.05, 0.05, 0.9],
            "covariate": ["age", "party", "age"],
            "ami": [0.0, 0.1, 0.9],
        }
    ).write_csv(backfill)

    alignment = load_covariate_alignment(tmp_path / "runs", backfill)

    sources = dict(alignment.select("run_uid", "ami_source").unique().iter_rows())
    assert sources == {"new": "run_time", "both": "run_time", "old": "backfill"}
    assert alignment.filter(pl.col("run_uid") == "both")["ami"].to_list() == [0.1]
    assert alignment.height == 5


def test_load_covariate_alignment_without_backfill(tmp_path):
    _write_manifest(tmp_path, "new", {"age": 0.2})
    alignment = load_covariate_alignment(tmp_path, tmp_path / "missing.csv")
    assert alignment["run_uid"].to_list() == ["new"]


def test_fill_run_alignment_keeps_existing_values_and_order():
    results = pl.DataFrame(
        {
            "run_uid": ["b", "a", None, "c"],
            "meta_ami_mean": [0.3, None, None, None],
        }
    )
    alignment = pl.DataFrame(
        {"run_uid": ["a", "a", "b"], "meta_ami_mean": [0.1, 0.1, 0.9]}
    )
    filled = fill_run_alignment(results, alignment)
    assert filled["run_uid"].to_list() == ["b", "a", None, "c"]
    assert filled["meta_ami_mean"].to_list() == [0.3, 0.1, None, None]


def test_fill_run_alignment_adds_missing_column():
    results = pl.DataFrame({"run_uid": ["a"], "c_v": [0.5]})
    alignment = pl.DataFrame({"run_uid": ["a"], "meta_ami_mean": [0.2]})
    assert fill_run_alignment(results, alignment)["meta_ami_mean"].to_list() == [0.2]


def test_ablation_alignment_delta_is_variant_minus_reference():
    catalog = {
        "base": {"role": "baseline", "label": "Base"},
        "var": {"role": "ablation", "label": "Var", "baseline_id": "base"},
    }
    rows = [
        {
            "catalog_id": model,
            "role": catalog[model]["role"],
            "dataset_label": "anes",
            "condition": "remove_rep_stopwords",
            "random_state": 36201624,
            "nr_topics": 10,
            "meta_ami_mean": ami,
        }
        for model, ami in (("base", 0.05), ("var", 0.02))
    ]
    datasets, _, _ = compute_ablation_comparisons(pl.DataFrame(rows), catalog)
    row = datasets.filter(
        (pl.col("Metric") == "meta_ami_mean") & (pl.col("Dataset") == "anes")
    ).row(0, named=True)
    assert row["Direction"] == "outcome"
    assert row["Improvement delta"] == pytest.approx(-0.03)
