from datetime import date

import numpy as np
import polars as pl
import pytest

from src.metadata_alignment import N_BINS, discretize, metadata_alignment


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
