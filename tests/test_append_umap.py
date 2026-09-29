import numpy as np
import pytest
from scipy.spatial.distance import pdist

from src.append_umap import AppendUMAP, _mean_pairwise_sq_dist
from src.models import get_algorithm


@pytest.fixture
def views():
    rng = np.random.default_rng(0)
    text = rng.normal(size=(40, 8))
    meta = rng.uniform(size=(40, 3))
    return text, meta


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_mean_pairwise_sq_dist_matches_brute_force(views):
    text, _ = views
    assert _mean_pairwise_sq_dist(text) == pytest.approx(
        np.mean(pdist(text, "sqeuclidean"))
    )


@pytest.mark.parametrize("w", [0.0, 0.1, 0.5, 1.0])
def test_weighted_join_distance_identity(views, w):
    """Joined squared distance is the exact w-mixture of calibrated view distances."""
    text, meta = views
    reducer = AppendUMAP(metadata=meta, metadata_weight=w, metric="euclidean")
    joined = reducer._concatenate_metadata(text, fit=True)

    d_text = pdist(_unit(text), "sqeuclidean")
    d_meta = pdist(meta, "sqeuclidean")
    expected = (1 - w) * d_text / d_text.mean() + w * d_meta / d_meta.mean()
    np.testing.assert_allclose(pdist(joined, "sqeuclidean"), expected, atol=1e-12)


def test_weight_zero_ignores_metadata_and_keeps_cosine_ranking(views):
    text, meta = views
    reducer = AppendUMAP(metadata=meta, metadata_weight=0.0, metric="euclidean")
    joined = reducer._concatenate_metadata(text, fit=True)

    cosine_rank = np.argsort(pdist(text, "cosine"))
    joined_rank = np.argsort(pdist(joined, "sqeuclidean"))
    np.testing.assert_array_equal(cosine_rank, joined_rank)


def test_transform_reuses_fitted_scales(views):
    text, meta = views
    reducer = AppendUMAP(metadata=meta, metadata_weight=0.3, metric="euclidean")
    fitted = reducer._concatenate_metadata(text, fit=True)
    scales = (reducer.text_scale_, reducer.metadata_scale_)

    again = reducer._concatenate_metadata(text)
    np.testing.assert_array_equal(fitted, again)
    assert (reducer.text_scale_, reducer.metadata_scale_) == scales


def test_transform_before_fit_raises(views):
    text, meta = views
    reducer = AppendUMAP(metadata=meta, metadata_weight=0.3, metric="euclidean")
    with pytest.raises(ValueError, match="not been fitted"):
        reducer._concatenate_metadata(text)


@pytest.mark.parametrize("w", [-0.1, 1.5, True])
def test_invalid_weight_rejected(views, w):
    _, meta = views
    with pytest.raises(ValueError, match="metadata_weight"):
        AppendUMAP(metadata=meta, metadata_weight=w, metric="euclidean")


def test_weighted_mode_requires_euclidean_metric(views):
    _, meta = views
    with pytest.raises(ValueError, match="euclidean"):
        AppendUMAP(metadata=meta, metadata_weight=0.3, metric="cosine")


def test_constant_metadata_cannot_be_calibrated(views):
    text, _ = views
    reducer = AppendUMAP(
        metadata=np.ones((40, 2)), metadata_weight=0.3, metric="euclidean"
    )
    with pytest.raises(ValueError, match="zero spread"):
        reducer._concatenate_metadata(text, fit=True)


def test_naive_mode_unchanged(views):
    text, meta = views
    reducer = AppendUMAP(metadata=meta, metric="cosine")
    np.testing.assert_array_equal(
        reducer._concatenate_metadata(text, fit=True), np.hstack((text, meta))
    )


def test_fit_transform_appends_metadata_once(views):
    text, meta = views
    reducer = AppendUMAP(metadata=meta, n_components=2, random_state=0)
    embedding = reducer.fit_transform(text)
    assert reducer._raw_data.shape[1] == text.shape[1] + meta.shape[1]
    assert embedding.shape == (text.shape[0], 2)


def test_get_algorithm_passes_metadata_weight(views):
    _, meta = views
    reducer = get_algorithm(
        {
            "type": "append_umap",
            "params": {"metric": "euclidean", "metadata_weight": 0.2},
        },
        metadata=meta,
        random_state=1,
    )
    assert isinstance(reducer, AppendUMAP)
    assert reducer.metadata_weight == 0.2
