from pathlib import Path

import numpy as np
import pytest
import yaml
from bertopic import BERTopic

import src.models as models


def test_bertopic_reference_profile_defaults():
    """
    Verifies that the reference BERTopic() instance in the pinned environment
    exposes the expected defaults for UMAP and HDBSCAN.
    """
    bt = BERTopic()
    umap_params = bt.umap_model.get_params(deep=False)
    hdbscan_params = bt.hdbscan_model.get_params(deep=False)

    # Reference UMAP settings
    assert umap_params["n_components"] == 5
    assert umap_params["n_neighbors"] == 15
    assert umap_params["min_dist"] == 0.0
    assert umap_params["metric"] == "cosine"
    assert umap_params["low_memory"] is False

    # Reference HDBSCAN settings
    assert hdbscan_params["min_cluster_size"] == 10
    assert hdbscan_params["metric"] == "euclidean"
    assert hdbscan_params["cluster_selection_method"] == "eom"
    assert hdbscan_params["prediction_data"] is True


def test_active_standard_configs_parity():
    """
    Audits all 274 active standard YAML configuration files across all 5 datasets
    (standard and stemmed) to guarantee 100% compliance with BERTopic defaults
    and dataset-specific clustering sizing decisions.
    """
    active_dirs = [
        "anes",
        "anes_stemmed",
        "fed",
        "fed_stemmed",
        "gadarian",
        "gadarian_stemmed",
        "trump",
        "trump_stemmed",
        "yelp",
        "yelp_stemmed",
    ]

    active_files = []
    for d in active_dirs:
        dir_path = Path("experiments") / d
        assert dir_path.exists(), f"Directory missing: {dir_path}"
        active_files.extend(list(dir_path.glob("*.yaml")))

    for f in active_files:
        with open(f, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        model_cfg = cfg.get("model", {})
        dr = model_cfg.get("dimensionality_reduction")
        if dr:
            dr_type = dr.get("type")
            dr_params = dr.get("params") or {}

            if dr_type in ("umap", "append_umap", "aligned_umap"):
                assert dr_params.get("n_components") == 5, f"{f}: n_components != 5"
                assert dr_params.get("n_neighbors") == 15, f"{f}: n_neighbors != 15"
                assert dr_params.get("min_dist") == 0.0, f"{f}: min_dist != 0.0"
                # Weighted AppendUMAP calibrates squared Euclidean distances on
                # L2-normalized text, which ranks text neighbors as cosine does.
                expected_metric = (
                    "euclidean"
                    if dr_type == "append_umap" and "metadata_weight" in dr_params
                    else "cosine"
                )
                assert dr_params.get("metric") == expected_metric, (
                    f"{f}: metric != {expected_metric}"
                )
                assert dr_params.get("low_memory") is False, f"{f}: low_memory != False"

            elif dr_type == "pca":
                assert dr_params.get("n_components") == 5, f"{f}: PCA n_components != 5"

        cl = model_cfg.get("clustering")
        if cl and cl.get("type") == "hdbscan":
            cl_params = cl.get("params") or {}
            dataset = f.parent.name.replace("_stemmed", "")

            # Dataset sizing policy
            if dataset in ("anes", "gadarian"):
                expected_min_size = 5
            elif dataset in ("fed", "yelp"):
                expected_min_size = 10
            elif dataset == "trump":
                expected_min_size = 30
            else:
                pytest.fail(f"Unknown dataset {dataset} in {f}")

            actual_min_size = cl_params.get("min_cluster_size")
            assert actual_min_size == expected_min_size, (
                f"{f}: expected min_cluster_size={expected_min_size}, "
                f"got {actual_min_size}"
            )
            assert cl_params.get("prediction_data") is True, (
                f"{f}: prediction_data != True"
            )


def test_pca_omitted_components_fallback():
    """
    Verifies that get_algorithm applies n_components=5 fallback
    if PCA params omit n_components.
    """
    config = {"type": "pca", "params": {}}
    model = models.get_algorithm(config, metadata=None, random_state=42)
    assert model.n_components == 5


def test_synthetic_fit_5d_reducers():
    """
    Synthetic fit verifying that UMAP, AppendUMAP, AlignedUMAP, and PCA
    produce 5 output dimensions upon fitting.
    """
    rng = np.random.default_rng(42)
    n_samples = 40
    n_features = 16
    n_meta = 4
    X = rng.standard_normal((n_samples, n_features))
    meta = rng.standard_normal((n_samples, n_meta))

    # 1. Plain UMAP
    u_cfg = {
        "type": "umap",
        "params": {
            "n_components": 5,
            "n_neighbors": 15,
            "min_dist": 0.0,
            "metric": "cosine",
            "low_memory": False,
        },
    }
    u_model = models.get_algorithm(u_cfg, metadata=None, random_state=42)
    u_reduced = u_model.fit_transform(X)
    assert u_reduced.shape == (n_samples, 5)

    # 2. AppendUMAP
    app_cfg = {
        "type": "append_umap",
        "params": {
            "n_components": 5,
            "n_neighbors": 15,
            "min_dist": 0.0,
            "metric": "cosine",
            "low_memory": False,
        },
    }
    app_model = models.get_algorithm(app_cfg, metadata=meta, random_state=42)
    app_reduced = app_model.fit_transform(X)
    assert app_reduced.shape == (n_samples, 5)

    # 3. AlignedUMAP
    ali_cfg = {
        "type": "aligned_umap",
        "params": {
            "n_components": 5,
            "n_neighbors": 15,
            "min_dist": 0.0,
            "metric": "cosine",
            "low_memory": False,
        },
    }
    ali_wrapper = models.get_algorithm(ali_cfg, metadata=meta, random_state=42)
    ali_wrapper.fit(X)
    ali_reduced = ali_wrapper.transform(X)
    assert ali_reduced.shape == (n_samples, 5)

    # 4. PCA
    pca_cfg = {"type": "pca", "params": {"n_components": 5}}
    pca_model = models.get_algorithm(pca_cfg, metadata=None, random_state=42)
    pca_reduced = pca_model.fit_transform(X)
    assert pca_reduced.shape == (n_samples, 5)
