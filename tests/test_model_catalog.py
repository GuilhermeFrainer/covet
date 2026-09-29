from pathlib import Path

import polars as pl
import pytest
import yaml

from src.experiment_tracker import scan_experiment_configs
from src.model_catalog import (
    CATALOG_PATH,
    annotate_coverage,
    annotate_models,
    configuration_models,
    filter_catalog,
    kmeans_algorithms,
    load_catalog,
    resolve_model_id,
    sort_catalog,
)


@pytest.fixture
def catalog():
    return load_catalog()


def test_agreed_primary_boundaries(catalog):
    primary = {
        "baseline", "append_umap", "aligned_umap", "mv_hdbscan",
        "umap_spectral", "mv_spectral", "mv_spectral_info0",
        "mv_co_reg_spectral", "mv_co_reg_spectral_info0", "pca_k_means",
        "pca_mv_k_means", "pca_mv_spherical_k_means", "stm", "tritopic", "fast_tritopic",
    }
    assert all(catalog[mid]["priority"] == "primary" for mid in primary)
    for mid, entry in catalog.items():
        if mid.startswith(("append_umap_mv_", "aligned_umap_mv_")):
            assert entry["priority"] == "secondary"
        if entry["reduction"] == "pca" and entry["family"] != "k_means":
            assert entry["priority"] == "secondary"
        if entry["reduction"] == "umap" and entry["family"] == "k_means":
            assert entry["priority"] == "secondary"


def test_scope_keeps_reference_and_explicit_external_baselines(catalog):
    df = annotate_models(pl.DataFrame({"model_name": list(catalog)}), catalog)
    selected = filter_catalog(df, baselines=["baseline"])
    weighted_append = {f"append_umap_w{w}" for w in ("000", "005", "010", "020", "030", "050")}
    assert set(selected["catalog_id"]) == {
        "baseline", "append_umap", "aligned_umap", "mv_hdbscan", *weighted_append
    }
    selected = filter_catalog(df, families=["spectral"], include_external=True)
    assert {"stm", "tritopic", "fast_tritopic", "umap_spectral"} <= set(selected["catalog_id"])
    assert "append_umap_mv_spectral" not in selected["catalog_id"]
    assert filter_catalog(df, priority="all").height == df.height
    assert set(filter_catalog(df, baselines=["pca_k_means"])["catalog_id"]) == {
        "pca_k_means", "pca_mv_k_means", "pca_mv_spherical_k_means"
    }


def test_historical_names_and_unknowns(catalog):
    assert resolve_model_id("stemmed_mv_spectral_info0_2_seed36201624", catalog) == "mv_spectral_info0"
    assert resolve_model_id("umap_mv_hdbscan_1", catalog) == "mv_hdbscan"
    weighted_run = "stemmed_append_umap_w010_3_seed36201624"
    assert resolve_model_id(weighted_run, catalog) == "append_umap_w010"
    df = annotate_models(pl.DataFrame({"model_id": ["future_model", None, "baseline"]}), catalog)
    assert filter_catalog(df, priority="unclassified").height == 2
    assert filter_catalog(df, priority="all").height == 3
    assert annotate_models(pl.DataFrame(), catalog).is_empty()


def test_active_configs_are_classified(catalog):
    experiments = scan_experiment_configs(Path(__file__).resolve().parents[1] / "experiments")
    identities = configuration_models(experiments)
    assert identities
    assert all(resolve_model_id(mid, catalog) for mid in identities.values())


@pytest.mark.parametrize("mutation", ["reference", "alias", "priority", "field"])
def test_invalid_catalog_rejected(tmp_path, mutation):
    data = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    entry = data["models"]["mv_spectral"]
    if mutation == "reference":
        entry["baseline_id"] = "mv_spectral"
    elif mutation == "alias":
        entry["aliases"] = ["baseline"]
    elif mutation == "priority":
        entry["priority"] = "important"
    else:
        entry["typo"] = True
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_catalog(path)


def test_duplicate_yaml_keys_rejected(tmp_path):
    path = tmp_path / "catalog.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\nmodels: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_catalog(path)


def test_kmeans_exclusions_use_actual_algorithm_names():
    df = pl.DataFrame({"clustering_algo": ["kmeans", "k_means", "multi_view_k_means",
                                          "multi_view_spherical_k_means", "spectral", None]})
    assert set(kmeans_algorithms(df)) == {
        "kmeans", "k_means", "multi_view_k_means", "multi_view_spherical_k_means"
    }


def test_coverage_classifies_unrun_configs_and_orders_baseline_first(tmp_path, catalog):
    experiments = []
    for mid in ["pca_mv_k_means", "pca_k_means", "append_umap_mv_k_means"]:
        path = tmp_path / f"{mid}.yaml"
        path.write_text(yaml.safe_dump({"model": {"id": mid}}), encoding="utf-8")
        experiments.append({"file_path": str(path), "dataset_label": "fed",
                            "canonical_name": f"fed_standard_{mid}"})
    matrix = pl.DataFrame({
        "dataset_label": ["fed"] * 3,
        "experiment_name": [exp["canonical_name"] for exp in experiments],
        "completed_count": [0] * 3,
    })
    scoped = filter_catalog(annotate_coverage(matrix, experiments, catalog))
    assert sort_catalog(scoped)["catalog_id"].to_list() == ["pca_k_means", "pca_mv_k_means"]
    assert scoped["completed_count"].sum() == 0


def test_dashboard_cache_tracks_scope_and_retains_pca_kmeans():
    from scripts.dashboard import get_cached_best_models

    df = pl.DataFrame({
        "dataset_name": ["fed", "fed"],
        "model_name": ["baseline_1", "pca_k_means_1"],
        "clustering_algo": ["hdbscan", "k_means"],
        "dim_red_algo": ["umap", "pca"],
        "c_v": [0.5, 0.7],
    })
    kwargs = dict(dataset="fed", condition="all", exclude_clustering=None,
                  exclude_dim_red=None, dump=False, average=False,
                  merge_info0=False, suppress_nulls=False)
    get_cached_best_models.clear()
    both = get_cached_best_models(df=df, **kwargs)
    assert both["c_v"].height == 2
    one = get_cached_best_models(df=df.head(1), **kwargs)
    assert one["c_v"].height == 1
