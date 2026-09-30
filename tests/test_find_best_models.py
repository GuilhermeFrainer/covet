import sys
from pathlib import Path

import polars as pl

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.results_analysis import extract_model_type, find_best_models


def test_extract_model_type():
    assert extract_model_type("baseline_1") == "baseline"
    assert extract_model_type("mv_co_reg_spectral_1") == "mv_co_reg_spectral"
    assert extract_model_type("pca_mv_spectral_info0_10") == "pca_mv_spectral_info0"
    assert extract_model_type("vanilla") == "vanilla"


def test_find_best_models():
    data = {
        "model_name": ["baseline_1", "baseline_2", "mv_1", "mv_2"],
        "dataset_name": ["fed", "fed", "fed", "fed"],
        "c_v": [0.5, 0.7, 0.6, 0.8],
        "u_mass": [-1.5, -1.2, -1.4, -1.1],
        "n_topics": [10, 20, 10, 20],
    }
    df = pl.DataFrame(data)

    results = find_best_models(df, "fed")

    assert "c_v" in results
    assert "u_mass" in results

    cv_results = results["c_v"]
    assert cv_results.height == 2

    assert cv_results.filter(pl.col("model_type") == "baseline")["max_value"][0] == 0.7
    assert (
        cv_results.filter(pl.col("model_type") == "baseline")["best_model_name"][0]
        == "baseline_2"
    )

    assert cv_results.filter(pl.col("model_type") == "mv")["max_value"][0] == 0.8
    assert (
        cv_results.filter(pl.col("model_type") == "mv")["best_model_name"][0] == "mv_2"
    )


def test_find_best_models_filtering():
    data = {
        "model_name": ["baseline_1", "baseline_2"],
        "dataset_name": ["fed", "yelp"],
        "c_v": [0.5, 0.9],
    }
    df = pl.DataFrame(data)

    results_fed = find_best_models(df, "fed")
    assert (
        results_fed["c_v"].filter(pl.col("model_type") == "baseline")["max_value"][0]
        == 0.5
    )

    results_yelp = find_best_models(df, "yelp")
    assert (
        results_yelp["c_v"].filter(pl.col("model_type") == "baseline")["max_value"][0]
        == 0.9
    )


def test_find_best_models_exclusion():
    data = {
        "model_name": ["pca_mv_1", "umap_mv_1", "k_means_1", "spectral_1"],
        "dataset_name": ["fed", "fed", "fed", "fed"],
        "clustering_algo": ["mv", "mv", "k_means", "spectral"],
        "dim_red_algo": ["pca", "umap", "umap", "umap"],
        "c_v": [0.5, 0.6, 0.7, 0.8],
    }
    df = pl.DataFrame(data)

    # Exclude PCA
    res_no_pca = find_best_models(df, "fed", exclude_dim_red=["pca"])
    # Should only have umap_mv_1, k_means_1, spectral_1
    all_best_names = []
    for m_df in res_no_pca.values():
        all_best_names.extend(m_df["best_model_name"].to_list())
    assert "pca_mv_1" not in all_best_names

    # Exclude K-Means
    res_no_kmeans = find_best_models(df, "fed", exclude_clustering=["k_means"])
    all_best_names = []
    for m_df in res_no_kmeans.values():
        all_best_names.extend(m_df["best_model_name"].to_list())
    assert "k_means_1" not in all_best_names


def test_find_best_models_dump():
    data = {
        "model_name": ["mv_1", "mv_2"],
        "dataset_name": ["fed", "fed"],
        "c_v": [0.5, 0.6],
    }
    df = pl.DataFrame(data)

    # Normal mode (only best per type)
    res_normal = find_best_models(df, "fed", dump=False)
    assert res_normal["c_v"].height == 1
    assert res_normal["c_v"]["max_value"][0] == 0.6

    # Dump mode (all models)
    res_dump = find_best_models(df, "fed", dump=True)
    assert res_dump["c_v"].height == 2
    assert 0.5 in res_dump["c_v"]["max_value"].to_list()
    assert 0.6 in res_dump["c_v"]["max_value"].to_list()


def test_find_best_models_nan_handling():
    data = {
        "model_name": ["mv_1", "mv_2", "mv_3"],
        "dataset_name": ["fed", "fed", "fed"],
        "c_v": [0.5, float("nan"), 0.6],
        "u_mass": [float("nan"), float("nan"), -1.0],
    }
    df = pl.DataFrame(data)

    results = find_best_models(df, "fed")

    # For c_v, mv_3 (0.6) should be better than mv_1 (0.5), and NaN should be ignored
    assert results["c_v"]["max_value"][0] == 0.6
    assert results["c_v"]["best_model_name"][0] == "mv_3"

    # For u_mass, only mv_3 has a valid value
    assert results["u_mass"].height == 1
    assert results["u_mass"]["max_value"][0] == -1.0
    assert results["u_mass"]["best_model_name"][0] == "mv_3"


def test_find_best_models_stemmed_normalization():
    data = {
        "model_name": ["stemmed_baseline_1", "stemmed_mv_1"],
        "dataset_name": ["anes_stemmed", "anes_stemmed"],
        "c_v": [0.7, 0.8],
    }
    df = pl.DataFrame(data)

    # Calling with "anes" should find "anes_stemmed" data and normalize it
    results = find_best_models(df, "anes")

    assert "c_v" in results
    cv_results = results["c_v"]

    # Model types should be stripped of "stemmed_"
    assert "baseline" in cv_results["model_type"].to_list()
    assert "mv" in cv_results["model_type"].to_list()

    # Best model names should be stripped of "stemmed_"
    assert "baseline_1" in cv_results["best_model_name"].to_list()
    assert "mv_1" in cv_results["best_model_name"].to_list()

    # Check extract_model_type directly
    assert extract_model_type("stemmed_baseline_1") == "baseline"


def test_find_best_models_keeps_samples_apart_except_yelp():
    df = pl.DataFrame(
        {
            "model_name": ["baseline_1", "baseline_2", "baseline_3"],
            "dataset_name": ["trump", "trump_s25000", "yelp_s10000"],
            "c_v": [0.5, 0.9, 0.7],
        }
    )

    trump_best = find_best_models(df, "trump")["c_v"]
    assert trump_best["best_model_name"].to_list() == ["baseline_1"]

    sample_best = find_best_models(df, "trump_s25000")["c_v"]
    assert sample_best["best_model_name"].to_list() == ["baseline_2"]

    yelp_best = find_best_models(df, "yelp")["c_v"]
    assert yelp_best["best_model_name"].to_list() == ["baseline_3"]


def test_find_best_models_average_std():
    data = {
        "model_name": ["baseline_seed1", "baseline_seed2", "baseline_seed3"],
        "dataset_name": ["fed", "fed", "fed"],
        "c_v": [0.5, 0.6, 0.7],
        "random_state": [1, 2, 3],
    }
    df = pl.DataFrame(data)

    results = find_best_models(df, "fed", average=True)
    assert "c_v" in results
    cv_df = results["c_v"]
    assert cv_df.height == 1
    assert abs(cv_df["max_value"][0] - 0.6) < 1e-5
    assert cv_df["n_seeds"][0] == 3
    assert cv_df["std_value"][0] > 0.0


def test_generate_best_models_latex_table_std():
    from src.make_table import generate_best_models_latex_table

    data = {
        "model_name": ["baseline_seed1", "baseline_seed2"],
        "dataset_name": ["fed", "fed"],
        "c_v": [0.5, 0.7],
        "random_state": [1, 2],
    }
    df = pl.DataFrame(data)
    results = find_best_models(df, "fed", average=True)

    latex_str = generate_best_models_latex_table(results, "fed", average=True)
    assert "\\begin{table}" in latex_str
    assert "\\pm" in latex_str
    assert "mean" in latex_str or "std" in latex_str
