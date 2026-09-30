from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import scipy.stats as scipy_stats

METRICS = ["u_mass", "c_v", "c_npmi", "irbo", "topic_diversity"]

# Dataset names that report under another name. Yelp's 10k sample is the only
# sampled dataset reported as its full dataset; every other sample (e.g.
# trump_s25000) keeps its suffix so it never pools with full-corpus results.
DATASET_ALIASES = {
    "anes_stemmed": "anes",
    "yelp_s10000": "yelp",
}


def canonical_dataset_expr(expr: pl.Expr) -> pl.Expr:
    """Maps a dataset-name expression onto its reporting name.

    Args:
        expr: A Polars string expression holding dataset names.

    Returns:
        The expression with aliased dataset names replaced.
    """
    return expr.replace(DATASET_ALIASES)


def extract_model_type(model_name: str, merge_info0: bool = False) -> str:
    """Extracts the base model type from a model name.

    Example: 'baseline_1' -> 'baseline', 'baseline_seed36201624' -> 'baseline'.
    """
    if not isinstance(model_name, str):
        return str(model_name)

    res = model_name
    # Strip common prefixes
    if res.startswith("stemmed_"):
        res = res[len("stemmed_") :]

    if res.startswith("stm_"):
        return "stm"

    # Remove seed suffixes if present (e.g., _seed36201624)
    if "_seed" in res:
        res = res.split("_seed")[0]

    # Remove any trailing numbers (e.g., baseline_1 -> baseline)
    if "_" in res:
        parts = res.split("_")
        if parts[-1].isdigit():
            res = "_".join(parts[:-1])

    # Then optionally strip info0
    if merge_info0 and res.endswith("_info0"):
        res = res[: -len("_info0")]

    return res


def find_best_models(
    df: pl.DataFrame,
    dataset: str,
    exclude_clustering: List[str] | None = None,
    exclude_dim_red: List[str] | None = None,
    dump: bool = False,
    average: bool = False,
    merge_info0: bool = False,
    suppress_nulls: bool = False,
) -> Dict[str, pl.DataFrame]:
    """
    Finds the best performing model (or average performance with standard deviation)
    of each model type for each metric in the given dataset.

    Args:
        df: Polars DataFrame containing experiment results.
        dataset: The name of the dataset to filter by.
        exclude_clustering: Optional list of clustering algorithms to exclude.
        exclude_dim_red: Optional list of dimensionality reduction
            algorithms to exclude.
        dump: If True, returns all model configurations instead of
            just the best per type.
        average: If True, calculates the average performance and standard deviation
            per model type across runs/seeds.
        merge_info0: If True, treats models with and without _info0 as the same type.
        suppress_nulls: If True, filters out rows with NaNs in any metric column.

    Returns:
        A dictionary where keys are metric names and values are DataFrames
        with models and their scores (including max_value, std_value, n_seeds).
    """
    # Normalize dataset name and model names
    if "dataset_name" in df.columns:
        df = df.with_columns(canonical_dataset_expr(pl.col("dataset_name")))

    if "model_name" in df.columns:
        df = df.with_columns(pl.col("model_name").str.replace("^stemmed_", ""))

    # Filter by dataset
    if "dataset_name" in df.columns:
        df = df.filter(pl.col("dataset_name") == dataset)

    if df.is_empty():
        return {}

    # Optional: Suppress rows with any NaNs in metric columns
    if suppress_nulls:
        actual_metric_cols = [m for m in METRICS if m in df.columns]
        if actual_metric_cols:
            # Drop rows where any metric column has a NaN/Null
            # We cast to float to ensure uniform check
            null_mask = pl.any_horizontal(
                pl.col(actual_metric_cols).cast(pl.Float64, strict=False).is_null()
                | pl.col(actual_metric_cols).cast(pl.Float64, strict=False).is_nan()
            )
            df = df.filter(~null_mask)

    if df.is_empty():
        return {}

    # Apply exclusion filters
    # Default: exclude PCA dim red and any k_means variant if not explicitly provided
    if "clustering_algo" in df.columns:
        if exclude_clustering is not None:
            df = df.filter(~pl.col("clustering_algo").is_in(exclude_clustering))
        else:
            df = df.filter(~pl.col("clustering_algo").str.contains("k_means"))

    if "dim_red_algo" in df.columns:
        if exclude_dim_red is not None:
            df = df.filter(~pl.col("dim_red_algo").is_in(exclude_dim_red))
        else:
            df = df.filter(pl.col("dim_red_algo") != "pca")

    if df.is_empty():
        return {}

    # Add or update model_type column based on model_name and merge_info0
    if "model_name" in df.columns:
        df = df.with_columns(
            pl.col("model_name")
            .map_elements(
                lambda x: extract_model_type(x, merge_info0=merge_info0),
                return_dtype=pl.String,
            )
            .alias("model_type")
        )

    available_metrics = [m for m in METRICS if m in df.columns]

    results = {}
    for metric in available_metrics:
        # Ensure metric is numeric and filter out nulls/NaNs
        if df[metric].dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            continue

        # filter out nulls and NaNs
        metric_df = df.filter(pl.col(metric).is_not_null())
        if metric_df[metric].dtype in [pl.Float32, pl.Float64]:
            metric_df = metric_df.filter(~pl.col(metric).is_nan())

        if metric_df.is_empty():
            continue

        if dump:
            # For dump mode, just sort by the metric and keep everything
            best_per_type = metric_df.sort(metric, descending=True).select(
                [
                    pl.col("model_name").alias("best_model_name"),
                    pl.col(metric).alias("max_value"),
                    pl.lit(0.0).alias("std_value"),
                    pl.lit(1).alias("n_seeds"),
                    pl.col("model_type"),
                ]
            )
        elif average:
            # For average mode, group by model_type and calculate mean, std, n_seeds
            best_per_type = (
                metric_df.group_by("model_type")
                .agg(
                    pl.col(metric).mean().alias("max_value"),
                    pl.col(metric).std().fill_null(0.0).alias("std_value"),
                    pl.col(metric).count().alias("n_seeds"),
                    pl.col("model_type").first().alias("best_model_name"),
                )
                .sort("max_value", descending=True)
            )
        else:
            # For each metric, group by model_type and find the max
            best_per_type = (
                metric_df.sort(metric, descending=True)
                .group_by("model_type")
                .agg(
                    pl.col(metric).first().alias("max_value"),
                    pl.lit(0.0).alias("std_value"),
                    pl.lit(1).alias("n_seeds"),
                    pl.col("model_name").first().alias("best_model_name"),
                )
                .sort("max_value", descending=True)
            )
        results[metric] = best_per_type

    return results


def calculate_hdbscan_noise_coverage(
    df: pl.DataFrame,
    dataset: str | None = None,
    group_by_model_type: bool = True,
    merge_info0: bool = False,
) -> pl.DataFrame:
    """Calculates noise-cluster coverage for models using HDBSCAN clustering.

    Noise-cluster coverage measures the proportion/percentage of documents assigned
    to the noise cluster (outlier topic -1) relative to total document observations.

    Args:
        df: Polars DataFrame containing experiment results.
        dataset: Optional dataset name to filter by.
        group_by_model_type: If True, aggregates across seeds per model type.
        merge_info0: If True, treats models with and without _info0 as same type.

    Returns:
        Polars DataFrame containing noise coverage metrics.
    """
    if df.is_empty():
        return pl.DataFrame()

    # Filter for HDBSCAN clustering algorithms (single-view, multi-view, and stacked)
    if "clustering_algo" in df.columns:
        filtered_df = df.filter(
            pl.col("clustering_algo").is_in(
                [
                    "hdbscan",
                    "mv_hdbscan",
                    "multi_view_hdbscan",
                    "feature_stacking_hdbscan",
                    "stacked_hdbscan",
                ]
            )
            | pl.col("clustering_algo").str.contains("(?i)hdbscan")
        )
    else:
        filtered_df = df

    if dataset and "dataset_name" in filtered_df.columns:
        filtered_df = filtered_df.filter(pl.col("dataset_name") == dataset)

    if filtered_df.is_empty():
        return pl.DataFrame()

    # Ensure required columns exist
    if (
        "outliers" not in filtered_df.columns
        or "n_observations" not in filtered_df.columns
    ):
        raise ValueError(
            "DataFrame must contain 'outliers' and 'n_observations' columns."
        )

    # Calculate individual noise coverage ratio and percentage
    filtered_df = filtered_df.with_columns(
        (pl.col("outliers") / pl.col("n_observations")).alias("noise_ratio"),
        ((pl.col("outliers") / pl.col("n_observations")) * 100.0).alias(
            "noise_coverage_pct"
        ),
        ((1.0 - (pl.col("outliers") / pl.col("n_observations"))) * 100.0).alias(
            "clustered_coverage_pct"
        ),
    )

    if not group_by_model_type:
        return filtered_df

    # Extract model_type if needed
    if "model_name" in filtered_df.columns:
        filtered_df = filtered_df.with_columns(
            pl.col("model_name")
            .map_elements(
                lambda x: extract_model_type(x, merge_info0=merge_info0),
                return_dtype=pl.String,
            )
            .alias("model_type")
        )

    group_cols = ["model_type"]
    if "dataset_name" in filtered_df.columns:
        group_cols.insert(0, "dataset_name")

    aggregated = (
        filtered_df.group_by(group_cols)
        .agg(
            pl.len().alias("n_runs"),
            pl.col("n_observations").first().alias("n_observations"),
            pl.col("outliers").mean().round(2).alias("outliers_mean"),
            pl.col("outliers").std().fill_null(0.0).round(2).alias("outliers_std"),
            pl.col("noise_coverage_pct")
            .mean()
            .round(2)
            .alias("noise_coverage_pct_mean"),
            pl.col("noise_coverage_pct")
            .std()
            .fill_null(0.0)
            .round(2)
            .alias("noise_coverage_pct_std"),
            pl.col("noise_coverage_pct").min().round(2).alias("noise_coverage_pct_min"),
            pl.col("noise_coverage_pct").max().round(2).alias("noise_coverage_pct_max"),
            pl.col("clustered_coverage_pct")
            .mean()
            .round(2)
            .alias("clustered_coverage_pct_mean"),
        )
        .sort(group_cols)
    )

    return aggregated


def compute_stopword_impact(
    df_remove_rep_stopwords: pl.DataFrame,
    df_keep_rep_stopwords: pl.DataFrame,
    dataset: str = "fed",
    exclude_clustering: List[str] | None = None,
    exclude_dim_red: List[str] | None = None,
    merge_info0: bool = False,
) -> Dict[str, pl.DataFrame]:
    """Computes metric differences between remove and keep stopwords runs.

    Calculates pairwise difference (remove_rep_stopwords - keep_rep_stopwords)
    for corresponding models (same configuration and seed) and aggregates
    by model_type.

    Args:
        df_remove_rep_stopwords: Polars DataFrame of results with representation
            stopwords removed.
        df_keep_rep_stopwords: Polars DataFrame of results with representation
            stopwords kept.
        dataset: Name of dataset to filter by.
        exclude_clustering: Optional list of clustering algorithms to exclude.
        exclude_dim_red: Optional list of dim reduction algorithms to exclude.
        merge_info0: If True, treats _info0 model variants as their base model type.

    Returns:
        A dictionary mapping each metric to a Polars DataFrame with columns:
        ["model_type", "mean_delta", "std_delta", "n_pairs"].
    """
    df_standard = df_remove_rep_stopwords
    df_no_stopword = df_keep_rep_stopwords

    if df_standard.is_empty() or df_no_stopword.is_empty():
        return {}

    # Standardize dataset column if present
    for df in [df_standard, df_no_stopword]:
        if "dataset_name" in df.columns:
            df = df.with_columns(canonical_dataset_expr(pl.col("dataset_name")))

    if "dataset_name" in df_standard.columns:
        df_standard = df_standard.filter(pl.col("dataset_name") == dataset)
    if "dataset_name" in df_no_stopword.columns:
        df_no_stopword = df_no_stopword.filter(pl.col("dataset_name") == dataset)

    if df_standard.is_empty() or df_no_stopword.is_empty():
        return {}

    # Strip stemmed_ prefix
    if "model_name" in df_standard.columns:
        df_standard = df_standard.with_columns(
            pl.col("model_name").str.replace("^stemmed_", "")
        )
    if "model_name" in df_no_stopword.columns:
        df_no_stopword = df_no_stopword.with_columns(
            pl.col("model_name").str.replace("^stemmed_", "")
        )

    # Filter clustering and dim reduction algorithms
    if "clustering_algo" in df_standard.columns:
        if exclude_clustering is not None:
            df_standard = df_standard.filter(
                ~pl.col("clustering_algo").is_in(exclude_clustering)
            )
        else:
            df_standard = df_standard.filter(
                ~pl.col("clustering_algo").str.contains("k_means")
            )
    if "clustering_algo" in df_no_stopword.columns:
        if exclude_clustering is not None:
            df_no_stopword = df_no_stopword.filter(
                ~pl.col("clustering_algo").is_in(exclude_clustering)
            )
        else:
            df_no_stopword = df_no_stopword.filter(
                ~pl.col("clustering_algo").str.contains("k_means")
            )

    if "dim_red_algo" in df_standard.columns:
        if exclude_dim_red is not None:
            df_standard = df_standard.filter(
                ~pl.col("dim_red_algo").is_in(exclude_dim_red)
            )
        else:
            df_standard = df_standard.filter(pl.col("dim_red_algo") != "pca")
    if "dim_red_algo" in df_no_stopword.columns:
        if exclude_dim_red is not None:
            df_no_stopword = df_no_stopword.filter(
                ~pl.col("dim_red_algo").is_in(exclude_dim_red)
            )
        else:
            df_no_stopword = df_no_stopword.filter(pl.col("dim_red_algo") != "pca")

    if df_standard.is_empty() or df_no_stopword.is_empty():
        return {}

    # Join standard and no_stopword dataframes
    if "model_name" in df_standard.columns and "model_name" in df_no_stopword.columns:
        # Deduplicate before join to prevent cartesian products
        std_unique = df_standard.unique(subset=["model_name"])
        no_unique = df_no_stopword.unique(subset=["model_name"])
        joined = std_unique.join(no_unique, on="model_name", suffix="_no_stopword")
    else:
        # Fallback to join on random_state and n_topics if available
        join_cols = [
            c
            for c in ["random_state", "n_topics", "clustering_algo", "dim_red_algo"]
            if c in df_standard.columns and c in df_no_stopword.columns
        ]
        if not join_cols:
            return {}
        joined = df_standard.join(df_no_stopword, on=join_cols, suffix="_no_stopword")

    if joined.is_empty():
        return {}

    # Map model_type
    joined = joined.with_columns(
        pl.col("model_name")
        .map_elements(
            lambda x: extract_model_type(x, merge_info0=merge_info0),
            return_dtype=pl.String,
        )
        .alias("model_type")
    )

    available_metrics = [m for m in METRICS if m in joined.columns]
    results = {}

    for metric in available_metrics:
        ns_col = f"{metric}_no_stopword"
        if ns_col not in joined.columns:
            continue

        metric_df = joined.filter(
            pl.col(metric).is_not_null() & pl.col(ns_col).is_not_null()
        )
        if metric_df[metric].dtype in [pl.Float32, pl.Float64]:
            metric_df = metric_df.filter(
                ~pl.col(metric).is_nan() & ~pl.col(ns_col).is_nan()
            )

        if metric_df.is_empty():
            continue

        metric_df = metric_df.with_columns(
            (pl.col(metric) - pl.col(ns_col)).alias("delta")
        )

        aggregated = (
            metric_df.group_by("model_type")
            .agg(
                pl.col("delta").mean().alias("mean_delta"),
                pl.col("delta").std().fill_null(0.0).alias("std_delta"),
                pl.col("delta").count().alias("n_pairs"),
                pl.col("model_type").first().alias("best_model_name"),
            )
            .sort("mean_delta", descending=True)
        )
        results[metric] = aggregated

    return results


def parse_model_type_and_topic(
    model_name: str, merge_info0: bool = False
) -> Tuple[str, int]:
    """Extracts base model type and topic count index from a model run name.

    Examples:
        'baseline_1_seed36201624' -> ('baseline', 1)
        'stemmed_mv_spectral_3_seed100' -> ('mv_spectral', 3)
        'append_umap_5' -> ('append_umap', 5)

    Args:
        model_name: The raw experiment model name.
        merge_info0: Whether to strip trailing '_info0'.

    Returns:
        Tuple of (model_type, topic_index).
    """
    if not isinstance(model_name, str):
        return str(model_name), 0

    clean = model_name
    if clean.startswith("stemmed_"):
        clean = clean[len("stemmed_") :]
    if clean.startswith("stm_"):
        # Handle stm naming if needed
        pass
    if "_seed" in clean:
        clean = clean.split("_seed")[0]

    parts = clean.split("_")
    if parts[-1].isdigit():
        topic_idx = int(parts[-1])
        base = "_".join(parts[:-1])
    else:
        topic_idx = 0
        base = clean

    if merge_info0 and base.endswith("_info0"):
        base = base[: -len("_info0")]

    return base, topic_idx


def wilcoxon_exact_test(
    differences: Sequence[float] | np.ndarray,
    alternative: str = "two-sided",
    zero_method: str = "pratt",
) -> Tuple[float, float]:
    """Computes exact paired Wilcoxon signed-rank test statistic and p-value.

    Following Demšar (2006), uses the exact permutation distribution for small
    sample sizes (such as N=5 topic counts) and applies Pratt's method for handling
    zero differences without discarding paired blocks.

    Args:
        differences: Paired differences (Alternative - Default) across sample blocks.
        alternative: Test direction ('two-sided', 'greater', or 'less').
        zero_method: Zero handling method ('pratt', 'wilcoxon', or 'zsplit').

    Returns:
        Tuple of (statistic, p_value).
    """
    diff_arr = np.asarray(differences, dtype=float)
    if len(diff_arr) == 0:
        return 0.0, 1.0

    # If all differences are zero, there is no variance and no difference
    if np.all(diff_arr == 0):
        return 0.0, 1.0

    try:
        res = scipy_stats.wilcoxon(
            diff_arr,
            zero_method=zero_method,
            method="exact",
            alternative=alternative,
        )
        p_val = float(res.pvalue)
        if np.isnan(p_val):
            p_val = 1.0
        return float(res.statistic), p_val
    except Exception:
        # Fallback for degenerate distributions
        return 0.0, 1.0


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down adjustment to control FWER.

    Controls family-wise error rate across a family of hypotheses:
    p_adj_(i) = min(1.0, max_{j <= i} ((m - j + 1) * p_(j)))

    Args:
        p_values: Sequence of raw p-values.

    Returns:
        List of adjusted p-values in original input order.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        multiplier = n - rank
        adj = min(1.0, multiplier * p)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max

    return adjusted


def compute_demsar_delta_table(
    df_default: pl.DataFrame,
    df_alternative: pl.DataFrame,
    datasets: Optional[Sequence[str] | str] = None,
    dataset: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    alpha: float = 0.10,
    alternative: str = "two-sided",
    zero_method: str = "pratt",
    correction: str = "per_metric",
    exclude_clustering: Optional[List[str]] = None,
    exclude_dim_red: Optional[List[str]] = None,
    merge_info0: bool = False,
) -> Dict[str, Any]:
    """Computes Demšar-compliant Model-by-Metric Delta Table and statistical tests.

    Following Demšar (2006) recommendations for comparing two conditions/algorithms,
    each benchmark dataset is treated as an independent evaluation block. For each
    (dataset, model), scores are aggregated over random seeds and topic counts. Paired
    differences are then evaluated using the exact Wilcoxon signed-rank test across
    the N datasets.

    Workflow:
    1. Dataset Normalization & Filtering: Cleans dataset identifiers and filters
       by requested datasets.
    2. Model Type Annotation: Extracts base model types.
    3. Seed and Topic Count Aggregation: Averages scores across random seeds and
       topic counts for each (dataset_name, model_type).
    4. Delta Computation: Computes paired differences across N datasets.
    5. Non-Parametric Significance: Runs paired exact Wilcoxon signed-rank test
       for each (model, metric) across the N datasets.
    6. FWER Control: Applies Holm-Bonferroni step-down correction.
    7. Formatting: Creates structured delta tables with significance flags.

    Args:
        df_default: Polars DataFrame of baseline/default runs.
        df_alternative: Polars DataFrame of alternative configuration runs.
        datasets: Dataset identifier(s) to filter by (or None / "all" for all).
        dataset: Backwards-compatible alias for datasets.
        metrics: List of metrics to evaluate (default: METRICS).
        alpha: Significance threshold (e.g., 0.10 for two-tailed or 0.05
            for one-tailed).
        alternative: Test direction ('two-sided', 'greater', or 'less').
        zero_method: Zero handling method for Wilcoxon ('pratt' or 'wilcoxon').
        correction: FWER correction mode ('per_metric', 'table', or 'none').
        exclude_clustering: Clustering algorithms to exclude.
        exclude_dim_red: Dimensionality reduction algorithms to exclude.
        merge_info0: Treat '_info0' variants as base model type.

    Returns:
        Dictionary containing:
        - 'df_summary': Polars DataFrame with formatted strings per cell
        - 'df_details': Polars DataFrame with granular statistics (delta, std,
          w_stat, p_raw, p_adj, sig)
        - 'models': List of model types
        - 'metrics': List of evaluated metrics
        - 'alpha': Alpha threshold used
        - 'alternative': Test direction used
        - 'correction': Correction method used
        - 'n_datasets': Number of paired dataset blocks (N)
        - 'n_blocks': Number of paired dataset blocks (N)
        - 'n_topic_blocks': Backwards-compatible alias for n_blocks
        - 'datasets': List of dataset names evaluated
    """
    if df_default.is_empty() or df_alternative.is_empty():
        return {
            "df_summary": pl.DataFrame(),
            "df_details": pl.DataFrame(),
            "models": [],
            "metrics": [],
            "alpha": alpha,
            "alternative": alternative,
            "correction": correction,
            "n_datasets": 0,
            "n_blocks": 0,
            "n_topic_blocks": 0,
            "datasets": [],
        }

    target_metrics = [m for m in (metrics or METRICS)]

    # 1. Dataset normalization and filtering
    std_df = df_default.clone()
    alt_df = df_alternative.clone()

    for idx, df_ref in enumerate([std_df, alt_df]):
        if "dataset_name" in df_ref.columns:
            cleaned_ds = df_ref.select(
                canonical_dataset_expr(pl.col("dataset_name"))
            ).to_series()
            if idx == 0:
                std_df = std_df.with_columns(cleaned_ds.alias("dataset_name"))
            else:
                alt_df = alt_df.with_columns(cleaned_ds.alias("dataset_name"))
        else:
            if idx == 0:
                std_df = std_df.with_columns(pl.lit("dataset").alias("dataset_name"))
            else:
                alt_df = alt_df.with_columns(pl.lit("dataset").alias("dataset_name"))

    target_ds = datasets if datasets is not None else dataset
    if target_ds is not None and target_ds != "all":
        if isinstance(target_ds, str):
            std_df = std_df.filter(pl.col("dataset_name") == target_ds)
            alt_df = alt_df.filter(pl.col("dataset_name") == target_ds)
        else:
            std_df = std_df.filter(pl.col("dataset_name").is_in(list(target_ds)))
            alt_df = alt_df.filter(pl.col("dataset_name").is_in(list(target_ds)))

    if std_df.is_empty() or alt_df.is_empty():
        return {
            "df_summary": pl.DataFrame(),
            "df_details": pl.DataFrame(),
            "models": [],
            "metrics": target_metrics,
            "alpha": alpha,
            "alternative": alternative,
            "correction": correction,
            "n_datasets": 0,
            "n_blocks": 0,
            "n_topic_blocks": 0,
            "datasets": [],
        }

    # 2. Extract model_type and topic_idx
    def annotate_types_and_topics(df: pl.DataFrame) -> pl.DataFrame:
        m_types = []
        t_indices = []
        model_names = df["model_name"].to_list() if "model_name" in df.columns else []
        for name in model_names:
            mtype, tidx = parse_model_type_and_topic(str(name), merge_info0=merge_info0)
            m_types.append(mtype)
            t_indices.append(tidx)

        annotated = df.with_columns(
            [
                pl.Series("model_type", m_types),
                pl.Series("topic_idx", t_indices),
            ]
        )

        # Fallback for topic_idx if parsed as 0 but n_topics / nr_topics exists
        if "nr_topics" in annotated.columns:
            annotated = annotated.with_columns(
                pl.when(pl.col("topic_idx") == 0)
                .then(pl.col("nr_topics").fill_null(0))
                .otherwise(pl.col("topic_idx"))
                .alias("topic_idx")
            )
        elif "n_topics" in annotated.columns:
            annotated = annotated.with_columns(
                pl.when(pl.col("topic_idx") == 0)
                .then(pl.col("n_topics").fill_null(0))
                .otherwise(pl.col("topic_idx"))
                .alias("topic_idx")
            )

        return annotated

    std_df = annotate_types_and_topics(std_df)
    alt_df = annotate_types_and_topics(alt_df)

    # 3. Apply exclusion filters with normalized matching
    def filter_algos(
        df: pl.DataFrame,
        col_name: str,
        exclusions: Optional[List[str]],
        default_pattern: Optional[str],
    ) -> pl.DataFrame:
        if col_name not in df.columns:
            return df
        if exclusions is not None:
            norm_exclusions = [e.lower().replace("_", "") for e in exclusions]
            # Match if normalized value is in exclusions or contains any excluded token
            cond = (
                df[col_name]
                .fill_null("")
                .map_elements(
                    lambda x: any(
                        ex in x.lower().replace("_", "")
                        or x.lower().replace("_", "") == ex
                        for ex in norm_exclusions
                    ),
                    return_dtype=pl.Boolean,
                )
            )
            return df.filter(~cond)
        elif default_pattern:
            return df.filter(~pl.col(col_name).str.contains(default_pattern))
        return df

    std_df = filter_algos(std_df, "clustering_algo", exclude_clustering, "k_means")
    alt_df = filter_algos(alt_df, "clustering_algo", exclude_clustering, "k_means")

    std_df = filter_algos(std_df, "dim_red_algo", exclude_dim_red, "pca")
    alt_df = filter_algos(alt_df, "dim_red_algo", exclude_dim_red, "pca")

    if std_df.is_empty() or alt_df.is_empty():
        return {
            "df_summary": pl.DataFrame(),
            "df_details": pl.DataFrame(),
            "models": [],
            "metrics": target_metrics,
            "alpha": alpha,
            "alternative": alternative,
            "correction": correction,
            "n_datasets": 0,
            "n_blocks": 0,
            "n_topic_blocks": 0,
            "datasets": [],
        }

    available_metrics = [
        m for m in target_metrics if m in std_df.columns and m in alt_df.columns
    ]
    if not available_metrics:
        return {
            "df_summary": pl.DataFrame(),
            "df_details": pl.DataFrame(),
            "models": [],
            "metrics": [],
            "alpha": alpha,
            "alternative": alternative,
            "correction": correction,
            "n_datasets": 0,
            "n_blocks": 0,
            "n_topic_blocks": 0,
            "datasets": [],
        }

    # 4. Aggregation across random seeds and topic counts per (dataset_name, model_type)
    agg_exprs = [
        pl.col(m).cast(pl.Float64, strict=False).mean().alias(m)
        for m in available_metrics
    ]

    std_agg = (
        std_df.group_by(["dataset_name", "model_type"])
        .agg(agg_exprs)
        .sort(["dataset_name", "model_type"])
    )
    alt_agg = (
        alt_df.group_by(["dataset_name", "model_type"])
        .agg(agg_exprs)
        .sort(["dataset_name", "model_type"])
    )

    # 5. Join default and alternative on paired (dataset_name, model_type)
    joined = std_agg.join(
        alt_agg, on=["dataset_name", "model_type"], suffix="_alternative"
    )
    if joined.is_empty():
        return {
            "df_summary": pl.DataFrame(),
            "df_details": pl.DataFrame(),
            "models": [],
            "metrics": available_metrics,
            "alpha": alpha,
            "alternative": alternative,
            "correction": correction,
            "n_datasets": 0,
            "n_blocks": 0,
            "n_topic_blocks": 0,
            "datasets": [],
        }

    # Desired canonical model order
    canonical_order = [
        "mv_co_reg_spectral",
        "mv_co_reg_spectral_info0",
        "mv_spectral",
        "mv_spectral_info0",
        "aligned_umap",
        "append_umap",
        "baseline",
        "umap_spectral",
        "stm",
    ]

    present_models = joined["model_type"].unique().to_list()
    ordered_models = [m for m in canonical_order if m in present_models]
    ordered_models += sorted([m for m in present_models if m not in canonical_order])

    # 6. Compute deltas across datasets and exact Wilcoxon test for each (model, metric)
    records = []
    for model in ordered_models:
        model_sub = joined.filter(pl.col("model_type") == model).sort("dataset_name")
        n_blocks = len(model_sub)

        for metric in available_metrics:
            alt_col = f"{metric}_alternative"
            d_series = (model_sub[alt_col] - model_sub[metric]).drop_nulls()
            d_vals = d_series.to_numpy()

            if len(d_vals) == 0:
                mean_d = 0.0
                std_d = 0.0
                w_stat = 0.0
                p_raw = 1.0
            else:
                mean_d = float(np.mean(d_vals))
                std_d = float(np.std(d_vals))
                w_stat, p_raw = wilcoxon_exact_test(
                    d_vals,
                    alternative=alternative,
                    zero_method=zero_method,
                )

            records.append(
                {
                    "model_type": model,
                    "metric": metric,
                    "mean_delta": mean_d,
                    "std_delta": std_d,
                    "n_blocks": n_blocks,
                    "w_stat": w_stat,
                    "p_raw": p_raw,
                }
            )

    # 7. Apply FWER Adjustment
    if correction == "per_metric":
        # Group by metric and apply Holm-Bonferroni per column
        for metric in available_metrics:
            metric_indices = [i for i, r in enumerate(records) if r["metric"] == metric]
            raw_ps = [records[i]["p_raw"] for i in metric_indices]
            adj_ps = holm_bonferroni(raw_ps)
            for i, adj_p in zip(metric_indices, adj_ps):
                records[i]["p_adj"] = adj_p
    elif correction == "table":
        # Global Holm-Bonferroni across all (model, metric) pairs
        raw_ps = [r["p_raw"] for r in records]
        adj_ps = holm_bonferroni(raw_ps)
        for r, adj_p in zip(records, adj_ps):
            r["p_adj"] = adj_p
    else:
        # No adjustment
        for r in records:
            r["p_adj"] = r["p_raw"]

    # 8. Add significance flags and cell formatting
    for r in records:
        is_sig = bool(r["p_adj"] < alpha)
        r["is_significant"] = is_sig
        flag = "*" if is_sig else ""
        mean_d = r["mean_delta"]
        # Format string e.g. +0.034* or -0.012
        r["formatted"] = f"{mean_d:+.3f}{flag}"

    df_details = pl.DataFrame(records)

    # 9. Build Summary Table (Rows: Models, Columns: Metrics)
    summary_rows = []
    for model in ordered_models:
        row_dict = {"Model": model}
        for metric in available_metrics:
            match = df_details.filter(
                (pl.col("model_type") == model) & (pl.col("metric") == metric)
            )
            if not match.is_empty():
                row_dict[metric] = match["formatted"][0]
            else:
                row_dict[metric] = "N/A"
        summary_rows.append(row_dict)

    df_summary = pl.DataFrame(summary_rows)

    n_blocks_val = df_details["n_blocks"].max() if not df_details.is_empty() else 0
    return {
        "df_summary": df_summary,
        "df_details": df_details,
        "models": ordered_models,
        "metrics": available_metrics,
        "alpha": alpha,
        "alternative": alternative,
        "correction": correction,
        "n_datasets": n_blocks_val,
        "n_blocks": n_blocks_val,
        "n_topic_blocks": n_blocks_val,
        "datasets": (
            list(joined["dataset_name"].unique())
            if "dataset_name" in joined.columns
            else []
        ),
    }


def friedman_omnibus_test(
    mean_ranks: Sequence[float] | np.ndarray,
    n_blocks: int,
    k_models: int,
) -> Dict[str, float]:
    """Computes Friedman chi-squared statistic and Iman-Davenport F-statistic.

    Following Demšar (2006):
    - Friedman statistic:
      chi2_F = (12 * N / (k * (k + 1))) * (sum(R_j^2) - (k * (k + 1)^2 / 4))
    - Iman and Davenport (1987) correction:
      F_F = ((N - 1) * chi2_F) / (N * (k - 1) - chi2_F)

    Args:
        mean_ranks: Average rank for each model across the N blocks.
        n_blocks: Number of evaluation blocks (N).
        k_models: Number of algorithms (k).

    Returns:
        Dictionary with keys:
        - 'chi2_f': Friedman chi-squared statistic
        - 'p_chi2': Asymptotic chi-squared p-value
        - 'f_f': Iman-Davenport F-statistic
        - 'p_f_f': F-distribution p-value
        - 'df1': Numerator degrees of freedom (k - 1)
        - 'df2': Denominator degrees of freedom ((k - 1) * (N - 1))
        - 'n_blocks': Number of blocks (N)
        - 'k_models': Number of models (k)
    """
    if n_blocks <= 1 or k_models <= 1:
        return {
            "chi2_f": 0.0,
            "p_chi2": 1.0,
            "f_f": 0.0,
            "p_f_f": 1.0,
            "df1": max(1, k_models - 1),
            "df2": max(1, (k_models - 1) * (n_blocks - 1)),
            "n_blocks": n_blocks,
            "k_models": k_models,
        }

    ranks_arr = np.asarray(mean_ranks, dtype=float)
    k = k_models
    n = n_blocks

    sum_sq_ranks = float(np.sum(ranks_arr**2))
    term = (k * ((k + 1) ** 2)) / 4.0
    chi2_f = (12.0 * n / (k * (k + 1))) * (sum_sq_ranks - term)

    # Clean numerical inaccuracies around zero
    if chi2_f < 0.0:
        chi2_f = 0.0

    df1 = k - 1
    df2 = (k - 1) * (n - 1)
    p_chi2 = float(scipy_stats.chi2.sf(chi2_f, df1))

    denom = (n * (k - 1)) - chi2_f
    if denom <= 1e-12:
        # Extreme significance: identical ranks in every block
        f_f = float("inf")
        p_f_f = 0.0
    else:
        f_f = float(((n - 1) * chi2_f) / denom)
        p_f_f = float(scipy_stats.f.sf(f_f, df1, df2))

    return {
        "chi2_f": chi2_f,
        "p_chi2": p_chi2,
        "f_f": f_f,
        "p_f_f": p_f_f,
        "df1": df1,
        "df2": df2,
        "n_blocks": n,
        "k_models": k,
    }


def nemenyi_critical_difference(
    n_blocks: int,
    k_models: int,
    alpha: float = 0.05,
) -> float:
    """Computes Nemenyi Critical Difference (CD) for all-vs-all model comparison.

    Following Demšar (2006), CD = q_alpha * sqrt((k * (k + 1)) / (6 * N)),
    where q_alpha is the Studentized range critical value divided by sqrt(2).

    Args:
        n_blocks: Number of evaluation blocks (N).
        k_models: Number of models (k).
        alpha: Significance level (default: 0.05).

    Returns:
        The Critical Difference threshold as a float.
    """
    if n_blocks <= 0 or k_models <= 1:
        return 0.0

    # Upper alpha-quantile of studentized range distribution with k treatments, inf df
    q_alpha = float(
        scipy_stats.studentized_range.ppf(1.0 - alpha, k_models, np.inf) / np.sqrt(2)
    )
    cd = q_alpha * np.sqrt((k_models * (k_models + 1)) / (6.0 * n_blocks))
    return float(cd)


def compute_nemenyi_cliques(
    models: Sequence[str],
    mean_ranks: Dict[str, float],
    cd: float,
) -> Dict[str, str]:
    """Computes statistical equivalence cliques (letters) based on Nemenyi CD.

    Models whose rank difference is <= CD belong to the same clique.
    Assigns letters ('a', 'b', 'c', ...) following multcomp standard.

    Args:
        models: List of model names.
        mean_ranks: Dict mapping model name to mean rank.
        cd: Nemenyi Critical Difference value.

    Returns:
        Dict mapping model name to comma-separated letter codes (e.g. 'a', 'a, b').
    """
    if not models:
        return {}

    # Sort models by mean rank ascending (best to worst)
    sorted_models = sorted(models, key=lambda m: (mean_ranks.get(m, float("inf")), m))
    k = len(sorted_models)
    if k == 1:
        return {sorted_models[0]: "a"}

    # Find maximal contiguous subsets [i, j] where R_j - R_i <= CD
    valid_subsets = []
    for i in range(k):
        for j in range(i, k):
            r_diff = mean_ranks[sorted_models[j]] - mean_ranks[sorted_models[i]]
            if r_diff <= cd + 1e-9:
                valid_subsets.append((i, j))

    # Keep only maximal subsets (not strictly contained within any other subset)
    maximal_subsets = []
    for i, j in valid_subsets:
        is_maximal = True
        for oi, oj in valid_subsets:
            if (oi < i and oj >= j) or (oi <= i and oj > j):
                is_maximal = False
                break
        if is_maximal and (i, j) not in maximal_subsets:
            maximal_subsets.append((i, j))

    # Sort maximal subsets by start index
    maximal_subsets.sort(key=lambda s: (s[0], s[1]))

    # Assign letters: 'a', 'b', 'c', ...
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    model_cliques: Dict[str, List[str]] = {m: [] for m in sorted_models}

    for idx, (start, end) in enumerate(maximal_subsets):
        letter = alphabet[idx % len(alphabet)]
        if idx >= len(alphabet):
            letter = f"{letter}{idx // len(alphabet)}"
        for m_idx in range(start, end + 1):
            model_name = sorted_models[m_idx]
            model_cliques[model_name].append(letter)

    return {
        m: ", ".join(letters) if letters else "a"
        for m, letters in model_cliques.items()
    }


def pairwise_all_vs_all_tests(
    models: Sequence[str],
    mean_ranks: Dict[str, float],
    n_blocks: int,
    k_models: int,
    alpha: float = 0.05,
    correction: str = "holm",
) -> pl.DataFrame:
    """Performs pairwise comparison between all model pairs using rank z-statistics.

    Following Demšar (2006), z = (R_A - R_B) / sqrt((k * (k + 1)) / (6 * N)),
    with Holm-Bonferroni step-down correction for multiple testing.

    Args:
        models: List of model names.
        mean_ranks: Dict mapping model name to mean rank.
        n_blocks: Number of evaluation blocks (N).
        k_models: Number of models (k).
        alpha: Significance level threshold (default: 0.05).
        correction: Multiple testing correction ('holm' or 'none').

    Returns:
        Polars DataFrame containing pairwise test results.
    """
    if k_models <= 1 or n_blocks <= 0:
        return pl.DataFrame()

    se = np.sqrt((k_models * (k_models + 1)) / (6.0 * n_blocks))
    pairs = []
    unique_pairs = []

    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            m_a = models[i]
            m_b = models[j]
            r_a = mean_ranks[m_a]
            r_b = mean_ranks[m_b]
            diff_r = r_a - r_b
            z_val = diff_r / se
            p_raw = float(2.0 * scipy_stats.norm.sf(abs(z_val)))

            pair_dict = {
                "model_a": m_a,
                "model_b": m_b,
                "rank_a": r_a,
                "rank_b": r_b,
                "rank_diff": diff_r,
                "z_stat": float(z_val),
                "p_raw": p_raw,
            }
            pairs.append(pair_dict)

            # Keep canonical upper pair for multiplicity correction
            if i < j:
                unique_pairs.append((m_a, m_b, p_raw))

    # Apply Holm correction on unique pairs
    if correction == "holm" and unique_pairs:
        raw_p_list = [p for _, _, p in unique_pairs]
        adj_p_list = holm_bonferroni(raw_p_list)
        p_adj_map = {}
        for (m_a, m_b, _), p_adj in zip(unique_pairs, adj_p_list):
            p_adj_map[(m_a, m_b)] = p_adj
            p_adj_map[(m_b, m_a)] = p_adj
    else:
        p_adj_map = {(p["model_a"], p["model_b"]): p["p_raw"] for p in pairs}

    for p in pairs:
        p["p_adj"] = p_adj_map.get((p["model_a"], p["model_b"]), p["p_raw"])
        p["is_significant"] = bool(p["p_adj"] < alpha)

    return pl.DataFrame(pairs)


def compute_demsar_all_vs_all(
    df: pl.DataFrame | Sequence[pl.DataFrame],
    dataset: Optional[str | Sequence[str]] = None,
    datasets: Optional[str | Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    exclude_clustering: Optional[Sequence[str]] = None,
    exclude_dim_red: Optional[Sequence[str]] = None,
    merge_info0: bool = False,
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Performs full Demšar (2006) All-vs-All Model Comparison across metrics.

    Following Demšar (2006) recommendations, algorithms are compared across multiple
    independent benchmark datasets. On each dataset, scores are averaged over random
    seeds and topic counts. Models are ranked on each dataset, and differences in
    mean ranks across datasets are evaluated via Friedman / Iman-Davenport omnibus
    tests and Nemenyi post-hoc critical differences (or Holm-adjusted pairwise tests).

    Workflow:
    1. Dataset Normalization & Filtering: Cleans dataset identifiers and filters
       by specified datasets.
    2. Seed & Topic Aggregation: Averages scores over random seeds and topic
       counts for each (dataset, model_type).
    3. Block Construction: Constructs the (N datasets x k models) evaluation matrix.
    4. Dataset-by-Dataset Ranking: Converts scores to fractional ranks (Rank 1 = best)
       for each dataset and computes mean rank R_j for each model across datasets.
    5. Omnibus Testing: Computes Friedman chi-square and Iman-Davenport F_F test.
    6. Post-Hoc Analysis: If omnibus test is significant (p < alpha), computes
       Nemenyi CD, equivalence cliques, and all-pairs z-tests with Holm correction.
    7. Deliverables Generation: Creates Model Ranking Summary Table and k x k
       Pairwise Delta Matrix for each metric.

    Args:
        df: Polars DataFrame or sequence of DataFrames with results.
        dataset: Dataset identifier or list of datasets to filter by (None = all).
        datasets: Alias for dataset parameter.
        metrics: List of metrics to evaluate (default: METRICS).
        alpha: Significance threshold (default: 0.05).
        exclude_clustering: Clustering algorithms to exclude.
        exclude_dim_red: Dimensionality reduction algorithms to exclude.
        merge_info0: Treat '_info0' variants as base model type.
        higher_is_better: Optional dict indicating optimization direction per metric.

    Returns:
        A dictionary mapping each metric name to its results dict, plus 'metadata'.
    """
    if isinstance(df, (list, tuple)):
        if not df:
            return {
                "metrics": {},
                "metadata": {
                    "n_blocks": 0,
                    "n_datasets": 0,
                    "k_models": 0,
                    "alpha": alpha,
                },
            }
        combined_df = pl.concat(
            [d.clone() for d in df if isinstance(d, pl.DataFrame) and not d.is_empty()],
            how="diagonal",
        )
    else:
        combined_df = df.clone()

    if combined_df.is_empty():
        return {
            "metrics": {},
            "metadata": {"n_blocks": 0, "n_datasets": 0, "k_models": 0, "alpha": alpha},
        }

    target_metrics = [m for m in (metrics or METRICS)]

    # Directionality default (NLP metrics: higher is better)
    direction_map = {m: True for m in target_metrics}
    if higher_is_better:
        direction_map.update(higher_is_better)

    # 1. Dataset normalization and filtering
    if "dataset_name" not in combined_df.columns:
        combined_df = combined_df.with_columns(pl.lit("dataset").alias("dataset_name"))
    else:
        combined_df = combined_df.with_columns(
            canonical_dataset_expr(pl.col("dataset_name"))
        )

    target_ds = datasets if datasets is not None else dataset
    if target_ds is not None and target_ds != "all":
        if isinstance(target_ds, str):
            combined_df = combined_df.filter(pl.col("dataset_name") == target_ds)
        else:
            combined_df = combined_df.filter(
                pl.col("dataset_name").is_in(list(target_ds))
            )

    if combined_df.is_empty():
        return {
            "metrics": {},
            "metadata": {"n_blocks": 0, "n_datasets": 0, "k_models": 0, "alpha": alpha},
        }

    # 2. Extract model_type and topic_idx
    m_types = []
    t_indices = []
    model_names = (
        combined_df["model_name"].to_list()
        if "model_name" in combined_df.columns
        else []
    )
    for name in model_names:
        mtype, tidx = parse_model_type_and_topic(str(name), merge_info0=merge_info0)
        m_types.append(mtype)
        t_indices.append(tidx)

    annotated = combined_df.with_columns(
        [
            pl.Series("model_type", m_types),
            pl.Series("topic_idx", t_indices),
        ]
    )

    if "nr_topics" in annotated.columns:
        annotated = annotated.with_columns(
            pl.when(pl.col("topic_idx") == 0)
            .then(pl.col("nr_topics").fill_null(0))
            .otherwise(pl.col("topic_idx"))
            .alias("topic_idx")
        )
    elif "n_topics" in annotated.columns:
        annotated = annotated.with_columns(
            pl.when(pl.col("topic_idx") == 0)
            .then(pl.col("n_topics").fill_null(0))
            .otherwise(pl.col("topic_idx"))
            .alias("topic_idx")
        )

    # 3. Apply algorithm exclusion filters
    def filter_algos(
        dframe: pl.DataFrame,
        col_name: str,
        exclusions: Optional[Sequence[str]],
        default_pattern: Optional[str],
    ) -> pl.DataFrame:
        if col_name not in dframe.columns:
            return dframe
        if exclusions is not None:
            norm_exclusions = [e.lower().replace("_", "") for e in exclusions]
            cond = (
                dframe[col_name]
                .fill_null("")
                .map_elements(
                    lambda x: any(
                        ex in x.lower().replace("_", "")
                        or x.lower().replace("_", "") == ex
                        for ex in norm_exclusions
                    ),
                    return_dtype=pl.Boolean,
                )
            )
            return dframe.filter(~cond)
        elif default_pattern:
            return dframe.filter(~pl.col(col_name).str.contains(default_pattern))
        return dframe

    annotated = filter_algos(
        annotated, "clustering_algo", exclude_clustering, "k_means"
    )
    annotated = filter_algos(annotated, "dim_red_algo", exclude_dim_red, "pca")

    if annotated.is_empty():
        return {
            "metrics": {},
            "metadata": {"n_blocks": 0, "n_datasets": 0, "k_models": 0, "alpha": alpha},
        }

    # Canonical model ordering preference
    canonical_order = [
        "mv_co_reg_spectral",
        "mv_co_reg_spectral_info0",
        "mv_spectral",
        "mv_spectral_info0",
        "aligned_umap",
        "append_umap",
        "baseline",
        "umap_spectral",
        "stm",
    ]

    all_present_models = annotated["model_type"].unique().to_list()
    ordered_models = [m for m in canonical_order if m in all_present_models]
    ordered_models += sorted(
        [m for m in all_present_models if m not in canonical_order]
    )

    available_metrics = [m for m in target_metrics if m in annotated.columns]

    # In Demšar (2006), evaluation blocks are the benchmark datasets
    block_cols = ["dataset_name"]

    # 4. Aggregate across random seeds and topic counts for each
    # (dataset_name, model_type)
    agg_exprs = [
        pl.col(m).cast(pl.Float64, strict=False).mean().alias(m)
        for m in available_metrics
    ]
    df_agg = (
        annotated.group_by(block_cols + ["model_type"])
        .agg(agg_exprs)
        .sort(block_cols + ["model_type"])
    )

    # Unique blocks (datasets)
    blocks_df = df_agg.select(block_cols).unique().sort(block_cols)
    n_blocks = len(blocks_df)

    results_by_metric: Dict[str, Any] = {}

    for metric in available_metrics:
        hib = direction_map.get(metric, True)

        # Pivot to Block x Model matrix
        # Filter models with complete observations across blocks
        valid_models = []
        for m in ordered_models:
            sub = df_agg.filter(
                (pl.col("model_type") == m) & pl.col(metric).is_not_null()
            )
            if len(sub) == n_blocks:
                valid_models.append(m)

        if len(valid_models) < 2 or n_blocks < 2:
            continue

        k_models = len(valid_models)

        # Build (N x k) score matrix
        score_matrix = np.zeros((n_blocks, k_models), dtype=float)
        for b_idx, b_row in enumerate(blocks_df.iter_rows(named=True)):
            for m_idx, model in enumerate(valid_models):
                filter_cond = pl.col("model_type") == model
                for b_col in block_cols:
                    filter_cond = filter_cond & (pl.col(b_col) == b_row[b_col])
                val = df_agg.filter(filter_cond)[metric][0]
                score_matrix[b_idx, m_idx] = float(val)

        # Block-by-block ranking (1 = best)
        # Using scipy.stats.rankdata:
        # rankdata on negative scores if higher_is_better assigns 1 to highest
        rank_matrix = np.zeros_like(score_matrix)
        for b_idx in range(n_blocks):
            row_vals = score_matrix[b_idx, :]
            to_rank = -row_vals if hib else row_vals
            rank_matrix[b_idx, :] = scipy_stats.rankdata(to_rank, method="average")

        mean_ranks_arr = np.mean(rank_matrix, axis=0)
        mean_scores_arr = np.mean(score_matrix, axis=0)
        std_scores_arr = np.std(score_matrix, axis=0)

        mean_ranks_dict = {
            model: float(mean_ranks_arr[idx]) for idx, model in enumerate(valid_models)
        }
        mean_scores_dict = {
            model: float(mean_scores_arr[idx]) for idx, model in enumerate(valid_models)
        }
        std_scores_dict = {
            model: float(std_scores_arr[idx]) for idx, model in enumerate(valid_models)
        }

        # Omnibus Test
        omnibus = friedman_omnibus_test(mean_ranks_arr, n_blocks, k_models)
        is_omnibus_sig = bool(omnibus["p_f_f"] < alpha)

        # Critical Difference & Post-Hoc
        cd = nemenyi_critical_difference(n_blocks, k_models, alpha=alpha)
        cliques = compute_nemenyi_cliques(valid_models, mean_ranks_dict, cd)

        # Pairwise All-vs-All tests
        pairwise_df = pairwise_all_vs_all_tests(
            models=valid_models,
            mean_ranks=mean_ranks_dict,
            n_blocks=n_blocks,
            k_models=k_models,
            alpha=alpha,
            correction="holm",
        )

        # Build Model Ranking Summary Table
        summary_rows = []
        # Sort by mean rank ascending (best to worst)
        ranked_models = sorted(
            valid_models, key=lambda m: (mean_ranks_dict[m], -mean_scores_dict[m])
        )

        best_model = ranked_models[0] if ranked_models else None

        for m in ranked_models:
            m_score = mean_scores_dict[m]
            m_std = std_scores_dict[m]
            m_rank = mean_ranks_dict[m]
            group = cliques.get(m, "a")

            summary_rows.append(
                {
                    "Model": m,
                    "Mean Score": m_score,
                    "Std Score": m_std,
                    "Mean Score (±SD)": f"{m_score:.3f} (±{m_std:.3f})",
                    "Mean Rank": m_rank,
                    "Significance Group": group,
                    "Is Best": (m == best_model),
                }
            )

        df_summary = pl.DataFrame(summary_rows)

        # Build Pairwise Delta Matrix (k x k)
        # Rows: Model A, Cols: Model B
        # Cell: Score_A - Score_B with '*' if post-hoc p_adj < alpha
        delta_matrix_rows = []
        for m_a in ranked_models:
            row_dict = {"Model": m_a}
            for m_b in ranked_models:
                if m_a == m_b:
                    row_dict[m_b] = "-"
                else:
                    diff_score = mean_scores_dict[m_a] - mean_scores_dict[m_b]
                    # Check post-hoc significance from pairwise_df
                    pair_match = pairwise_df.filter(
                        (pl.col("model_a") == m_a) & (pl.col("model_b") == m_b)
                    )
                    sig_flag = ""
                    if not pair_match.is_empty() and is_omnibus_sig:
                        if pair_match["is_significant"][0]:
                            sig_flag = "*"
                    elif (
                        is_omnibus_sig
                        and abs(mean_ranks_dict[m_a] - mean_ranks_dict[m_b]) > cd
                    ):
                        sig_flag = "*"

                    row_dict[m_b] = f"{diff_score:+.3f}{sig_flag}"
            delta_matrix_rows.append(row_dict)

        df_delta_matrix = pl.DataFrame(delta_matrix_rows)

        results_by_metric[metric] = {
            "metric": metric,
            "higher_is_better": hib,
            "n_blocks": n_blocks,
            "k_models": k_models,
            "models": ranked_models,
            "mean_ranks": mean_ranks_dict,
            "mean_scores": mean_scores_dict,
            "std_scores": std_scores_dict,
            "omnibus": omnibus,
            "is_significant": is_omnibus_sig,
            "critical_difference": cd,
            "cliques": cliques,
            "summary_table": df_summary,
            "pairwise_tests": pairwise_df,
            "pairwise_delta_matrix": df_delta_matrix,
        }

    return {
        "metrics": results_by_metric,
        "metadata": {
            "n_blocks": n_blocks,
            "n_datasets": n_blocks,
            "k_models": len(ordered_models),
            "alpha": alpha,
            "datasets": (
                list(combined_df["dataset_name"].unique())
                if "dataset_name" in combined_df.columns
                else []
            ),
        },
    }
