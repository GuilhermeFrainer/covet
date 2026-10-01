"""Interactive dashboard for BERTopic experiment results analysis.

This module provides a Streamlit-based dashboard to load, filter, and visualize
experimental results from CSV files in the results directory.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import datetime
import glob
import json
import os
import re
from typing import Optional

import altair as alt
import polars as pl
import streamlit as st

from src.experiment_tracker import (
    DEFAULT_SLURM_SCRIPT_CLUSTER,
    DEFAULT_SLURM_SCRIPT_LOCAL,
    build_coverage_matrix,
    classify_result_condition,
    extract_model_name,
    generate_grouped_slurm_commands,
    generate_slurm_command,
    scan_experiment_configs,
)
from src.make_table import (
    generate_best_models_latex_table,
    generate_best_models_markdown_table,
    generate_best_models_table_data,
    generate_demsar_all_vs_all_latex_table,
    generate_demsar_all_vs_all_report,
    generate_demsar_delta_latex_table,
    generate_demsar_delta_markdown_table,
    generate_noise_coverage_latex_table,
    generate_noise_coverage_markdown_table,
    generate_pairwise_delta_latex_matrix,
    generate_stopword_impact_latex_table,
    generate_stopword_impact_markdown_table,
    generate_stopword_impact_table_data,
    style_demsar_delta_dataframe,
    style_demsar_pairwise_matrix,
)
from src.metadata_alignment import fill_run_alignment, load_covariate_alignment
from src.model_catalog import (
    annotate_coverage,
    annotate_models,
    configuration_models,
    filter_catalog,
    kmeans_algorithms,
    load_catalog,
    sort_catalog,
)
from src.comparisons.analysis import (
    INFERENTIAL_METRICS,
    compute_ablation_comparisons,
    compute_registered_edge_comparisons,
    load_rq1_edges,
)
from src.results_analysis import (
    calculate_hdbscan_noise_coverage,
    canonical_dataset_expr,
    compute_demsar_all_vs_all,
    compute_demsar_delta_table,
    compute_stopword_impact,
    extract_model_type,
    find_best_models,
)

TABLES_DIR = PROJECT_ROOT / "tables"

# Default metrics for the visualization
DEFAULT_X_AXIS = "u_mass"
DEFAULT_Y_AXIS = "irbo"

# Configuration for metrics: which direction is "better"
# This can be easily extended in the future.
METRIC_CONFIG = {
    "duration_seconds": "min",
    "c_v": "max",
    "c_npmi": "max",
    "u_mass": "max",  # Higher (closer to 0) is better
    "irbo": "max",
    "topic_diversity": "max",
    "n_topics": "max",
    "n_observations": "max",
}

# Shown alongside the metrics but never highlighted as "best": models that take
# covariates as input can raise topic–metadata alignment by construction.
DESCRIPTIVE_METRICS = {"meta_ami_mean"}

ASSIGNMENTS_DIR = PROJECT_ROOT / "output" / "document_assignments"
ALIGNMENT_BACKFILL = (
    PROJECT_ROOT / "results" / "derived" / "metadata_alignment_backfill.csv"
)


@st.cache_data
def load_alignment() -> pl.DataFrame:
    """Per-covariate topic–metadata AMI from run manifests and the backfill."""
    return load_covariate_alignment(ASSIGNMENTS_DIR, ALIGNMENT_BACKFILL)


@st.cache_data
def load_all_results(results_dir: str = "results") -> pl.DataFrame:
    """Loads and concatenates result files. Extracts dataset and date objects."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    all_files = csv_files + json_files

    if not all_files:
        return pl.DataFrame()

    dfs = []
    for file in all_files:
        try:
            if file.endswith(".csv"):
                df = pl.read_csv(file, infer_schema_length=None)
            elif file.endswith(".json"):
                df = pl.read_json(file, infer_schema_length=None)
            else:
                continue

            # Normalize dataset and model names early
            if "dataset_name" in df.columns:
                df = df.with_columns(canonical_dataset_expr(pl.col("dataset_name")))

            if "model_name" in df.columns:
                df = df.with_columns(pl.col("model_name").str.replace("^stemmed_", ""))
            elif "model_id" in df.columns:
                df = df.with_columns(pl.col("model_id").str.replace("^stemmed_", ""))

            file_basename = os.path.basename(file)

            # Extract Date as actual date object
            date_match = re.search(r"-(\d{8})-", file_basename)
            exp_date = None
            if date_match:
                d_str = date_match.group(1)
                try:
                    exp_date = datetime.date(
                        int(d_str[:4]), int(d_str[4:6]), int(d_str[6:])
                    )
                except ValueError:
                    exp_date = None

            # Fallback to timestamp column if exp_date is still None
            if exp_date is None and "timestamp" in df.columns and len(df) > 0:
                ts_val = df["timestamp"][0]
                if ts_val and isinstance(ts_val, str):
                    try:
                        # Handle ISO formats like 2026-04-02T13:05:32.462016
                        exp_date = datetime.datetime.fromisoformat(ts_val).date()
                    except (ValueError, TypeError):
                        pass

            # Dataset extraction (New logic: use dataset_name column if it exists)
            dataset = "unknown"
            if "dataset_name" in df.columns and len(df) > 0:
                dataset = df["dataset_name"][0]
            elif "trump" in file_basename.lower():
                dataset = "trump"
            elif "yelp" in file_basename.lower():
                dataset = "yelp"
            else:
                dataset = file_basename.split("-")[0].split("_")[0]

            # Legacy fallback: ensure _embeddings suffix is stripped
            dataset = dataset.replace("_embeddings", "")

            cond = classify_result_condition(
                source_file=file_basename,
                exp_id=(
                    df["experiment_id"][0]
                    if "experiment_id" in df.columns and len(df) > 0
                    else ""
                ),
                stopword_removal_col=(
                    df["stopword_removal"][0]
                    if "stopword_removal" in df.columns and len(df) > 0
                    else None
                ),
            )

            df = df.with_columns(
                pl.lit(os.path.splitext(file_basename)[0]).alias("source_file"),
                pl.lit(dataset).alias("dataset_label"),
                pl.lit(cond).alias("condition"),
                pl.lit(exp_date).cast(pl.Date).alias("experiment_date"),
                pl.lit(
                    "optimizer" if "opt" in file_basename.lower() else "non-optimizer"
                ).alias("experiment_type"),
            )

            if "model_name" in df.columns:
                df = df.with_columns(
                    pl.col("model_name")
                    .map_elements(extract_model_type, return_dtype=pl.String)
                    .alias("model_type")
                )
            elif "model_id" in df.columns:
                df = df.with_columns(
                    pl.col("model_id")
                    .map_elements(extract_model_type, return_dtype=pl.String)
                    .alias("model_type")
                )
            else:
                df = df.with_columns(pl.lit("unknown").alias("model_type"))

            # Standardize complex columns across files to prevent schema mismatches
            if "representation" in df.columns:
                dtype = df["representation"].dtype
                if dtype == pl.String or dtype == pl.Utf8:

                    def parse_repr(x):
                        if not isinstance(x, str) or not x.strip():
                            return []
                        if x.startswith("["):
                            try:
                                return [str(w) for w in json.loads(x)]
                            except Exception:
                                pass
                        return [w.strip() for w in x.split(",")]

                    df = df.with_columns(
                        pl.col("representation").map_elements(
                            parse_repr, return_dtype=pl.List(pl.String)
                        )
                    )

            if "representative_docs" in df.columns:
                dtype = df["representative_docs"].dtype
                if dtype != pl.List(pl.String):
                    if isinstance(dtype, pl.List):
                        df = df.with_columns(
                            pl.col("representative_docs").cast(pl.List(pl.String))
                        )
                    else:
                        df = df.with_columns(
                            pl.col("representative_docs").map_elements(
                                lambda x: [str(x)] if x is not None else [],
                                return_dtype=pl.List(pl.String),
                            )
                        )

            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs, how="diagonal_relaxed")


def render_table_export_bar(
    latex_content: str,
    csv_content: str,
    markdown_content: str,
    file_slug: str,
    key_prefix: str,
):
    """Renders export buttons (LaTeX, CSV, Markdown) and a Save to tables/ button."""
    file_slug += "_" + st.session_state.get("catalog_scope_slug", "all")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button(
            label="📥 Download LaTeX (.tex)",
            data=latex_content,
            file_name=f"{file_slug}.tex",
            mime="text/x-tex",
            key=f"{key_prefix}_dl_tex",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="📥 Download CSV (.csv)",
            data=csv_content,
            file_name=f"{file_slug}.csv",
            mime="text/csv",
            key=f"{key_prefix}_dl_csv",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            label="📥 Download Markdown (.md)",
            data=markdown_content,
            file_name=f"{file_slug}.md",
            mime="text/markdown",
            key=f"{key_prefix}_dl_md",
            use_container_width=True,
        )
    with col4:
        if st.button(
            "💾 Save to tables/",
            key=f"{key_prefix}_save_btn",
            use_container_width=True,
            help=(
                "Saves .tex, .csv, and .md files directly into the project's "
                "tables/ directory."
            ),
        ):
            TABLES_DIR.mkdir(parents=True, exist_ok=True)
            tex_file = TABLES_DIR / f"{file_slug}.tex"
            md_file = TABLES_DIR / f"{file_slug}.md"
            csv_file = TABLES_DIR / f"{file_slug}.csv"
            tex_file.write_text(latex_content, encoding="utf-8")
            md_file.write_text(markdown_content, encoding="utf-8")
            csv_file.write_text(csv_content, encoding="utf-8")
            st.success(
                f"Saved `{file_slug}` (.tex, .md, .csv) to `{TABLES_DIR.name}/`!"
            )


@st.cache_data
def get_cached_best_models(
    df: pl.DataFrame,
    dataset: str,
    condition: str,
    exclude_clustering: tuple[str, ...] | None,
    exclude_dim_red: tuple[str, ...] | None,
    dump: bool,
    average: bool,
    merge_info0: bool,
    suppress_nulls: bool,
):
    f_df = df
    if condition != "all" and "condition" in f_df.columns:
        f_df = f_df.filter(pl.col("condition") == condition)

    ex_clust = list(exclude_clustering or ())
    ex_dim = list(exclude_dim_red or ())

    return find_best_models(
        f_df,
        dataset=dataset,
        exclude_clustering=ex_clust,
        exclude_dim_red=ex_dim,
        dump=dump,
        average=average,
        merge_info0=merge_info0,
        suppress_nulls=suppress_nulls,
    )


@st.cache_data
def get_cached_demsar_all_vs_all(
    df: pl.DataFrame,
    datasets: tuple[str, ...],
    condition: str,
    metrics: tuple[str, ...],
    alpha: float,
    exclude_clustering: tuple[str, ...] | None,
    exclude_dim_red: tuple[str, ...] | None,
    merge_info0: bool,
):
    f_df = df
    if condition != "all" and "condition" in f_df.columns:
        f_df = f_df.filter(pl.col("condition") == condition)

    filter_ds = None if "all" in [d.lower() for d in datasets] else list(datasets)

    ex_clust = list(exclude_clustering or ())
    ex_dim = list(exclude_dim_red or ())

    return compute_demsar_all_vs_all(
        df=f_df,
        datasets=filter_ds,
        metrics=list(metrics),
        alpha=alpha,
        exclude_clustering=ex_clust,
        exclude_dim_red=ex_dim,
        merge_info0=merge_info0,
    )


@st.cache_data
def get_cached_demsar_delta(
    df: pl.DataFrame,
    datasets: tuple[str, ...],
    condition: str,
    alpha: float,
    correction: str,
    exclude_clustering: tuple[str, ...] | None,
    exclude_dim_red: tuple[str, ...] | None,
    merge_info0: bool,
):
    df_std = df.filter(pl.col("condition") == "remove_rep_stopwords")
    df_alt = df.filter(pl.col("condition") == condition)

    filter_ds = None if "all" in [d.lower() for d in datasets] else list(datasets)

    ex_clust = list(exclude_clustering or ())
    ex_dim = list(exclude_dim_red or ())

    return compute_demsar_delta_table(
        df_default=df_std,
        df_alternative=df_alt,
        datasets=filter_ds,
        alpha=alpha,
        correction=correction,
        exclude_clustering=ex_clust,
        exclude_dim_red=ex_dim,
        merge_info0=merge_info0,
    )


@st.cache_data
def get_cached_stopword_impact(
    df: pl.DataFrame,
    dataset: str,
    exclude_clustering: tuple[str, ...] | None,
    exclude_dim_red: tuple[str, ...] | None,
    merge_info0: bool,
):
    df_rem = df.filter(pl.col("condition") == "remove_rep_stopwords")
    df_keep = df.filter(pl.col("condition") == "keep_rep_stopwords")

    ex_clust = list(exclude_clustering or ())
    ex_dim = list(exclude_dim_red or ())

    return compute_stopword_impact(
        df_remove_rep_stopwords=df_rem,
        df_keep_rep_stopwords=df_keep,
        dataset=dataset,
        exclude_clustering=ex_clust,
        exclude_dim_red=ex_dim,
        merge_info0=merge_info0,
    )


@st.cache_data
def get_cached_noise_coverage(
    df: pl.DataFrame,
    dataset: str | None,
    condition: str,
    merge_info0: bool,
):
    f_df = df
    if condition != "all" and "condition" in f_df.columns:
        f_df = f_df.filter(pl.col("condition") == condition)

    ds_arg = None if dataset == "all" else dataset
    return calculate_hdbscan_noise_coverage(
        df=f_df,
        dataset=ds_arg,
        group_by_model_type=True,
        merge_info0=merge_info0,
    )


def render_alignment_section(filtered_df: pl.DataFrame, alignment: pl.DataFrame):
    """Renders per-model topic–metadata AMI for one dataset and condition."""
    if not {"meta_ami_mean", "run_uid"} <= set(filtered_df.columns):
        st.info("No topic–metadata alignment values in the selected results.")
        return
    runs = filtered_df.filter(
        pl.col("run_uid").is_not_null() & pl.col("meta_ami_mean").is_not_null()
    ).unique(subset="run_uid", keep="first")
    if runs.is_empty():
        st.info("No topic–metadata alignment values in the selected results.")
        return

    col_ds, col_cond = st.columns(2)
    with col_ds:
        datasets = sorted(runs["dataset_label"].unique().to_list())
        dataset = st.selectbox("Dataset:", datasets, key="ami_dataset")
    with col_cond:
        in_dataset = runs.filter(pl.col("dataset_label") == dataset)
        conditions = sorted(in_dataset["condition"].unique().to_list())
        condition = st.selectbox(
            "Condition:", conditions,
            index=conditions.index("remove_rep_stopwords")
            if "remove_rep_stopwords" in conditions else 0,
            key="ami_condition",
        )
    runs = runs.filter(
        (pl.col("dataset_label") == dataset) & (pl.col("condition") == condition)
    ).select("run_uid", "model_type", pl.col("meta_ami_mean").cast(pl.Float64))
    model_order = (
        runs.group_by("model_type").agg(pl.col("meta_ami_mean").mean())
        .sort("meta_ami_mean", descending=True)["model_type"].to_list()
    )

    by_covariate = runs.join(
        alignment.select("run_uid", "covariate", "ami", "ami_source"), on="run_uid"
    )
    cells = by_covariate.group_by("model_type", "covariate").agg(
        pl.col("ami").mean().alias("AMI"), pl.len().alias("Runs")
    )
    means = runs.group_by("model_type").agg(
        pl.lit("(mean)").alias("covariate"),
        pl.col("meta_ami_mean").mean().alias("AMI"),
        pl.len().cast(pl.UInt32).alias("Runs"),
    )
    covariates = sorted(cells["covariate"].unique().to_list()) + ["(mean)"]
    heat = pl.concat([cells, means], how="vertical_relaxed").to_dicts()
    height = max(200, 28 * len(model_order))

    st.subheader("Mean AMI by covariate")
    base = alt.Chart(alt.Data(values=heat)).encode(
        x=alt.X(
            "covariate:N", sort=covariates, title=None,
            axis=alt.Axis(labelAngle=-30),
        ),
        y=alt.Y("model_type:N", sort=model_order, title=None),
    )
    rects = base.mark_rect().encode(
        color=alt.Color(
            "AMI:Q", scale=alt.Scale(scheme="blues", domainMin=0, clamp=True),
            legend=alt.Legend(title="AMI"),
        ),
        tooltip=[
            "model_type:N", "covariate:N", alt.Tooltip("AMI:Q", format=".4f"), "Runs:Q",
        ],
    )
    labels = base.mark_text(fontSize=11).encode(
        text=alt.Text("AMI:Q", format=".3f"),
        color=alt.value("#222222"),
    )
    st.altair_chart((rects + labels).properties(height=height), width="stretch")
    st.caption(
        "Cells average over seeds and requested topic counts. The (mean) column "
        "is meta_ami_mean, the run-level average over covariates."
    )

    st.subheader("Run-level meta_ami_mean")
    run_points = alt.Chart(runs).mark_circle(size=60, opacity=0.5).encode(
        x=alt.X("meta_ami_mean:Q", title="meta_ami_mean"),
        y=alt.Y("model_type:N", sort=model_order, title=None),
        color=alt.Color(
            "model_type:N", legend=None, scale=alt.Scale(scheme="tableau10")
        ),
        tooltip=[
            "model_type:N", alt.Tooltip("meta_ami_mean:Q", format=".4f"), "run_uid:N",
        ],
    )
    mean_marks = alt.Chart(means.to_pandas()).mark_point(
        shape="diamond", filled=True, color="#111111", size=140
    ).encode(
        x="AMI:Q",
        y=alt.Y("model_type:N", sort=model_order),
        tooltip=["model_type:N", alt.Tooltip("AMI:Q", format=".4f", title="Mean")],
    )
    st.altair_chart(
        (run_points + mean_marks).properties(height=height), width="stretch"
    )
    st.caption("Each dot is one run; the black diamond is the model mean.")

    summary = (
        runs.group_by("model_type").agg(
            pl.len().alias("Runs"),
            pl.col("meta_ami_mean").mean().alias("Mean AMI"),
            pl.col("meta_ami_mean").std().alias("SD"),
        )
        .join(
            by_covariate.unique("run_uid").group_by("model_type").agg(
                (pl.col("ami_source") == "backfill").sum().alias("Backfilled runs")
            ),
            on="model_type", how="left",
        )
        .sort("Mean AMI", descending=True)
    )
    st.dataframe(summary, hide_index=True, width="stretch")


def main():
    """Main function to run the Streamlit dashboard."""
    st.set_page_config(
        page_title="CA-BERTopic Experiment Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 CA-BERTopic Experiment Dashboard")

    # 1. Load Data
    results_dir = "results"
    output_dir = "output"
    df = load_all_results(results_dir)
    qual_df = load_all_results(output_dir)
    alignment = load_alignment()
    if not df.is_empty():
        df = fill_run_alignment(df, alignment)

    if df.is_empty():
        st.warning(f"No result files found in `{results_dir}/`.")

    # Catalog scope is shared by every tab, independent of tab-specific filters.
    try:
        catalog = load_catalog()
    except (ValueError, OSError) as exc:
        st.error(f"Cannot load model catalog: {exc}")
        return
    all_results = annotate_models(df, catalog)
    qual_df = annotate_models(qual_df, catalog)
    st.sidebar.header("Experiment Organization")
    priority = st.sidebar.selectbox(
        "Priority", ["primary", "secondary", "all", "unclassified"],
        format_func=str.title, key="catalog_priority",
    )
    families = st.sidebar.multiselect(
        "Family", ["hdbscan", "spectral", "k_means", "external"], key="catalog_families"
    )
    roles = st.sidebar.multiselect(
        "Role", ["baseline", "ablation", "external_baseline"], key="catalog_roles"
    )
    baselines = st.sidebar.multiselect(
        "Reference baseline", [mid for mid, entry in catalog.items() if entry["role"] == "baseline"],
        format_func=lambda mid: catalog[mid]["label"], key="catalog_baselines",
        help="Shows each selected baseline together with its ablations, within the selected priority.",
    )
    include_external = st.sidebar.checkbox(
        "Include external baselines alongside families", key="catalog_external"
    )
    scope = dict(priority=priority, families=families, roles=roles,
                 baselines=baselines, include_external=include_external)
    scope_slug = "_".join([priority] + sorted(families) + sorted(roles) + sorted(baselines)
                          + (["with_external"] if include_external else []))
    st.session_state["catalog_scope_slug"] = scope_slug
    df = filter_catalog(all_results, **scope)
    qual_df = filter_catalog(qual_df, **scope)
    st.caption(f"Catalog scope: {scope_slug}. {df.height} of {all_results.height} result rows selected.")
    unknown_count = all_results.filter(pl.col("priority") == "unclassified").height
    if unknown_count:
        st.sidebar.warning(f"{unknown_count} result rows are unclassified. Select All or Unclassified to inspect them.")

    # Show planned catalog members even when they have no configuration or results.
    discovered_catalog_exps = scan_experiment_configs(PROJECT_ROOT / "experiments")
    config_ids = configuration_models(discovered_catalog_exps)
    config_frame = annotate_models(pl.DataFrame(
        {"model_id": list(config_ids.values())}, schema={"model_id": pl.String}
    ), catalog)
    configured = set(config_frame["catalog_id"].to_list())
    observed = set(all_results["catalog_id"].to_list())
    catalog_frame = annotate_models(pl.DataFrame({"model_id": list(catalog)}), catalog)
    catalog_frame = filter_catalog(catalog_frame, **scope).with_columns(
        pl.col("catalog_id").is_in(configured).alias("configuration_available"),
        pl.col("catalog_id").is_in(observed).alias("results_available"),
    )
    catalog_frame = sort_catalog(catalog_frame)
    with st.expander("Model catalog and availability"):
        st.caption("Availability across active configurations and loaded results; independent of dataset filters.")
        st.dataframe(catalog_frame, hide_index=True, width="stretch")
    if df.is_empty():
        st.info("No results in this catalog scope. Configuration availability is shown above.")
        coverage = build_coverage_matrix(discovered_catalog_exps, all_results)
        if not coverage.is_empty():
            coverage = filter_catalog(annotate_coverage(coverage, discovered_catalog_exps, catalog), **scope)
            st.subheader("Experiment Coverage")
            st.dataframe(sort_catalog(coverage, "experiment_name"), hide_index=True, width="stretch")
        return

    # 2. Sidebar Filters
    st.sidebar.header("Data Filters")

    # --- Cascading Filter Logic Setup ---
    # We use session state to track selections and allow bidirectional filtering
    filter_config = {
        "dataset_label": {"label": "Datasets:", "is_sidebar": True},
        "model_type": {"label": "Model Types:", "is_sidebar": True},
        "experiment_type": {"label": "Experiment Types:", "is_sidebar": True},
        "clustering_algo": {"label": "Clustering Algo:", "is_sidebar": False},
        "dim_red_algo": {"label": "Dim Red Algo:", "is_sidebar": False},
        "n_observations": {"label": "N Observations:", "is_sidebar": False},
    }

    for key in filter_config:
        if key not in st.session_state:
            st.session_state[key] = []
        elif key in df.columns:
            available = set(df[key].unique().to_list())
            st.session_state[key] = [value for value in st.session_state[key] if value in available]

    def get_filtered_df(exclude_key: Optional[str] = None) -> pl.DataFrame:
        """
        Returns the dataframe filtered by all active filters except the
        one specified.
        """
        f_df = df
        for k in filter_config:
            if k != exclude_key and st.session_state[k]:
                f_df = f_df.filter(pl.col(k).is_in(st.session_state[k]))

        # Also apply date and file filters if they are not the excluded ones
        # (These are currently treated as "always apply" for simplicity in this helper)
        if "excluded_files" in st.session_state and st.session_state.excluded_files:
            f_df = f_df.filter(
                ~pl.col("source_file").is_in(st.session_state.excluded_files)
            )

        return f_df

    # Dataset Filter
    dataset_opts = sorted(
        get_filtered_df("dataset_label")["dataset_label"].unique().to_list()
    )
    st.sidebar.multiselect("Datasets:", options=dataset_opts, key="dataset_label")

    # Model Type Filter
    model_type_opts = sorted(
        get_filtered_df("model_type")["model_type"].unique().to_list()
    )
    st.sidebar.multiselect("Model Types:", options=model_type_opts, key="model_type")

    # Metadata Filters
    with st.sidebar.expander("Algorithmic & Data Filters", expanded=True):
        # Clustering Algo Filter
        if "clustering_algo" in df.columns:
            clustering_opts = sorted(
                get_filtered_df("clustering_algo")["clustering_algo"]
                .unique()
                .drop_nulls()
                .to_list()
            )
            st.multiselect(
                "Clustering Algo:", options=clustering_opts, key="clustering_algo"
            )

        # Dim Red Algo Filter
        if "dim_red_algo" in df.columns:
            dim_red_opts = sorted(
                get_filtered_df("dim_red_algo")["dim_red_algo"]
                .unique()
                .drop_nulls()
                .to_list()
            )
            st.multiselect("Dim Red Algo:", options=dim_red_opts, key="dim_red_algo")

        # N Observations Filter
        if "n_observations" in df.columns:
            n_obs_opts = sorted(
                get_filtered_df("n_observations")["n_observations"]
                .unique()
                .drop_nulls()
                .to_list()
            )
            st.multiselect("N Observations:", options=n_obs_opts, key="n_observations")

    # Date Range Filter (Not strictly cascaded with others to avoid circular complexity,
    # but we'll use the filtered DF for available dates)
    st.sidebar.subheader("Date Filtering")
    # Use DF filtered by everything else to find valid dates
    date_filtered_df = get_filtered_df()
    valid_dates = date_filtered_df.filter(pl.col("experiment_date").is_not_null())[
        "experiment_date"
    ]

    if not valid_dates.is_empty():
        min_date, max_date = valid_dates.min(), valid_dates.max()

        date_selection = st.sidebar.date_input(
            "Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Select a start and end date for filtering experiments.",
        )

        # Handle range selection (returns a tuple of 1 or 2 items)
        if isinstance(date_selection, tuple) and len(date_selection) == 2:
            start_date, end_date = date_selection
        elif isinstance(date_selection, tuple) and len(date_selection) == 1:
            start_date = end_date = date_selection[0]
        else:
            start_date = end_date = date_selection

        # Specific Date Multiselect (Inclusion)
        all_available_dates = sorted(valid_dates.unique().to_list())
        specific_dates = st.sidebar.multiselect(
            "Filter to specific dates:",
            options=all_available_dates,
            help=(
                "If selected, only these specific dates will be shown "
                "regardless of the range above."
            ),
        )
    else:
        start_date = end_date = None
        specific_dates = []

    # Experiment Type Filter
    exp_type_opts = sorted(
        get_filtered_df("experiment_type")["experiment_type"].unique().to_list()
    )
    st.sidebar.multiselect(
        "Experiment Types:", options=exp_type_opts, key="experiment_type"
    )

    # File Exclusion Filter
    with st.sidebar.expander("Exclude Specific Files"):
        all_files = sorted(df["source_file"].unique().to_list())
        st.multiselect("Files to ignore:", options=all_files, key="excluded_files")

    # Column Visibility Selector
    st.sidebar.header("Column Visibility")
    all_columns = df.columns
    # Default columns to show (hiding more technical/verbose ones)
    default_show = [
        c for c in all_columns if c not in ["source_file", "timestamp", "dataset_name"]
    ]
    selected_columns = st.sidebar.multiselect(
        "Columns to display in table:", options=all_columns, default=default_show
    )

    # Final Filter Application
    filter_expr = pl.lit(True)  # Start with always True

    for key in filter_config:
        if st.session_state[key]:
            filter_expr = filter_expr & (pl.col(key).is_in(st.session_state[key]))

    if "excluded_files" in st.session_state and st.session_state.excluded_files:
        filter_expr = filter_expr & (
            ~pl.col("source_file").is_in(st.session_state.excluded_files)
        )

    # Date logic: use specific dates if provided, otherwise use range
    if specific_dates:
        filter_expr = filter_expr & (pl.col("experiment_date").is_in(specific_dates))
    elif start_date and end_date:
        filter_expr = filter_expr & (
            pl.col("experiment_date").is_between(start_date, end_date)
        )

    filtered_df = df.filter(filter_expr)

    if filtered_df.is_empty():
        st.info("No data matches the selected filters.")
        return

    # 3. Main Tabs
    tab_metrics, tab_qualitative, tab_coverage, tab_paper_tables, tab_ablations = st.tabs(
        [
            "📊 Quantitative Metrics",
            "🔍 Qualitative Analysis",
            "📋 Experiment Coverage",
            "📑 Paper Results Tables",
            "🧪 Ablation Comparisons",
        ]
    )

    with tab_ablations:
        st.header("RQ1: Exploratory Ablation Results")
        st.warning(
            "Exploratory preview for presentation design and advisor discussion. "
            "These partial results are not final submission results. Regenerate the "
            "analysis after the planned runs and parity checks are complete."
        )
        st.caption(
            "Rows are explicit directed comparisons from the RQ1 edge registry. "
            "Each dataset contributes one value after averaging matched seeds and "
            "requested topic counts. Positive deltas favor the variant."
        )
        try:
            rq1_edges = load_rq1_edges()
        except (OSError, ValueError) as exc:
            st.error(f"Cannot load RQ1 comparison edges: {exc}")
            rq1_edges = []
        main_edges = [edge for edge in rq1_edges if edge["main_figure"]]
        edge_datasets, edge_summary, edge_runs = compute_registered_edge_comparisons(
            all_results, catalog, main_edges
        ) if main_edges else (pl.DataFrame(), pl.DataFrame(), pl.DataFrame())

        if edge_datasets.is_empty():
            st.info("No matched results are available for the registered RQ1 edges.")
        else:
            standard_condition = "remove_rep_stopwords"
            plot_rows = edge_datasets.filter(
                (pl.col("Condition") == standard_condition)
                & pl.col("Metric").is_in(["c_npmi", "irbo", "meta_ami_mean"])
                & pl.col("Improvement delta").is_not_null()
            )
            summary_by_edge_metric = {
                (row["Edge ID"], row["Metric"]): row
                for row in edge_summary.to_dicts()
            }
            edge_order = [edge["id"] for edge in main_edges]
            plot_row_names = {}
            for edge in main_edges:
                for metric in ("c_npmi", "irbo", "meta_ami_mean"):
                    result = summary_by_edge_metric.get((edge["id"], metric), {})
                    n = result.get("Datasets", 0) or 0
                    wins, ties, losses = (
                        result.get("Wins"), result.get("Ties"), result.get("Losses")
                    )
                    wtl = f"{wins}/{ties}/{losses}" if wins is not None else "not tested"
                    plot_row_names[(edge["id"], metric)] = (
                        f"{edge['chain']} · {edge['label']}   "
                        f"(W/T/L {wtl}; n={n}/5)"
                    )

            st.subheader("Dataset-level improvement plot")
            st.markdown(
                "Each dot is one dataset. The black diamond marks the median dataset "
                "delta. Dataset colors and marker shapes are consistent across panels. "
                "Rows are grouped by the registered comparison chain."
            )
            dataset_order = ["anes", "fed", "gadarian", "trump", "yelp"]
            dataset_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
            dataset_shapes = ["circle", "square", "triangle-up", "diamond", "cross"]

            def delta_panel(metric, title, x_title=None):
                rows = plot_rows.filter(pl.col("Metric") == metric).to_dicts()
                if not rows:
                    return None
                for row in rows:
                    row["Plot row"] = plot_row_names[(row["Edge ID"], metric)]
                rows.sort(key=lambda row: edge_order.index(row["Edge ID"]))
                ordered_labels = [
                    plot_row_names[(edge["id"], metric)] for edge in main_edges
                    if (edge["id"], metric) in summary_by_edge_metric
                ]
                medians = []
                for edge in main_edges:
                    summary = summary_by_edge_metric.get((edge["id"], metric))
                    if summary and summary.get("Median dataset delta") is not None:
                        medians.append({
                            "Plot row": plot_row_names[(edge["id"], metric)],
                            "Median dataset delta": summary["Median dataset delta"],
                        })
                max_abs = max(abs(row["Improvement delta"]) for row in rows)
                limit = max(max_abs * 1.15, 0.005)
                y = alt.Y(
                    "Plot row:N", sort=ordered_labels,
                    axis=alt.Axis(title=None, labelLimit=340, labelFontSize=10),
                )
                x = alt.X(
                    "Improvement delta:Q",
                    title=x_title or f"Improvement-oriented Δ {metric}",
                    scale=alt.Scale(domain=[-limit, limit]),
                )
                base = alt.Chart(alt.Data(values=rows))
                points = base.mark_point(filled=True, size=90, opacity=0.9).encode(
                    x=x,
                    y=y,
                    yOffset=alt.YOffset("Dataset:N", sort=dataset_order),
                    color=alt.Color(
                        "Dataset:N",
                        scale=alt.Scale(domain=dataset_order, range=dataset_colors),
                        legend=alt.Legend(title="Dataset"),
                    ),
                    shape=alt.Shape(
                        "Dataset:N",
                        scale=alt.Scale(domain=dataset_order, range=dataset_shapes),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Chain:N"), alt.Tooltip("Comparison:N"),
                        alt.Tooltip("Dataset:N"), alt.Tooltip("Improvement delta:Q", format=".5f"),
                        alt.Tooltip("Coverage:N"),
                    ],
                )
                median = alt.Chart(alt.Data(values=medians)).mark_point(
                    shape="diamond", color="#111111", size=170,
                ).encode(
                    x=alt.X("Median dataset delta:Q", scale=alt.Scale(domain=[-limit, limit])),
                    y=alt.Y("Plot row:N", sort=ordered_labels),
                    tooltip=[alt.Tooltip("Median dataset delta:Q", format=".5f")],
                )
                zero = alt.Chart(alt.Data(values=[{"zero": 0}])).mark_rule(
                    color="#666666", strokeDash=[4, 4],
                ).encode(x=alt.X("zero:Q"))
                height = max(260, 46 * len(ordered_labels))
                return (zero + points + median).properties(
                    title=title, height=height,
                ).configure_axis(grid=True, gridOpacity=0.2)

            figure_left, figure_right = st.columns(2)
            with figure_left:
                chart = delta_panel("c_npmi", "A · NPMI")
                if chart is not None:
                    st.altair_chart(chart, width="stretch")
            with figure_right:
                chart = delta_panel("irbo", "B · IRBO")
                if chart is not None:
                    st.altair_chart(chart, width="stretch")
            chart = delta_panel(
                "meta_ami_mean", "C · Topic–metadata AMI",
                x_title="Δ meta_ami_mean (variant − reference)",
            )
            if chart is not None:
                st.altair_chart(chart, width="stretch")
                st.caption(
                    "Panel C is a mechanism check, not a quality score: a positive "
                    "delta means the variant's topics follow the raw covariates more "
                    "closely. W/T/L counts increases/ties/decreases."
                )
            st.caption(
                "No confidence intervals are shown. Datasets are the independent units; "
                "seed and requested-count runs are repeated conditions. A row's dataset "
                "set can differ from another row, as shown by its n and W/T/L annotation."
            )

            main_table_rows = []
            edge_dataset_lookup = {
                (row["Edge ID"], row["Metric"]): row
                for row in edge_summary.to_dicts()
            }
            metric_labels = {
                "c_v": "C_v",
                "c_npmi": "NPMI",
                "u_mass": "UMass",
                "irbo": "IRBO",
                "topic_diversity": "Topic diversity",
                "duration_seconds": "Duration (s)",
                "outliers": "Outliers",
                "meta_ami_mean": "Metadata AMI",
            }
            for edge in main_edges:
                row = {
                    "Chain": edge["chain"],
                    "Comparison": edge["label"],
                    "Intended change": edge["intended_change"],
                    "Type": edge["classification"].replace("_", " "),
                }
                for metric in INFERENTIAL_METRICS:
                    summary = edge_dataset_lookup.get((edge["id"], metric), {})
                    label = metric_labels.get(metric, metric)
                    row[f"Median Δ {label}"] = summary.get("Median dataset delta")
                    wins, ties, losses = (
                        summary.get("Wins"), summary.get("Ties"), summary.get("Losses")
                    )
                    row[f"{label} W/T/L"] = (
                        f"{wins}/{ties}/{losses}" if wins is not None else "—"
                    )
                main_table_rows.append(row)
            st.subheader("Compact comparison table")
            st.dataframe(
                pl.DataFrame(main_table_rows, infer_schema_length=None),
                hide_index=True,
                width="stretch",
            )

            with st.expander("Exploratory statistical diagnostics"):
                stats_view = edge_summary.drop(
                    [column for column in ("Edge ID", "Model ID", "Baseline ID", "Edge order") if column in edge_summary.columns]
                )
                st.dataframe(stats_view, hide_index=True, width="stretch")
                st.caption(
                    "Exact p-values and Holm adjustments are exploratory. The adjustment "
                    "currently covers estimable registered main-figure edges within each "
                    "metric; incomplete edges are shown but omitted from that adjustment. "
                    "Final family membership remains to be approved."
                )
            with st.expander("Dataset-level and seed × topic-count details"):
                condition_options = sorted(edge_datasets["Condition"].unique().to_list())
                condition = st.selectbox(
                    "Preprocessing condition:", condition_options,
                    index=condition_options.index("remove_rep_stopwords")
                    if "remove_rep_stopwords" in condition_options else 0,
                    key="rq1_detail_condition",
                )
                dataset_view = edge_datasets.filter(pl.col("Condition") == condition).drop(
                    [column for column in ("Edge ID", "Model ID", "Baseline ID", "Edge order") if column in edge_datasets.columns]
                )
                run_view = edge_runs.filter(pl.col("Condition") == condition).drop(
                    [column for column in ("Edge ID", "Model ID", "Edge order") if column in edge_runs.columns]
                )
                st.markdown("**Dataset summaries and coverage**")
                st.dataframe(dataset_view, hide_index=True, width="stretch")
                st.markdown("**Matched seed × requested-topic-count values**")
                st.dataframe(run_view, hide_index=True, width="stretch")
                st.caption(
                    "Matching sample size does not prove identical document samples or "
                    "historical effective configurations. None of the comparisons has "
                    "passed a full historical parity audit."
                )

        with st.expander("All catalog baseline comparisons (supporting inventory)"):
            include_secondary = st.checkbox(
                "Include secondary catalog ablations",
                value=False,
                key="ablation_include_secondary",
            )
            primary_ids = {
                model_id for model_id, entry in catalog.items()
                if entry["role"] == "ablation" and entry["priority"] == "primary"
            }
            pair_datasets, pair_summary, _ = compute_ablation_comparisons(
                all_results,
                catalog,
                summary_model_ids=None if include_secondary else primary_ids,
            )
            if pair_summary.is_empty():
                st.info("No catalog baseline comparison results are available.")
            else:
                st.dataframe(
                    pair_summary.drop(
                        [column for column in ("Model ID", "Baseline ID") if column in pair_summary.columns]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "These rows follow model_catalog.yaml baseline_id references. They "
                    "supplement the directed RQ1 edges above and do not imply adjacent "
                    "component changes or verified strict ablations."
                )

    with tab_metrics:
        # 3. Data Table with Great Tables
        st.header("📋 Consolidated Results")

        # Metadata columns list
        metadata_cols = [
            "dataset_name",
            "dataset_label",
            "experiment_date",
            "timestamp",
            "model_type",
            "model_name",
            "clustering_algo",
            "dim_red_algo",
            "experiment_type",
            "source_file",
        ]

        # Identify numeric columns for metrics
        numeric_cols = [
            col
            for col, dtype in zip(filtered_df.columns, filtered_df.dtypes)
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            and col not in metadata_cols
        ]

        # Metrics overview
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Experiments", len(filtered_df))
        m_col2.metric("Datasets", filtered_df["dataset_label"].n_unique())
        m_col3.metric("Model Types", filtered_df["model_type"].n_unique())
        avg_dur = (
            filtered_df["duration_seconds"].mean()
            if "duration_seconds" in filtered_df.columns
            else 0
        )
        m_col4.metric(
            "Avg Duration (s)",
            round(avg_dur, 2) if avg_dur is not None else 0,
        )

        # Check for NaN / None metrics in filtered_df
        nan_info = []
        # Missing AMI marks runs without exported assignments, not a failed metric.
        for col in [c for c in numeric_cols if c not in DESCRIPTIVE_METRICS]:
            null_c = filtered_df[col].null_count()
            nan_c = (
                filtered_df.filter(pl.col(col).is_nan()).shape[0]
                if filtered_df[col].dtype in [pl.Float32, pl.Float64]
                else 0
            )
            total_invalid = null_c + nan_c
            if total_invalid > 0:
                nan_info.append(f"`{col}` ({total_invalid} NaN/missing)")

        if nan_info:
            cols_str = ", ".join(nan_info)
            st.error(
                f"🚨 **Metric Errors Detected:** Found invalid values in "
                f"{len(nan_info)} column(s): {cols_str}. "
                "Cells with NaNs/errors are styled in red below."
            )

        tab1_view_mode = st.radio(
            "Display Mode:",
            options=[
                "📊 Aggregated by Model Type (Mean ± SD across seeds & topic numbers)",
                "📄 Raw Individual Runs",
            ],
            horizontal=True,
            key="tab1_view_mode",
        )

        import pandas as pd

        if "📊 Aggregated" in tab1_view_mode:
            # Group by dataset and model_type (and condition if multiple)
            agg_group_cols = ["dataset_label", "model_type"]
            if (
                "condition" in filtered_df.columns
                and filtered_df["condition"].n_unique() > 1
            ):
                agg_group_cols.append("condition")

            # Determine numeric metrics present in filtered_df
            eval_metrics = [
                m
                for m in [
                    "u_mass",
                    "c_v",
                    "c_npmi",
                    "irbo",
                    "topic_diversity",
                    "outliers",
                    "meta_ami_mean",
                    "duration_seconds",
                ]
                if m in filtered_df.columns
            ]

            agg_exprs = [pl.len().alias("N_Runs")]
            for m in eval_metrics:
                agg_exprs.append(
                    pl.col(m).cast(pl.Float64, strict=False).mean().alias(f"{m}_mean")
                )
                agg_exprs.append(
                    pl.col(m)
                    .cast(pl.Float64, strict=False)
                    .std()
                    .fill_null(0.0)
                    .alias(f"{m}_std")
                )

            agg_df = (
                filtered_df.group_by(agg_group_cols)
                .agg(agg_exprs)
                .sort(["dataset_label", "model_type"])
            )

            # Build display and numeric pandas DataFrames
            disp_rows = []
            num_rows = []
            for r in agg_df.iter_rows(named=True):
                disp_r = {
                    "Dataset": r["dataset_label"].upper(),
                    "Model": r["model_type"],
                    "Runs (N)": r["N_Runs"],
                }
                num_r = {
                    "Dataset": r["dataset_label"].upper(),
                    "Model": r["model_type"],
                    "Runs (N)": r["N_Runs"],
                }
                if "condition" in agg_group_cols:
                    disp_r["Condition"] = r["condition"]
                    num_r["Condition"] = r["condition"]

                for m in eval_metrics:
                    m_mean = r[f"{m}_mean"]
                    m_std = r[f"{m}_std"]
                    num_r[m] = m_mean
                    if m_mean is None or (
                        isinstance(m_mean, float) and pd.isna(m_mean)
                    ):
                        disp_r[m] = "-"
                    elif m_std is not None and m_std > 0.0:
                        disp_r[m] = f"{m_mean:.3f} ± {m_std:.3f}"
                    else:
                        disp_r[m] = f"{m_mean:.3f}"

                disp_rows.append(disp_r)
                num_rows.append(num_r)

            disp_pandas = pd.DataFrame(disp_rows)
            num_pandas = pd.DataFrame(num_rows)

            def highlight_agg_metrics(col_series):
                c_name = col_series.name
                if c_name not in eval_metrics or c_name in DESCRIPTIVE_METRICS:
                    return [""] * len(col_series)

                styles = []
                direction = METRIC_CONFIG.get(c_name, "max")

                # Find best value per dataset
                for i in range(len(col_series)):
                    ds = num_pandas["Dataset"].iloc[i]
                    ds_mask = num_pandas["Dataset"] == ds
                    ds_vals = num_pandas.loc[ds_mask, c_name].dropna()
                    val = num_pandas[c_name].iloc[i]
                    if pd.isna(val) or ds_vals.empty:
                        styles.append("")
                    else:
                        best_val = (
                            ds_vals.max() if direction == "max" else ds_vals.min()
                        )
                        if val == best_val:
                            styles.append(
                                "background-color: #2E7D32; color: white; "
                                "font-weight: bold;"
                            )
                        else:
                            styles.append("")
                return styles

            styled_agg = disp_pandas.style.apply(highlight_agg_metrics, axis=0)
            st.dataframe(styled_agg, width="stretch", hide_index=True)
            st.caption(
                "Aggregated across seeds and topic counts (Mean ± SD). "
                ":green-background[**Green**: Best performing model type per dataset]. "
                "`meta_ami_mean` (topic–metadata AMI) is descriptive and never "
                "highlighted."
            )

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.download_button(
                    label="📥 Download Aggregated Table (.csv)",
                    data=disp_pandas.to_csv(index=False),
                    file_name=f"aggregated_results_{scope_slug}.csv",
                    mime="text/csv",
                    key="agg_dl_csv",
                    use_container_width=True,
                )
            with col_e2:
                st.download_button(
                    label="📥 Download Aggregated Table (.md)",
                    data=disp_pandas.to_markdown(index=False),
                    file_name=f"aggregated_results_{scope_slug}.md",
                    mime="text/markdown",
                    key="agg_dl_md",
                    use_container_width=True,
                )
        else:
            # Table with highlighting (Raw individual runs)
            display_df = filtered_df.clone()
            if "n_clusters" in display_df.columns:
                display_df = display_df.with_columns(
                    pl.col("n_clusters").cast(pl.Int64, strict=False)
                )

            pdf = display_df.to_pandas()

            def highlight_metrics(s):
                styles = [""] * len(s)
                is_metric = s.name in METRIC_CONFIG
                is_numeric = (
                    s.name in numeric_cols and s.name not in DESCRIPTIVE_METRICS
                )

                if is_metric or is_numeric:
                    numeric_s = pd.to_numeric(s, errors="coerce")

                    for i, val in enumerate(s):
                        num_val = numeric_s.iloc[i]
                        if pd.isna(val) or pd.isna(num_val):
                            styles[i] = (
                                "background-color: #F8D7DA; color: #721C24; "
                                "font-weight: bold; border: 1px solid #F5C6CB;"
                            )

                    if is_metric:
                        valid_s = numeric_s.dropna()
                        if not valid_s.empty:
                            direction = METRIC_CONFIG[s.name]
                            best_val = (
                                valid_s.max() if direction == "max" else valid_s.min()
                            )
                            for i, num_val in enumerate(numeric_s):
                                if (
                                    pd.notna(num_val)
                                    and num_val == best_val
                                    and styles[i] == ""
                                ):
                                    styles[i] = (
                                        "background-color: #2E7D32; "
                                        "color: white; font-weight: bold;"
                                    )

                return styles

            # Filter pdf to selected columns for display
            pdf_display = pdf[selected_columns] if selected_columns else pdf
            styled_table = pdf_display.style.apply(highlight_metrics).format(
                na_rep="⚠️ NaN (Error)"
            )
            st.dataframe(styled_table, width="stretch")

        # 4. Dynamic Plotting
        st.divider()
        st.header("📈 Visualization")

        pdf = filtered_df.to_pandas()
        plot_col1, plot_col2 = st.columns([1, 3])

        with plot_col1:
            st.subheader("Plot Settings")

            # Calculate default indices based on constants
            try:
                x_default_idx = numeric_cols.index(DEFAULT_X_AXIS)
            except ValueError:
                x_default_idx = 0

            try:
                y_default_idx = numeric_cols.index(DEFAULT_Y_AXIS)
            except ValueError:
                y_default_idx = min(1, len(numeric_cols) - 1)

            x_axis = st.selectbox("X-Axis", options=numeric_cols, index=x_default_idx)
            y_axis = st.selectbox("Y-Axis", options=numeric_cols, index=y_default_idx)

            color_options = [
                "model_type",
                "dataset_label",
                "experiment_date",
                "experiment_type",
            ] + numeric_cols
            color_by = st.selectbox("Color By", options=color_options, index=0)

        with plot_col2:
            is_numeric_color = color_by in numeric_cols
            color_shorthand = f"{color_by}:Q" if is_numeric_color else f"{color_by}:N"

            # In Altair, we convert date objects to ISO strings or handle them
            # as temporal
            # but for simple categorical coloring, :N works fine even with
            # date objects

            chart = (
                alt.Chart(pdf)
                .mark_circle(size=100)
                .encode(
                    x=alt.X(x_axis, scale=alt.Scale(zero=False)),
                    y=alt.Y(y_axis, scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        color_shorthand,
                        scale=alt.Scale(
                            scheme="viridis" if is_numeric_color else "tableau10"
                        ),
                    ),
                    tooltip=metadata_cols + numeric_cols,
                )
                .interactive()
                .properties(height=500)
            )
            st.altair_chart(chart, width="stretch")

        # 5. Topic–metadata alignment
        st.divider()
        st.header("🧭 Topic–Metadata Alignment")
        st.caption(
            "Adjusted mutual information between each document's final topic and "
            "each raw covariate (≈0 unrelated, 1 identical). Descriptive, not a "
            "quality score: models that take covariates as input can raise it by "
            "construction. HDBSCAN noise (topic −1) counts as its own topic. "
            "Older runs are filled from "
            "`results/derived/metadata_alignment_backfill.csv`."
        )
        render_alignment_section(filtered_df, alignment)

    with tab_qualitative:
        st.header("🔍 Qualitative Topic Analysis")

        if qual_df.is_empty():
            st.warning("No qualitative results found in `output/`.")
        else:
            # Filter qualitative results to match the current selection
            # We use multiple criteria to be robust against file renaming/aggregation
            active_datasets = filtered_df["dataset_label"].unique().to_list()
            active_model_types = filtered_df["model_type"].unique().to_list()
            active_timestamps = []
            if "timestamp" in filtered_df.columns:
                active_timestamps = (
                    filtered_df["timestamp"].unique().drop_nulls().to_list()
                )

            # Base filter: Match by Dataset and Model Type
            filtered_qual_df = qual_df.filter(
                (pl.col("dataset_label").is_in(active_datasets))
                & (pl.col("model_type").is_in(active_model_types))
            )

            # Refinement: If we have specific timestamps for the selected runs, use them
            if active_timestamps:
                ts_filtered = filtered_qual_df.filter(
                    pl.col("timestamp").is_in(active_timestamps)
                )
                # Only use timestamp filter if it doesn't result in an empty set
                # (helps handle cases with slightly mismatched timestamps or
                # missing data)
                if not ts_filtered.is_empty():
                    filtered_qual_df = ts_filtered

            if filtered_qual_df.is_empty():
                st.info("No qualitative data matches the current filters.")
            else:
                # 1. Keyword Search
                st.subheader("🔦 Keyword Search")
                search_query = st.text_input(
                    "Search for keywords in topic representations or "
                    "representative docs:",
                    placeholder="e.g., 'covid' or 'fake news'",
                )

                if search_query:
                    search_expr = pl.col("representation").str.contains(
                        search_query, literal=False
                    ) | pl.col("representative_docs").str.contains(
                        search_query, literal=False
                    )
                    search_results = filtered_qual_df.filter(search_expr)
                    st.write(
                        f"Found {len(search_results)} topics matching '{search_query}'."
                    )
                    st.dataframe(search_results.to_pandas(), width="stretch")

                st.divider()

                # 2. Side-by-Side Comparison
                st.subheader("⚖️ Side-by-Side Model Comparison")

                # Create a selection of unique model identifiers from the
                # filtered results
                # We'll use a combination of source_file and model_id to be unique
                filtered_qual_df = filtered_qual_df.with_columns(
                    pl.concat_str(
                        [pl.col("source_file"), pl.lit(" | "), pl.col("model_id")]
                    ).alias("unique_model_id")
                )
                model_options = (
                    filtered_qual_df["unique_model_id"].unique().sort().to_list()
                )

                col_left, col_right = st.columns(2)

                with col_left:
                    model_a = st.selectbox(
                        "Select Model A:", options=model_options, index=0
                    )
                    df_a = filtered_qual_df.filter(pl.col("unique_model_id") == model_a)
                    st.write(f"**Topics for {model_a.split('|')[-1].strip()}**")
                    st.dataframe(
                        df_a.select(
                            ["topic_id", "count", "name", "representation"]
                        ).to_pandas(),
                        width="stretch",
                        hide_index=True,
                    )

                with col_right:
                    # Default to second model if available
                    default_idx = 1 if len(model_options) > 1 else 0
                    model_b = st.selectbox(
                        "Select Model B:", options=model_options, index=default_idx
                    )
                    df_b = filtered_qual_df.filter(pl.col("unique_model_id") == model_b)
                    st.write(f"**Topics for {model_b.split('|')[-1].strip()}**")
                    st.dataframe(
                        df_b.select(
                            ["topic_id", "count", "name", "representation"]
                        ).to_pandas(),
                        width="stretch",
                        hide_index=True,
                    )

                st.divider()

                # 3. Detailed Topic Explorer
                st.subheader("🗺️ Detailed Topic Explorer")
                selected_model = st.selectbox(
                    "Select a model to explore its topics in detail:",
                    options=model_options,
                )

                model_detail_df = filtered_qual_df.filter(
                    pl.col("unique_model_id") == selected_model
                )

                topic_ids = model_detail_df["topic_id"].sort().to_list()
                selected_topic = st.selectbox("Select Topic ID:", options=topic_ids)

                topic_data = model_detail_df.filter(
                    pl.col("topic_id") == selected_topic
                ).to_dicts()[0]

                det_col1, det_col2 = st.columns([1, 2])

                with det_col1:
                    st.metric("Topic ID", topic_data["topic_id"])
                    st.metric("Document Count", topic_data["count"])
                    st.write("**Representation (c-TF-IDF words):**")
                    # Try to parse as JSON list if it looks like one, otherwise
                    # just show
                    try:
                        repr_words = json.loads(topic_data["representation"])
                        st.write(", ".join(repr_words))
                    except Exception:
                        st.write(topic_data["representation"])

                with det_col2:
                    st.write("**Representative Documents:**")
                    try:
                        docs = json.loads(topic_data["representative_docs"])
                        for i, doc in enumerate(docs):
                            with st.expander(f"Document {i + 1}", expanded=(i == 0)):
                                st.write(doc)
                    except Exception:
                        st.write(topic_data["representative_docs"])

    with tab_coverage:
        st.header("📋 Experiment Execution Coverage")
        st.write(
            "Track which experiments defined in `experiments/` have been executed "
            "and saved in `results/` across representation and dataset conditions."
        )

        cov_col1, cov_col2, cov_col3, cov_col4 = st.columns([2, 2, 2, 2])

        with cov_col1:
            all_cov_datasets = ["fed", "anes", "yelp", "trump", "gadarian"]
            selected_cov_datasets = st.multiselect(
                "Filter Datasets:",
                options=all_cov_datasets,
                default=[],
                key="cov_datasets",
                help="Leave empty to show all datasets.",
            )

        with cov_col2:
            status_filter = st.selectbox(
                "Filter Status:",
                options=[
                    "All",
                    "Fully Completed (3/3)",
                    "Partially Completed (1-2/3)",
                    "Has Errors",
                    "Not Run (0/3)",
                ],
                index=0,
                key="cov_status_filter",
            )

        with cov_col3:
            cov_search = st.text_input(
                "Search Experiment:",
                placeholder="e.g. aligned_umap",
                key="cov_search",
            )

        with cov_col4:
            st.write("")
            st.write("")
            include_archived = st.checkbox(
                "Include Archived",
                value=False,
                key="cov_include_archived",
                help="Include YAML files from experiments/archive/",
            )

        # Scan experiments & build coverage matrix
        discovered_exps = scan_experiment_configs(
            exp_dir=PROJECT_ROOT / "experiments",
            include_archived=include_archived,
        )

        cov_matrix = build_coverage_matrix(discovered_exps, all_results)
        if not cov_matrix.is_empty():
            cov_matrix = filter_catalog(annotate_coverage(cov_matrix, discovered_exps, catalog), **scope)
            cov_matrix = sort_catalog(cov_matrix, "experiment_name")

        if cov_matrix.is_empty():
            st.info("No experiment configurations match the catalog scope.")
        else:
            with st.expander("Completion by priority"):
                st.dataframe(
                    cov_matrix.group_by("priority").agg(
                        pl.len().alias("experiments"),
                        (pl.col("coverage_status") == "Fully Completed")
                        .sum()
                        .alias("fully_completed"),
                    ).sort("priority"),
                    hide_index=True,
                    width="stretch",
                )
            # Apply coverage filters
            filtered_matrix = cov_matrix

            if selected_cov_datasets:
                filtered_matrix = filtered_matrix.filter(
                    pl.col("dataset_label").is_in(selected_cov_datasets)
                )

            if status_filter == "Fully Completed (3/3)":
                filtered_matrix = filtered_matrix.filter(
                    pl.col("coverage_status") == "Fully Completed"
                )
            elif status_filter == "Partially Completed (1-2/3)":
                filtered_matrix = filtered_matrix.filter(
                    pl.col("coverage_status") == "Partially Completed"
                )
            elif status_filter == "Has Errors":
                filtered_matrix = filtered_matrix.filter(
                    pl.col("coverage_status").is_in(
                        ["Has Errors", "Completed with Errors"]
                    )
                    | pl.col("keep_rep_stopwords").str.contains("Error")
                    | pl.col("remove_rep_stopwords").str.contains("Error")
                    | pl.col("stemmed").str.contains("Error")
                )
            elif status_filter == "Not Run (0/3)":
                filtered_matrix = filtered_matrix.filter(
                    pl.col("coverage_status") == "Not Run"
                )

            if cov_search:
                filtered_matrix = filtered_matrix.filter(
                    pl.col("experiment_name").str.contains(cov_search, literal=False)
                )

            # Summary Metrics
            total_exps = len(filtered_matrix)
            full_completed = len(
                filtered_matrix.filter(pl.col("coverage_status") == "Fully Completed")
            )
            total_cell_runs = (
                filtered_matrix["completed_count"].sum() if total_exps > 0 else 0
            )
            total_possible_cells = total_exps * 3
            overall_pct = (
                round((total_cell_runs / total_possible_cells) * 100, 1)
                if total_possible_cells > 0
                else 0.0
            )
            missing_cells = total_possible_cells - total_cell_runs

            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.metric("Experiments Listed", total_exps)
            s_col2.metric("Fully Completed (3/3)", full_completed)
            s_col3.metric("Coverage Rate", f"{overall_pct}%")
            s_col4.metric("Missing Condition Runs", missing_cells)

            st.divider()

            # Target Environment Selection for SLURM runner
            env_choice = st.radio(
                "**Target Script Execution Location:**",
                options=[
                    "🖥️ Cluster (`./scripts/queue_exp.sh`)",
                    "💻 Local (`./scripts/pipelines/slurm/queue_exp.sh`)",
                ],
                index=0,
                horizontal=True,
                key="cov_env_choice",
                help=(
                    "Select whether commands should reference the cluster "
                    "script path or local pipeline path."
                ),
            )
            selected_script_path = (
                DEFAULT_SLURM_SCRIPT_CLUSTER
                if "Cluster" in env_choice
                else DEFAULT_SLURM_SCRIPT_LOCAL
            )

            # Batch Command Generator Expander
            expander_title = (
                f"⚡ Batch SLURM Commands for Missing Experiments "
                f"({missing_cells} missing in view)"
            )
            with st.expander(
                expander_title,
                expanded=(missing_cells > 0 and missing_cells <= 50),
            ):
                st.markdown(
                    "Generate batched commands for `queue_exp.sh` "
                    "to submit all missing or failed runs in the current view."
                )

                bg_col1, bg_col2, bg_col3 = st.columns(3)
                with bg_col1:
                    inc_not_run = st.checkbox(
                        "Include Not Run (❌)",
                        value=True,
                        key="cov_batch_not_run",
                    )
                    inc_errors = st.checkbox(
                        "Include Errors / NaNs (❌)",
                        value=True,
                        key="cov_batch_errors",
                    )
                with bg_col2:
                    inc_dry_runs = st.checkbox(
                        "Include Dry Runs (⚠️)",
                        value=False,
                        key="cov_batch_dry_runs",
                    )
                    batch_dry_run = st.checkbox(
                        "SLURM Preview / Dry Run (`-n`)",
                        value=False,
                        key="cov_batch_slurm_dry",
                    )
                with bg_col3:
                    batch_auto_yes = st.checkbox(
                        "Auto-Confirm Jobs (`-y`)",
                        value=False,
                        key="cov_batch_slurm_yes",
                    )

                grouped_slurm = generate_grouped_slurm_commands(
                    coverage_df=filtered_matrix,
                    script_path=selected_script_path,
                    include_not_run=inc_not_run,
                    include_errors=inc_errors,
                    include_dry_runs=inc_dry_runs,
                    dry_run=batch_dry_run,
                    auto_yes=batch_auto_yes,
                )

                all_cmds = []
                for cond_k in [
                    "keep_rep_stopwords",
                    "remove_rep_stopwords",
                    "stemmed",
                ]:
                    all_cmds.extend(grouped_slurm.get(cond_k, []))

                if not all_cmds:
                    st.success(
                        "🎉 No missing experiments match the selected "
                        "criteria in this view!"
                    )
                else:
                    st.write(f"**Total SLURM Dispatch Commands:** `{len(all_cmds)}`")

                    n_keep = len(grouped_slurm.get("keep_rep_stopwords", []))
                    n_remove = len(grouped_slurm.get("remove_rep_stopwords", []))
                    n_stem = len(grouped_slurm.get("stemmed", []))

                    # Display categorized tabs
                    tab_keep, tab_remove, tab_stem, tab_all = st.tabs(
                        [
                            f"Keep Stopwords ({n_keep})",
                            f"Remove Stopwords ({n_remove})",
                            f"Stemmed ({n_stem})",
                            f"All Commands Combined ({len(all_cmds)})",
                        ]
                    )

                    with tab_keep:
                        keep_cmds = grouped_slurm.get("keep_rep_stopwords", [])
                        if keep_cmds:
                            st.code("\n".join(keep_cmds), language="bash")
                        else:
                            st.info("No missing experiments for Keep Stopwords.")

                    with tab_remove:
                        remove_cmds = grouped_slurm.get("remove_rep_stopwords", [])
                        if remove_cmds:
                            st.code("\n".join(remove_cmds), language="bash")
                        else:
                            st.info("No missing experiments for Remove Stopwords.")

                    with tab_stem:
                        stem_cmds = grouped_slurm.get("stemmed", [])
                        if stem_cmds:
                            st.code("\n".join(stem_cmds), language="bash")
                        else:
                            st.info("No missing experiments for Stemmed.")

                    with tab_all:
                        all_text = "\n".join(all_cmds)
                        st.code(all_text, language="bash")
                        st.download_button(
                            label="📥 Download Commands Script (`run_missing.sh`)",
                            data=all_text,
                            file_name="run_missing.sh",
                            mime="text/x-sh",
                            key="cov_download_sh",
                        )

            st.divider()

            # Prepare Display Table
            display_cols = [
                "family", "role", "priority", "baseline_id",
                "dataset_label",
                "experiment_name",
                "keep_rep_stopwords",
                "remove_rep_stopwords",
                "stemmed",
                "coverage_score",
            ]
            pdf_cov = filtered_matrix.select(display_cols).to_pandas()
            pdf_cov = pdf_cov.rename(
                columns={
                    "dataset_label": "Dataset",
                    "experiment_name": "Experiment Name",
                    "keep_rep_stopwords": "Keep Stopwords",
                    "remove_rep_stopwords": "Remove Stopwords",
                    "stemmed": "Stemmed",
                    "coverage_score": "Score",
                }
            )

            # Cell styling helper
            def style_cell(val):
                if isinstance(val, str):
                    if val.startswith("✅"):
                        return (
                            "background-color: #D4EDDA; "
                            "color: #155724; font-weight: bold;"
                        )
                    elif "Error" in val:
                        return (
                            "background-color: #F8D7DA; "
                            "color: #721C24; font-weight: bold;"
                        )
                    elif val.startswith("⚠️"):
                        return (
                            "background-color: #FFF3CD; "
                            "color: #856404; font-weight: bold;"
                        )
                    elif val.startswith("❌"):
                        return "background-color: #F8D7DA; color: #721C24;"
                return ""

            style_fn = getattr(pdf_cov.style, "map", None) or getattr(
                pdf_cov.style, "applymap"
            )
            styled_cov_df = style_fn(style_cell)
            st.dataframe(styled_cov_df, width="stretch", hide_index=True)

            # Detail Expander
            st.divider()
            st.subheader("🔍 Experiment Details Inspector")
            exp_names = filtered_matrix["experiment_name"].to_list()
            if exp_names:
                selected_exp_name = st.selectbox(
                    "Select Experiment to Inspect Details:", options=exp_names
                )
                exp_detail_row = filtered_matrix.filter(
                    pl.col("experiment_name") == selected_exp_name
                ).to_dicts()[0]

                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.write("**Configuration Files:**")
                    std_path = exp_detail_row.get("yaml_standard") or "N/A"
                    stem_path = exp_detail_row.get("yaml_stemmed") or "N/A"
                    st.write(f"- Standard YAML: `{std_path}`")
                    st.write(f"- Stemmed YAML: `{stem_path}`")
                    st.write(f"- Dataset: `{exp_detail_row['dataset_label']}`")
                    st.write(
                        f"- Is Archived: `{exp_detail_row.get('is_archived', False)}`"
                    )

                with d_col2:
                    st.write("**Condition Details:**")
                    run_details = json.loads(
                        exp_detail_row.get("run_details_json", "{}")
                    )
                    for cond_key, cond_title in [
                        ("keep_rep_stopwords", "Keep Stopwords"),
                        ("remove_rep_stopwords", "Remove Stopwords"),
                        ("stemmed", "Stemmed"),
                    ]:
                        c_info = run_details.get(cond_key, {})
                        status = c_info.get("status", "Not Run")
                        cnt = c_info.get("count", 0)
                        err_cnt = c_info.get("error_count", 0)
                        dry_cnt = c_info.get("dry_run_count", 0)
                        if err_cnt > 0:
                            nan_list = ", ".join(c_info.get("nan_metrics", []))
                            st.write(
                                f"- **{cond_title}**: {status} "
                                f"({cnt} valid runs, :red[{err_cnt} with NaNs/Errors], "
                                f"{dry_cnt} dry runs)"
                            )
                            if nan_list:
                                st.write(f"  - *Metrics with NaNs*: `{nan_list}`")
                        else:
                            st.write(
                                f"- **{cond_title}**: {status} "
                                f"({cnt} full runs, {dry_cnt} dry runs)"
                            )

                # Individual Experiment SLURM Run Commands
                st.markdown(
                    "##### ⚡ Run Commands for this Experiment (`queue_exp.sh`)"
                )
                st.caption(
                    "Click the copy icon on the top right of each command block "
                    "to copy to your clipboard:"
                )

                model_name = extract_model_name(
                    selected_exp_name, exp_detail_row["dataset_label"]
                )

                for cond_key, cond_title in [
                    ("keep_rep_stopwords", "Keep Stopwords (Unstemmed)"),
                    ("remove_rep_stopwords", "Remove Stopwords (Unstemmed)"),
                    ("stemmed", "Stemmed Dataset"),
                ]:
                    status_text = exp_detail_row.get(cond_key, "❌ Not Run")
                    cmd_str = generate_slurm_command(
                        dataset=exp_detail_row["dataset_label"],
                        model=model_name,
                        condition=cond_key,
                        script_path=selected_script_path,
                    )

                    cond_badge = (
                        "✅"
                        if status_text.startswith("✅")
                        else "⚠️"
                        if status_text.startswith("⚠️")
                        else "❌"
                    )
                    st.write(f"**{cond_badge} {cond_title}** — Status: `{status_text}`")
                    st.code(cmd_str, language="bash")

    with tab_paper_tables:
        st.header("📑 Publication & Paper Results Tables")
        st.markdown(
            "Generate, inspect, and export publication-ready LaTeX tables, "
            "Markdown reports, and CSV datasets matching the experimental "
            "configurations and statistical tests in the paper."
        )

        table_choice = st.radio(
            "**Select Paper Table Type:**",
            options=[
                "🏆 Benchmark Results (Best Models / Seed Averages)",
                "📐 Demšar All-vs-All Statistical Comparison",
                "🔬 Demšar Condition Sensitivity Deltas",
                "🛑 Representation Stopword Impact",
                "📉 HDBSCAN Noise & Outlier Coverage",
                "📂 Pre-Generated Tables Archive",
            ],
            horizontal=True,
            key="pub_table_choice",
        )

        st.divider()

        # ----------------------------------------------------------------------
        # 1. Benchmark Results Table
        # ----------------------------------------------------------------------
        if "🏆 Benchmark" in table_choice:
            st.subheader("🏆 Model Performance Benchmark Table")
            st.caption(
                "Consolidated model comparisons across metrics ($C_{\\text{UMass}}$, "
                "$C_v$, $C_{npmi}$, IRBO, Topic Diversity) with Olympic 3-tier "
                "highlights (🥇 Gold, 🥈 Silver, 🥉 Bronze)."
            )

            bm_c1, bm_c2, bm_c3 = st.columns(3)
            with bm_c1:
                available_ds = sorted(df["dataset_label"].unique().to_list())
                default_ds_idx = (
                    available_ds.index("fed") if "fed" in available_ds else 0
                )
                bm_dataset = st.selectbox(
                    "Dataset:",
                    options=available_ds,
                    index=default_ds_idx,
                    key="bm_ds",
                )
            with bm_c2:
                cond_map = {
                    "Standard (Stopwords Removed)": "remove_rep_stopwords",
                    "Stemmed": "stemmed",
                    "Keep Stopwords (No Removal)": "keep_rep_stopwords",
                    "All Conditions Combined": "all",
                }
                bm_cond_label = st.selectbox(
                    "Preprocessing Condition:",
                    options=list(cond_map.keys()),
                    index=0,
                    key="bm_cond",
                )
                bm_cond = cond_map[bm_cond_label]
            with bm_c3:
                mode_options = [
                    "Average Across Seeds (Mean ± SD)",
                    "Best Run per Model Type",
                    "All Configurations Dump",
                ]
                bm_mode = st.selectbox(
                    "Aggregation Mode:",
                    options=mode_options,
                    index=0,
                    key="bm_mode",
                )
                is_average = bm_mode == "Average Across Seeds (Mean ± SD)"
                is_dump = bm_mode == "All Configurations Dump"

            with st.expander("⚙️ Algorithmic Exclusions & Options", expanded=False):
                bm_opt1, bm_opt2, bm_opt3, bm_opt4 = st.columns(4)
                with bm_opt1:
                    ex_clust = st.checkbox(
                        "Exclude K-Means", value=False, key="bm_ex_kmeans"
                    )
                with bm_opt2:
                    ex_pca = st.checkbox("Exclude PCA", value=False, key="bm_ex_pca")
                with bm_opt3:
                    merge_info0 = st.checkbox(
                        "Merge info0 variants",
                        value=False,
                        key="bm_merge_info0",
                    )
                with bm_opt4:
                    suppress_nulls = st.checkbox(
                        "Suppress Nulls", value=False, key="bm_suppress_nulls"
                    )

            ex_clust_list = kmeans_algorithms(df) if ex_clust else None
            ex_dim_list = ("pca",) if ex_pca else None

            bm_results = get_cached_best_models(
                df=df,
                dataset=bm_dataset,
                condition=bm_cond,
                exclude_clustering=ex_clust_list,
                exclude_dim_red=ex_dim_list,
                dump=is_dump,
                average=is_average,
                merge_info0=merge_info0,
                suppress_nulls=suppress_nulls,
            )

            if not bm_results:
                st.warning(
                    f"No valid metric results found for dataset '{bm_dataset}' "
                    f"with condition '{bm_cond_label}'."
                )
            else:
                table_data = generate_best_models_table_data(
                    results=bm_results,
                    dump=is_dump,
                    average=is_average,
                )
                latex_code = generate_best_models_latex_table(
                    results=bm_results,
                    dataset=bm_dataset,
                    dump=is_dump,
                    average=is_average,
                    result_type=bm_cond if bm_cond != "all" else None,
                )
                markdown_code = generate_best_models_markdown_table(
                    results=bm_results,
                    dataset=bm_dataset,
                    dump=is_dump,
                    average=is_average,
                    result_type=bm_cond_label,
                )
                csv_data = table_data["display_df"].to_csv(index=False)

                st.dataframe(table_data["styler"], width="stretch", hide_index=True)
                st.caption(
                    "Cell highlights: "
                    ":orange-background[**🥇 1st Best** (Gold)] | "
                    ":gray-background[**🥈 2nd Best** (Silver)] | "
                    "**🥉 3rd Best** (Bronze)"
                )

                mode_slug = "avg" if is_average else ("dump" if is_dump else "best")
                file_slug = f"best_models_{bm_dataset}_{bm_cond}_{mode_slug}"
                render_table_export_bar(
                    latex_content=latex_code,
                    csv_content=csv_data,
                    markdown_content=markdown_code,
                    file_slug=file_slug,
                    key_prefix="bm",
                )

                with st.expander("📄 View Raw LaTeX Code", expanded=False):
                    st.code(latex_code, language="latex")
                with st.expander("📝 View Markdown Code", expanded=False):
                    st.code(markdown_code, language="markdown")

        # ----------------------------------------------------------------------
        # 2. Demšar All-vs-All Statistical Comparison
        # ----------------------------------------------------------------------
        elif "📐 Demšar All-vs-All" in table_choice:
            st.subheader("📐 Demšar (2006) All-vs-All Statistical Ranking")
            st.caption(
                "Non-parametric multi-algorithm comparisons via Friedman / "
                "Iman-Davenport omnibus tests, Nemenyi Critical Difference (CD), "
                "Demšar cliques, and Holm-Bonferroni pairwise testing."
            )

            da_c1, da_c2, da_c3, da_c4 = st.columns(4)
            with da_c1:
                available_ds = sorted(df["dataset_label"].unique().to_list())
                da_datasets = st.multiselect(
                    "Datasets:",
                    options=available_ds,
                    default=available_ds,
                    key="da_ds",
                )
            with da_c2:
                cond_map = {
                    "Standard (Stopwords Removed)": "remove_rep_stopwords",
                    "Stemmed": "stemmed",
                    "Keep Stopwords (No Removal)": "keep_rep_stopwords",
                }
                da_cond_label = st.selectbox(
                    "Condition:",
                    options=list(cond_map.keys()),
                    index=0,
                    key="da_cond",
                )
                da_cond = cond_map[da_cond_label]
            with da_c3:
                metric_options = [
                    "u_mass",
                    "c_v",
                    "c_npmi",
                    "irbo",
                    "topic_diversity",
                ]
                da_metric = st.selectbox(
                    "Metric to Inspect:",
                    options=metric_options,
                    index=0,
                    key="da_metric",
                )
            with da_c4:
                da_alpha = st.select_slider(
                    "Alpha (α):",
                    options=[0.01, 0.05, 0.10],
                    value=0.05,
                    key="da_alpha",
                )

            with st.expander("⚙️ Exclusions & Additional Matrices", expanded=False):
                da_opt1, da_opt2, da_opt3, da_opt4 = st.columns(4)
                with da_opt1:
                    da_ex_kmeans = st.checkbox(
                        "Exclude K-Means", value=False, key="da_ex_kmeans"
                    )
                with da_opt2:
                    da_ex_pca = st.checkbox("Exclude PCA", value=False, key="da_ex_pca")
                with da_opt3:
                    da_merge_info0 = st.checkbox(
                        "Merge info0 variants",
                        value=False,
                        key="da_merge_info0",
                    )
                with da_opt4:
                    da_inc_deltas = st.checkbox(
                        "Include Pairwise Deltas",
                        value=True,
                        key="da_inc_deltas",
                    )

            if not da_datasets or len(da_datasets) < 2:
                st.warning(
                    "Demšar (2006) all-vs-all testing compares algorithms across "
                    "multiple datasets. Please select at least 2 datasets to compute "
                    "the Friedman/Iman-Davenport omnibus test and Nemenyi "
                    "critical differences."
                )
            else:
                ex_clust_list = kmeans_algorithms(df) if da_ex_kmeans else None
                ex_dim_list = ("pca",) if da_ex_pca else None

                da_results = get_cached_demsar_all_vs_all(
                    df=df,
                    datasets=tuple(da_datasets),
                    condition=da_cond,
                    metrics=tuple(metric_options),
                    alpha=da_alpha,
                    exclude_clustering=ex_clust_list,
                    exclude_dim_red=ex_dim_list,
                    merge_info0=da_merge_info0,
                )

                metrics_dict = da_results.get("metrics", {})
                if not metrics_dict or da_metric not in metrics_dict:
                    st.warning(
                        "Could not compute Demšar all-vs-all ranking for "
                        f"'{da_metric}'. Ensure multiple model configurations "
                        "and evaluation blocks exist."
                    )
                else:
                    m_data = metrics_dict[da_metric]
                    omnibus = m_data.get("omnibus", {})
                    f_stat = omnibus.get("f_f", 0.0)
                    p_val = omnibus.get("p_f_f", 1.0)
                    cd = m_data.get("critical_difference", 0.0)
                    is_sig = p_val < da_alpha

                    o_col1, o_col2, o_col3, o_col4 = st.columns(4)
                    o_col1.metric("Omnibus F_F", f"{f_stat:.3f}")
                    o_col2.metric("p-value", f"{p_val:.4f}")
                    o_col3.metric("Critical Difference (CD)", f"{cd:.3f}")
                    o_col4.metric(
                        "Significance Status",
                        "Statistically Significant" if is_sig else "Not Significant",
                    )

                    st.markdown(f"#### Model Ranking Summary: `{da_metric.upper()}`")
                    df_summary = m_data["summary_table"].to_pandas()
                    st.dataframe(df_summary, width="stretch", hide_index=True)

                    if da_inc_deltas and "pairwise_delta_matrix" in m_data:
                        st.markdown(
                            f"#### Pairwise Delta Matrix: `{da_metric.upper()}`"
                        )
                        df_matrix = m_data["pairwise_delta_matrix"].to_pandas()
                        st.dataframe(
                            style_demsar_pairwise_matrix(df_matrix),
                            width="stretch",
                            hide_index=True,
                        )

                    ds_slug = "_".join(da_datasets).lower()
                    ds_label = ", ".join(da_datasets).upper()
                    latex_table = generate_demsar_all_vs_all_latex_table(
                        da_results, metric=da_metric, dataset_label=ds_label
                    )
                    if da_inc_deltas:
                        delta_latex = generate_pairwise_delta_latex_matrix(
                            da_results, metric=da_metric, dataset_label=ds_label
                        )
                        latex_table += "\n\n" + delta_latex

                    md_report = generate_demsar_all_vs_all_report(
                        da_results,
                        dataset_label=ds_label,
                        include_deltas=da_inc_deltas,
                    )
                    csv_data = df_summary.to_csv(index=False)

                    file_slug = f"demsar_all_vs_all_{ds_slug}_{da_metric}_{da_cond}"
                    render_table_export_bar(
                        latex_content=latex_table,
                        csv_content=csv_data,
                        markdown_content=md_report,
                        file_slug=file_slug,
                        key_prefix="da",
                    )

                    with st.expander("📄 View LaTeX Code", expanded=False):
                        st.code(latex_table, language="latex")
                    with st.expander("📝 View Full Markdown Report", expanded=False):
                        st.code(md_report, language="markdown")

        # ----------------------------------------------------------------------
        # 3. Demšar Condition Sensitivity Deltas
        # ----------------------------------------------------------------------
        elif "🔬 Demšar Condition" in table_choice:
            st.subheader("🔬 Demšar Condition Sensitivity Deltas")
            st.caption(
                "Model-by-metric performance change (Alternative - Default "
                "Standard) across benchmark datasets. Evaluated using paired exact "
                "Wilcoxon signed-rank tests across datasets with "
                "Holm-Bonferroni FWER control."
            )

            dd_c1, dd_c2, dd_c3, dd_c4 = st.columns(4)
            with dd_c1:
                available_ds = sorted(df["dataset_label"].unique().to_list())
                dd_datasets = st.multiselect(
                    "Datasets:", options=available_ds, default=available_ds, key="dd_ds"
                )
            with dd_c2:
                dd_cond_map = {
                    "Stemmed vs Standard": "stemmed",
                    "Keep Stopwords vs Remove": "keep_rep_stopwords",
                }
                dd_cond_label = st.selectbox(
                    "Comparison Condition:",
                    options=list(dd_cond_map.keys()),
                    index=0,
                    key="dd_cond",
                )
                dd_cond = dd_cond_map[dd_cond_label]
            with dd_c3:
                dd_alpha = st.select_slider(
                    "Alpha (α):",
                    options=[0.01, 0.05, 0.10],
                    value=0.10,
                    key="dd_alpha",
                )
            with dd_c4:
                dd_corr = st.selectbox(
                    "Correction Scope:",
                    options=["per_metric", "table", "none"],
                    index=0,
                    key="dd_corr",
                )

            with st.expander("⚙️ Exclusions", expanded=False):
                dd_opt1, dd_opt2, dd_opt3 = st.columns(3)
                with dd_opt1:
                    dd_ex_kmeans = st.checkbox(
                        "Exclude K-Means", value=False, key="dd_ex_kmeans"
                    )
                with dd_opt2:
                    dd_ex_pca = st.checkbox("Exclude PCA", value=False, key="dd_ex_pca")
                with dd_opt3:
                    dd_merge_info0 = st.checkbox(
                        "Merge info0 variants",
                        value=False,
                        key="dd_merge_info0",
                    )

            if not dd_datasets or len(dd_datasets) < 2:
                st.warning(
                    "Please select at least 2 datasets to evaluate paired "
                    "Wilcoxon tests across datasets."
                )
            else:
                ex_clust_list = kmeans_algorithms(df) if dd_ex_kmeans else None
                ex_dim_list = ("pca",) if dd_ex_pca else None

                delta_results = get_cached_demsar_delta(
                    df=df,
                    datasets=tuple(dd_datasets),
                    condition=dd_cond,
                    alpha=dd_alpha,
                    correction=dd_corr,
                    exclude_clustering=ex_clust_list,
                    exclude_dim_red=ex_dim_list,
                    merge_info0=dd_merge_info0,
                )

                df_summary = delta_results.get("df_summary")
                if df_summary is None or df_summary.is_empty():
                    st.warning(
                        f"No paired results found for datasets {dd_datasets} "
                        f"comparing '{dd_cond_label}'. Ensure both Default and "
                        "Alternative runs exist."
                    )
                else:
                    pdf_summary = df_summary.to_pandas()
                    st.dataframe(
                        style_demsar_delta_dataframe(pdf_summary),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "Cell format: `Mean Δ ± SD` | "
                        ":green-background[**Green**: Improvement (Δ > 0)] | "
                        ":red-background[**Red**: Decline (Δ < 0)] | "
                        "**\\***: Statistically significant "
                        "($p_{\\text{adj}} < \\alpha$)"
                    )

                    dd_label = ", ".join(dd_datasets).upper()
                    dd_slug = "_".join(dd_datasets).lower()
                    latex_code = generate_demsar_delta_latex_table(
                        delta_results,
                        dataset_label=dd_label,
                        condition_name=dd_cond.capitalize(),
                    )
                    markdown_code = generate_demsar_delta_markdown_table(
                        delta_results,
                        dataset_label=dd_label,
                        condition_name=dd_cond.capitalize(),
                    )
                    csv_data = pdf_summary.to_csv(index=False)

                    file_slug = f"demsar_delta_{dd_slug}_{dd_cond}"
                render_table_export_bar(
                    latex_content=latex_code,
                    csv_content=csv_data,
                    markdown_content=markdown_code,
                    file_slug=file_slug,
                    key_prefix="dd",
                )

                with st.expander(
                    "🔍 Detailed Test Statistics (Raw p-values, W-stats)",
                    expanded=False,
                ):
                    df_details = delta_results.get("df_details")
                    if df_details is not None and not df_details.is_empty():
                        st.dataframe(
                            df_details.to_pandas(),
                            width="stretch",
                            hide_index=True,
                        )

                with st.expander("📄 View LaTeX Code", expanded=False):
                    st.code(latex_code, language="latex")
                with st.expander("📝 View Markdown Code", expanded=False):
                    st.code(markdown_code, language="markdown")

        # ----------------------------------------------------------------------
        # 4. Representation Stopword Impact
        # ----------------------------------------------------------------------
        elif "🛑 Representation" in table_choice:
            st.subheader("🛑 Representation Stopword Impact")
            st.caption(
                "Metric differences (Mean ± SD) between representation stopwords "
                "removed vs. kept."
            )

            sw_c1, sw_c2, sw_c3 = st.columns(3)
            with sw_c1:
                available_ds = sorted(df["dataset_label"].unique().to_list())
                sw_dataset = st.selectbox(
                    "Dataset:", options=available_ds, index=0, key="sw_ds"
                )
            with sw_c2:
                sw_ex_kmeans = st.checkbox(
                    "Exclude K-Means", value=False, key="sw_ex_kmeans"
                )
            with sw_c3:
                sw_merge_info0 = st.checkbox(
                    "Merge info0 variants",
                    value=False,
                    key="sw_merge_info0",
                )

            ex_clust_list = kmeans_algorithms(df) if sw_ex_kmeans else None

            sw_results = get_cached_stopword_impact(
                df=df,
                dataset=sw_dataset,
                exclude_clustering=ex_clust_list,
                exclude_dim_red=(),
                merge_info0=sw_merge_info0,
            )

            if not sw_results:
                st.warning(
                    f"No paired stopword removal data found for dataset '{sw_dataset}'."
                )
            else:
                sw_data = generate_stopword_impact_table_data(sw_results)
                st.dataframe(sw_data["styler"], width="stretch", hide_index=True)
                st.caption(
                    ":green-background[**Green**: Improvement (Δ > 0)] | "
                    ":red-background[**Red**: Decline (Δ < 0)]"
                )

                latex_code = generate_stopword_impact_latex_table(
                    sw_results, dataset=sw_dataset
                )
                markdown_code = generate_stopword_impact_markdown_table(
                    sw_results, dataset=sw_dataset
                )
                csv_data = sw_data["display_df"].to_csv(index=False)

                file_slug = f"stopword_impact_{sw_dataset}"
                render_table_export_bar(
                    latex_content=latex_code,
                    csv_content=csv_data,
                    markdown_content=markdown_code,
                    file_slug=file_slug,
                    key_prefix="sw",
                )

                with st.expander("📄 View LaTeX Code", expanded=False):
                    st.code(latex_code, language="latex")
                with st.expander("📝 View Markdown Code", expanded=False):
                    st.code(markdown_code, language="markdown")

        # ----------------------------------------------------------------------
        # 5. HDBSCAN Noise & Outlier Coverage
        # ----------------------------------------------------------------------
        elif "📉 HDBSCAN" in table_choice:
            st.subheader("📉 HDBSCAN Noise & Outlier Coverage")
            st.caption(
                "Noise document count and percentage coverage assigned to the "
                "outlier cluster (-1) across seeds."
            )

            nc_c1, nc_c2, nc_c3 = st.columns(3)
            with nc_c1:
                available_ds = ["all"] + sorted(df["dataset_label"].unique().to_list())
                nc_dataset = st.selectbox(
                    "Dataset:", options=available_ds, index=0, key="nc_ds"
                )
            with nc_c2:
                cond_opts = [
                    "all",
                    "remove_rep_stopwords",
                    "keep_rep_stopwords",
                    "stemmed",
                ]
                nc_cond = st.selectbox(
                    "Condition:", options=cond_opts, index=0, key="nc_cond"
                )
            with nc_c3:
                nc_merge_info0 = st.checkbox(
                    "Merge info0 variants",
                    value=False,
                    key="nc_merge_info0",
                )

            df_noise = get_cached_noise_coverage(
                df=df,
                dataset=nc_dataset,
                condition=nc_cond,
                merge_info0=nc_merge_info0,
            )

            if df_noise.is_empty():
                st.warning("No HDBSCAN runs found matching criteria.")
            else:
                pdf_noise = df_noise.to_pandas()
                st.dataframe(pdf_noise, width="stretch", hide_index=True)

                latex_code = generate_noise_coverage_latex_table(
                    df_noise, result_type=nc_cond if nc_cond != "all" else None
                )
                markdown_code = generate_noise_coverage_markdown_table(df_noise)
                csv_data = pdf_noise.to_csv(index=False)

                file_slug = f"noise_coverage_{nc_dataset}_{nc_cond}"
                render_table_export_bar(
                    latex_content=latex_code,
                    csv_content=csv_data,
                    markdown_content=markdown_code,
                    file_slug=file_slug,
                    key_prefix="nc",
                )

                with st.expander("📄 View LaTeX Code", expanded=False):
                    st.code(latex_code, language="latex")
                with st.expander("📝 View Markdown Code", expanded=False):
                    st.code(markdown_code, language="markdown")

        # ----------------------------------------------------------------------
        # 6. Pre-Generated Tables Archive
        # ----------------------------------------------------------------------
        elif "📂 Pre-Generated" in table_choice:
            st.subheader("📂 Pre-Generated Paper Tables Archive")
            st.caption(
                "Browse, preview, and download existing table artifacts saved "
                "in the `tables/` repository."
            )

            if not TABLES_DIR.exists():
                st.info("The `tables/` directory does not exist yet.")
            else:
                saved_files = sorted(
                    [
                        f.name
                        for f in TABLES_DIR.glob("*.*")
                        if f.suffix in [".tex", ".md", ".csv"] and f.name != ".gitkeep"
                    ]
                )
                if not saved_files:
                    st.info("No table files found in `tables/`.")
                else:
                    af_c1, af_c2 = st.columns([3, 1])
                    with af_c1:
                        sel_file = st.selectbox(
                            "Select saved table file:",
                            options=saved_files,
                            key="archive_sel",
                        )
                    with af_c2:
                        st.write("")
                        st.write("")
                        file_path = TABLES_DIR / sel_file
                        file_data = file_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                        mime_map = {
                            ".tex": "text/x-tex",
                            ".md": "text/markdown",
                            ".csv": "text/csv",
                        }
                        st.download_button(
                            label=f"📥 Download `{sel_file}`",
                            data=file_data,
                            file_name=sel_file,
                            mime=mime_map.get(file_path.suffix, "text/plain"),
                            key="archive_dl",
                            use_container_width=True,
                        )

                    st.markdown(f"**File**: `{sel_file}` ({len(file_data)} bytes)")
                    if sel_file.endswith(".md"):
                        st.markdown(file_data)
                    elif sel_file.endswith(".csv"):
                        try:
                            csv_df = pl.read_csv(file_path)
                            st.dataframe(
                                csv_df.to_pandas(),
                                width="stretch",
                                hide_index=True,
                            )
                        except Exception:
                            st.code(file_data, language="text")
                    elif sel_file.endswith(".tex"):
                        st.code(file_data, language="latex")


if __name__ == "__main__":
    main()
