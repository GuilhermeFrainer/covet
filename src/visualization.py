from pathlib import Path

import pandas as pd
import plotly.express as px
import polars as pl

# Mapping for prettier metric names
METRIC_DISPLAY_MAP = {
    "u_mass": "C<sub>UMass</sub>",
    "c_v": "C<sub>v</sub>",
    "c_npmi": "C<sub>npmi</sub>",
    "irbo": "IRBO",
    "topic_diversity": "Topic Diversity",
}

# Mapping for prettier model names (Plotly version)
MODEL_RENAME_MAP = {
    "append_umap": "Naive",
    "mv_co_reg_spectral": "{model_name}<sub>1</sub>",
    "mv_co_reg_spectral_info0": "{model_name}<sub>1-info0</sub>",
    "baseline": "BERTopic<sub>1</sub>",
    "umap_spectral": "BERTopic<sub>2</sub>",
    "mv_spectral": "{model_name}<sub>2</sub>",
    "mv_spectral_info0": "{model_name}<sub>2-info0</sub>",
    "aligned_umap": "{model_name}<sub>3</sub>",
    "mv_hdbscan": "{model_name}<sub>7</sub>",
    "feature_stacking_hdbscan": "{model_name}<sub>8</sub>",
    "stacked_hdbscan": "{model_name}<sub>8</sub>",
    "stm": "STM",
}


def prepare_plot_data(
    results: dict[str, pl.DataFrame], dump: bool, model_name: str = "CAST"
):
    """Common data preparation for Plotly visualizations."""
    id_col = "best_model_name" if dump else "model_type"
    rows = []

    # Map for formatting model names
    rename_map = {
        k: v.format(model_name=model_name) for k, v in MODEL_RENAME_MAP.items()
    }

    for metric, metric_df in results.items():
        display_metric = METRIC_DISPLAY_MAP.get(metric, metric)
        for row in metric_df.iter_rows(named=True):
            model_id = row[id_col]
            model_type = row["model_type"]

            # Legend label should be the "Type" name, not the individual run name
            legend_label = rename_map.get(model_type, model_type.replace("_", " "))

            # Specific display name for this individual line
            display_model = rename_map.get(model_id, model_id.replace("_", " "))

            # Categorize as "Ours" vs "Baseline"
            is_ours = model_id.startswith("mv_") or model_id == "aligned_umap"
            group = "Ours" if is_ours else "Baseline"

            # Check for info0
            is_info0 = "_info0" in model_id

            rows.append(
                {
                    "Model": display_model,
                    "LegendLabel": legend_label,
                    "ModelType": model_type,
                    "Metric": display_metric,
                    "Value": row["max_value"],
                    "RawMetric": metric,
                    "Group": group,
                    "IsInfo0": is_info0,
                }
            )

    df = pd.DataFrame(rows)

    # Apply Min-Max Normalization per Metric
    df["NormalizedValue"] = df.groupby("Metric")["Value"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 1.0
    )

    return df


def get_color_map(df: pd.DataFrame, model_name: str = "CAST", only_info0: bool = False):
    """Generates a color map with explicit paired colors for numbered labels."""
    color_map = {}

    # Explicit mapping based on numbered labels (with <sub> tags)
    # Pairs: (Dark, Light) -> (Info0, Regular)
    mapping = {
        f"{model_name}<sub>1</sub>": ("#e31a1c", "#fb9a99"),  # Red / Light Red
        f"{model_name}<sub>2</sub>": ("#ff7f00", "#fdbf6f"),  # Orange / Light Orange
        f"{model_name}<sub>3</sub>": ("#e7298a", "#df65b0"),  # Pink / Light Pink
        "BERTopic<sub>1</sub>": ("#1f78b4", "#a6cee3"),  # Blue / Light Blue
        "BERTopic<sub>2</sub>": ("#008080", "#40e0d0"),  # Teal / Turquoise
        "Naive": ("#33a02c", "#b2df8a"),  # Green / Light Green
        "STM": ("#984ea3", "#decbe4"),  # Purple / Light Purple
    }

    labels = sorted(df["LegendLabel"].unique())
    for label in labels:
        # Determine base label and if it's info0
        is_info0 = "info0" in label.lower()

        # If we are in "only-info0" mode, force info0 colors for our spectral models
        if only_info0 and label in [
            f"{model_name}<sub>1</sub>",
            f"{model_name}<sub>2</sub>",
        ]:
            is_info0 = True

        # Strip info0 to find the base pair
        base = label.replace("-info0", "")

        if base in mapping:
            pair = mapping[base]
            color_map[label] = pair[0] if is_info0 else pair[1]
        else:
            # Fallback
            color_map[label] = "#7f7f7f"

    return color_map


def get_legend_order(model_name: str = "CAST"):
    """Returns the standardized legend order with <sub> tags."""
    return [
        f"{model_name}<sub>1</sub>",
        f"{model_name}<sub>1-info0</sub>",
        f"{model_name}<sub>2</sub>",
        f"{model_name}<sub>2-info0</sub>",
        f"{model_name}<sub>3</sub>",
        "Naive",
        "BERTopic<sub>1</sub>",
        "BERTopic<sub>2</sub>",
        "STM",
    ]


def clean_legend(fig):
    """Removes 'Ours' and 'Baseline' and group suffixes from legend."""
    for trace in fig.data:
        if trace.name:
            # Remove ", Ours" or ", Baseline" or "Ours, " etc.
            new_name = trace.name.split(",")[0].strip()
            trace.name = new_name
    return fig


def generate_star_plot(
    results: dict[str, pl.DataFrame],
    dump: bool,
    output_path: Path,
    model_name: str = "CAST",
    only_info0: bool = False,
):
    """Generates a star plot (radar chart) using Plotly with Min-Max normalization."""
    df = prepare_plot_data(results, dump, model_name=model_name)
    color_map = get_color_map(df, model_name=model_name, only_info0=only_info0)
    legend_order = get_legend_order(model_name=model_name)

    # Build the star plot
    fig = px.line_polar(
        df,
        r="NormalizedValue",
        theta="Metric",
        color="LegendLabel",
        line_close=True,
        hover_data={
            "Value": ":.4f",
            "NormalizedValue": False,
            "Model": True,
            "Metric": True,
        },
        template="plotly_white",
        color_discrete_map=color_map,
        category_orders={"LegendLabel": legend_order},
    )

    fig.update_traces(
        fill="toself",
        opacity=0.2,
        line=dict(width=3.0),
        marker=dict(size=8),
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                gridcolor="#E5E5E5",
            ),
            angularaxis=dict(
                gridcolor="#E5E5E5",
                linecolor="#E5E5E5",
            ),
        ),
        legend=dict(
            title="Model",
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=20, b=20, l=40, r=40),
    )

    save_plotly_figure(clean_legend(fig), output_path, ".pdf")


def generate_parallel_plot(
    results: dict[str, pl.DataFrame],
    dump: bool,
    output_path: Path,
    model_name: str = "CAST",
    only_info0: bool = False,
):
    """Generates a parallel coordinates plot (lines) using Plotly."""
    df = prepare_plot_data(results, dump, model_name=model_name)
    color_map = get_color_map(df, model_name=model_name, only_info0=only_info0)
    legend_order = get_legend_order(model_name=model_name)

    # Sort metrics for consistent X axis
    metric_order = [METRIC_DISPLAY_MAP.get(m, m) for m in METRIC_DISPLAY_MAP]
    df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)
    df = df.sort_values("Metric")

    fig = px.line(
        df,
        x="Metric",
        y="NormalizedValue",
        color="LegendLabel",
        line_group="Model",
        hover_data={"Value": ":.4f", "Model": True, "NormalizedValue": False},
        template="plotly_white",
        color_discrete_map=color_map,
        markers=True,
        category_orders={"LegendLabel": legend_order},
    )

    fig.update_traces(opacity=0.6, line=dict(width=2))

    fig.update_layout(
        yaxis=dict(title="Relative Performance (Normalized)", range=[0, 1.05]),
        xaxis=dict(title="Metric"),
        legend=dict(
            title="Model",
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=20, b=20, l=40, r=40),
    )

    save_plotly_figure(clean_legend(fig), output_path, "_parallel.pdf")


def generate_cleveland_plot(
    results: dict[str, pl.DataFrame],
    dump: bool,
    output_path: Path,
    model_name: str = "CAST",
    only_info0: bool = False,
):
    """Generates a Cleveland dot plot using Plotly."""
    df = prepare_plot_data(results, dump, model_name=model_name)
    color_map = get_color_map(df, model_name=model_name, only_info0=only_info0)
    legend_order = get_legend_order(model_name=model_name)

    fig = px.scatter(
        df,
        x="NormalizedValue",
        y="Metric",
        color="LegendLabel",
        symbol="Group",
        symbol_map={"Ours": "circle", "Baseline": "square"},
        hover_data={"Value": ":.4f", "Model": True, "NormalizedValue": False},
        template="plotly_white",
        color_discrete_map=color_map,
        category_orders={"LegendLabel": legend_order},
    )

    fig.update_traces(marker=dict(size=12, opacity=0.8))

    fig.update_layout(
        xaxis=dict(title="Relative Performance (Normalized)", range=[-0.05, 1.05]),
        yaxis=dict(title="Metric", automargin=True),
        legend=dict(
            title="Model",
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=20, b=20, l=40, r=40),
    )

    save_plotly_figure(clean_legend(fig), output_path, "_cleveland.pdf")


def save_plotly_figure(fig, output_path: Path, default_suffix: str):
    """Helper to save a Plotly figure to disk."""
    if not output_path.suffix:
        output_path = output_path.with_suffix(default_suffix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path))
    print(f"Plot saved to {output_path}")
