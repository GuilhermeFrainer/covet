import argparse
import sys
from pathlib import Path

import polars as pl

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.make_table import generate_best_models_latex_table
from src.results_analysis import canonical_dataset_expr, find_best_models
from src.visualization import (
    generate_cleveland_plot,
    generate_parallel_plot,
    generate_star_plot,
)

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_NAME = "CAST"

HIGHLIGHT_COLORS = ("FFD700", "C0C0C0", "CD7F32")

MODEL_SHORT_NAMES = {
    "mv_co_reg_spectral": "{model_name}_1",
    "mv_co_reg_spectral_info0": "{model_name}_1-info0",
    "mv_spectral": "{model_name}_2",
    "mv_spectral_info0": "{model_name}_2-info0",
    "aligned_umap": "{model_name}_3",
    "baseline": "BERTopic_1",
    "umap_spectral": "BERTopic_2",
    "append_umap": "Naive",
    "stm": "STM",
}

LABEL_DESCRIPTIONS = {
    "{model_name}_1": "{model_name} with Co-Reg. Spectral Clustering",
    "{model_name}_1-info0": "Same as above with the info0 parameter",
    "{model_name}_2": "{model_name} with MV Spectral Clustering",
    "{model_name}_2-info0": "Same as above with the info0 parameter",
    "{model_name}_3": "{model_name} with Aligned UMAP",
    "BERTopic_1": "Default BERTopic",
    "BERTopic_2": "BERTopic with Spectral",
    "Naive": "Naive baseline",
    "STM": "Structural Topic Model",
}


def generate_label_table(
    only_info0: bool = False, result_type: str | None = None
) -> str:
    """Generates a LaTeX table for model label descriptions.

    Args:
        only_info0: If True, excludes info0 variants from the table.
        result_type: Optional result type identifier used to append preprocessing context
            to the table caption. Valid options:
            - 'standard': Standard unstemmed text with representation stopwords removed.
            - 'stemmed': Stemmed text with stopwords removed.
            - 'no_stopword_removal' (or 'with_stopwords', 'no_stopword'): Unstemmed text
              without representation stopword removal.

    Returns:
        A LaTeX table string.
    """
    res_descr = ""
    if result_type:
        rt = result_type.lower()
        if rt == "standard":
            res_descr = (
                " (Standard Unstemmed Text with Representation Stopwords Removed)"
            )
        elif rt == "stemmed":
            res_descr = " (Stemmed Text with Stopwords Removed)"
        elif rt in ("no_stopword", "no_stopword_removal", "with_stopwords"):
            res_descr = " (Unstemmed Text without Representation Stopword Removal)"

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Model Label Descriptions{res_descr}}}",
        "\\label{tab:label_descriptions}",
        "\\begin{tabular}{ll}",
        "    \\toprule",
        "    Label & Description \\\\",
        "    \\midrule",
    ]
    for label, desc in LABEL_DESCRIPTIONS.items():
        if only_info0 and "-info0" in label:
            continue

        # Use math mode for labels to support subscripts
        fmt_label = label.format(model_name=MODEL_NAME)
        if "_" in fmt_label:
            base, sub = fmt_label.split("_", 1)
            fmt_label = f"$\\text{{{base}}}_{{{sub}}}$"

        fmt_desc = desc.format(model_name=MODEL_NAME)
        lines.append(f"    {fmt_label} & {fmt_desc} \\\\")

    lines.extend(
        [
            "    \\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Find best models from experiment results across metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    req_group = parser.add_argument_group("Required / Core Arguments")
    req_group.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset (e.g., fed, trump, anes, yelp). Required unless --label-table is provided.",
    )

    proc_group = parser.add_argument_group("Analysis & Processing Options")
    proc_group.add_argument(
        "--exclude-clustering",
        type=str,
        nargs="+",
        default=["kmeans", "spherical_kmeans"],
        help="Clustering algorithms to exclude (default: kmeans spherical_kmeans). Pass 'none' to disable filtering.",
    )
    proc_group.add_argument(
        "--exclude-dim-red",
        type=str,
        nargs="+",
        default=["pca"],
        help="Dimensionality reduction algorithms to exclude (default: pca). Pass 'none' to disable filtering.",
    )
    proc_group.add_argument(
        "--dump",
        action="store_true",
        help="Dump all model configurations instead of selecting best per type.",
    )
    proc_group.add_argument(
        "--suppress-nulls",
        action="store_true",
        help="Suppress printing metric standard deviation/value when metric is null.",
    )
    proc_group.add_argument(
        "--average",
        action="store_true",
        help="Calculate the average performance per model type instead of finding the best.",
    )
    proc_group.add_argument(
        "--merge-info0",
        action="store_true",
        help="Treat models with and without the 'info0' parameter as the same model type.",
    )
    proc_group.add_argument(
        "--only-info0",
        action="store_true",
        help="Focus analysis only on multi-view models that use the 'info0' parameter.",
    )
    proc_group.add_argument(
        "--result-type",
        type=str,
        choices=[
            "standard",
            "stemmed",
            "no_stopword_removal",
            "with_stopwords",
            "no_stopword",
            "all",
        ],
        default="all",
        help="Filter results by preprocessing result type (default: all).",
    )

    output_group = parser.add_argument_group("Output Options (Plots & Tables)")
    output_group.add_argument(
        "--latex",
        type=str,
        nargs="?",
        const="__STDOUT__",
        metavar="FILE",
        help="Output the results as a LaTeX table. If FILE is omitted, prints to stdout.",
    )
    output_group.add_argument(
        "--star-plot",
        type=str,
        metavar="FILE",
        help="Generate a star plot and save to FILE (e.g., output.pdf).",
    )
    output_group.add_argument(
        "--parallel",
        type=str,
        metavar="FILE",
        help="Generate a parallel coordinates plot and save to FILE.",
    )
    output_group.add_argument(
        "--cleveland",
        type=str,
        metavar="FILE",
        help="Generate a Cleveland dot plot and save to FILE.",
    )
    output_group.add_argument(
        "--label-table",
        action="store_true",
        help="Output the LaTeX table of model label descriptions and exit.",
    )

    args = parser.parse_args()

    if args.label_table:
        print(
            generate_label_table(
                only_info0=args.only_info0, result_type=args.result_type
            )
        )
        return

    if not args.dataset:
        parser.error("--dataset is required unless --label-table is provided.")

    dataset = args.dataset
    if dataset == "anes_stemmed":
        dataset = "anes"

    csv_files = list(RESULTS_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {RESULTS_DIR}")
        return

    target_res_type = args.result_type
    if target_res_type in ("no_stopword", "with_stopwords"):
        target_res_type = "no_stopword_removal"

    all_dfs = []
    for f in csv_files:
        filename = f.name.lower()
        if target_res_type == "stemmed" and "stemmed" not in filename:
            continue
        if target_res_type == "no_stopword_removal" and "no_stopword" not in filename:
            continue
        if target_res_type == "standard" and (
            "stemmed" in filename or "no_stopword" in filename
        ):
            continue

        try:
            df = pl.read_csv(f, infer_schema_length=None)
            # Filter by dataset early if possible
            if "dataset_name" in df.columns:
                # Normalize dataset name
                df = df.with_columns(canonical_dataset_expr(pl.col("dataset_name")))
                df = df.filter(pl.col("dataset_name") == dataset)
                if not df.is_empty():
                    all_dfs.append(df)

        except Exception:
            continue

    if not all_dfs:
        print(
            f"No results found for dataset: {dataset} (result-type: {args.result_type})"
        )
        return

    # Combine all dataframes
    df = pl.concat(all_dfs, how="diagonal")

    if args.only_info0:
        has_info0 = df.filter(pl.col("model_name").str.contains("info0")).height > 0
        if has_info0:
            df = df.filter(
                ~pl.col("model_name").str.contains("mv_spectral|co_reg_spectral")
                | pl.col("model_name").str.contains("info0")
            )
            # Treat them as vanilla models in downstream processing
            df = df.with_columns(pl.col("model_name").str.replace("_info0", ""))
        else:
            print("\n" + "!" * 80)
            print(
                f"WARNING: --only-info0 was requested, but no info0 variants were found for dataset '{dataset}'."
            )
            print(
                "Falling back to non-info0 variants, but they will be colored as info0 in the plots."
            )
            print("!" * 80 + "\n")

    results = find_best_models(
        df,
        dataset,
        exclude_clustering=args.exclude_clustering,
        exclude_dim_red=args.exclude_dim_red,
        dump=args.dump,
        average=args.average,
        merge_info0=args.merge_info0,
        suppress_nulls=args.suppress_nulls,
    )

    if not results:
        print(f"No valid metric results found for dataset: {dataset}")
        return

    if args.star_plot:
        generate_star_plot(
            results,
            args.dump,
            Path(args.star_plot),
            model_name=MODEL_NAME,
            only_info0=args.only_info0,
        )

    if args.parallel:
        generate_parallel_plot(
            results,
            args.dump,
            Path(args.parallel),
            model_name=MODEL_NAME,
            only_info0=args.only_info0,
        )

    if args.cleveland:
        generate_cleveland_plot(
            results,
            args.dump,
            Path(args.cleveland),
            model_name=MODEL_NAME,
            only_info0=args.only_info0,
        )

    if args.latex:
        latex_table = generate_best_models_latex_table(
            results,
            dataset,
            dump=args.dump,
            average=args.average,
            highlight_colors=HIGHLIGHT_COLORS,
            result_type=args.result_type,
        )
        if args.latex == "__STDOUT__":
            print("\n" + latex_table)
        else:
            output_path = Path(args.latex)
            output_path.write_text(latex_table, encoding="utf-8")
            print(f"LaTeX table saved to {output_path}")
    elif not (args.star_plot or args.parallel or args.cleveland):
        title = (
            "All model configurations"
            if args.dump
            else ("Average model performance" if args.average else "Best models")
        )
        print(f"\n{title} for dataset: {dataset}")
        print("=" * (len(title) + 14 + len(dataset)))

        for metric, best_per_type in results.items():
            print(f"\nMetric: {metric}")
            print("-" * (8 + len(metric)))

            for row in best_per_type.iter_rows(named=True):
                model_name = row["best_model_name"]
                max_value = row["max_value"]
                std_value = row.get("std_value", 0.0)
                if std_value is not None and std_value > 0.0:
                    print(f"  {model_name:<50} | {max_value:>8.4f} ± {std_value:<7.4f}")
                else:
                    print(f"  {model_name:<50} | {max_value:>8.4f}")


if __name__ == "__main__":
    main()
