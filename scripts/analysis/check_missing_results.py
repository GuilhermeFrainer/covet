"""Diagnostic CLI to check for missing and partial experiment results."""

import argparse
import pathlib
import sys
from typing import List

import polars as pl

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.merge_results import get_dataset_info
from src.verification import verify_dataset_completeness

DEFAULT_DATASETS = ["fed", "yelp", "trump", "anes", "gadarian"]
DEFAULT_RESULT_TYPES = ["standard", "stemmed", "no_stopword_removal"]


def main():
    parser = argparse.ArgumentParser(
        description="Verify experiment completeness and detect partial runs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name (fed, yelp, trump, anes, gadarian, or 'all').",
    )
    parser.add_argument(
        "--result-type",
        type=str,
        choices=["all", "standard", "stemmed", "no_stopword_removal", "no_stopword"],
        default="all",
        help="Preprocessing result type to verify (default: all).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing CSV results (default: results).",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Directory containing YAML configurations (default: experiments).",
    )

    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    exp_dir = pathlib.Path(args.experiments_dir)

    target_datasets = (
        DEFAULT_DATASETS if args.dataset == "all" else [args.dataset.strip().lower()]
    )

    res_type = args.result_type
    if res_type == "no_stopword":
        res_type = "no_stopword_removal"

    target_types = (
        DEFAULT_RESULT_TYPES if res_type == "all" else [res_type.strip().lower()]
    )

    all_csvs = list(results_dir.glob("*.csv")) if results_dir.exists() else []

    has_any_partial = False

    for dataset in target_datasets:
        for r_type in target_types:
            # Collect matching CSV files for this dataset and result type
            matched_dfs: List[pl.DataFrame] = []
            for csv_file in all_csvs:
                d_name, d_type = get_dataset_info(csv_file)
                if d_name == dataset and d_type == r_type:
                    try:
                        # All strings: raw run files and merged files infer
                        # different types for the same column (e.g. an empty
                        # column as String in one, Int64 in another), which
                        # makes the diagonal concat fail.
                        df = pl.read_csv(csv_file, infer_schema_length=0)
                        matched_dfs.append(df)
                    except Exception:
                        pass

            combined_df = (
                pl.concat(matched_dfs, how="diagonal") if matched_dfs else None
            )

            report = verify_dataset_completeness(
                dataset_name=dataset,
                dataset_type=r_type,
                df=combined_df,
                experiments_dir=exp_dir,
            )

            # Only print if configurations exist for this dataset
            if report.complete_models or report.partial_models or report.unrun_models:
                print(report.summary())
                slurm_cmd = report.slurm_rerun_command()
                if slurm_cmd:
                    print("\nTo re-run partial models on Slurm, execute:")
                    print(f"  {slurm_cmd}\n")

                if report.has_partial_models:
                    has_any_partial = True

    if has_any_partial:
        print(
            "\n[WARNING] Partial models detected! "
            "Complete all missing runs before merging."
        )
        sys.exit(1)
    else:
        print(
            "\n[SUCCESS] Verification complete. All executed models are 100% complete."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
