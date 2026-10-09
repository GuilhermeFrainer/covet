import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import datetime
import traceback

import numpy as np
import polars as pl

import src.data as data
import src.logger_config as logger_config
import src.make_table as make_table
import src.run_provenance as run_provenance
import src.utils as utils
from src.optimizer import Optimizer

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"
TABLES_DIR = PROJECT_ROOT / "tables"


def main() -> int:
    """Runs one experiment config.

    Returns:
        0 if every run trained; 1 if the pipeline crashed or any run failed.
        Failed runs are otherwise only logged, so SLURM would report the job
        as completed.
    """
    parser = argparse.ArgumentParser(description="Run a hyperparameter optimization.")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Name of the optimization yaml file (e.g., yelp_opt_spectral)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Override the sample size specified in the config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted optimization run if existing results are found.",
    )
    parser.add_argument(
        "--model", type=int, help="Run only the n-th model configuration (1-indexed)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Run on a single specific random seed, overriding seeds in config.",
    )
    parser.add_argument(
        "--single-seed",
        action="store_true",
        help="Run only the first random seed from the config (ideal for dry runs).",
    )
    parser.add_argument(
        "--remove-rep-stopwords",
        action="store_true",
        default=True,
        help=(
            "Remove English stop words from BERTopic topic representations "
            "(c-TF-IDF) using CountVectorizer (default: True)."
        ),
    )
    parser.add_argument(
        "--keep-rep-stopwords",
        action="store_false",
        dest="remove_rep_stopwords",
        help="Keep English stop words in BERTopic topic representations.",
    )
    args = parser.parse_args()

    logger = None  # Initialize logger to None

    try:
        # Setup
        config = utils.load_config(args.exp, EXPERIMENTS_DIR)

        # Override sample size if requested
        if args.sample is not None:
            config["experiment"]["sample_size"] = args.sample

        exp_name = config["experiment"]["name"]
        random_state = utils.get_random_state(config["experiment"]["random_state"])
        if args.seed is not None:
            random_state = [args.seed]
        elif args.single_seed:
            random_state = (
                [random_state[0]] if isinstance(random_state, list) else [random_state]
            )

        primary_random_state = (
            random_state[0] if isinstance(random_state, list) else random_state
        )

        logger = logger_config.setup_logging(exp_name, LOG_DIR)

        if (
            isinstance(random_state, list)
            and len(random_state) > 1
            and (
                args.sample is not None
                or config["experiment"].get("sample_size") is not None
            )
        ):
            logger.warning(
                f"Multiple seeds are specified, but data sampling is active. "
                f"Only the first seed ({primary_random_state}) will be used "
                "for data sampling to ensure consistent data inputs across "
                "different model runs."
            )

        # Data loading
        logger.info("Loading and preparing data...")
        prepared = data.load_and_prep_data(
            config, random_state=primary_random_state, return_prepared=True
        )
        text, embeddings, scaled_metadata = prepared

        # Check for NaNs and warn if found
        if isinstance(scaled_metadata, pl.DataFrame):
            if scaled_metadata.width > 0:
                null_counts = scaled_metadata.null_count()
                nan_cols = [
                    col for col in scaled_metadata.columns if null_counts[col][0] > 0
                ]
                if nan_cols:
                    logger.warning(
                        f"Metadata contains null/NaN values in "
                        f"{len(nan_cols)} feature columns: {nan_cols}"
                    )
        elif scaled_metadata is not None and getattr(scaled_metadata, "size", 0) > 0:
            if np.isnan(scaled_metadata).any():
                nan_indices = np.where(np.isnan(scaled_metadata).any(axis=0))[0]
                logger.warning(
                    f"Metadata contains NaN values in "
                    f"{len(nan_indices)} feature columns."
                )
                logger.warning(f"NaN indices: {nan_indices.tolist()}")

        # Model configuration
        logger.info("Loading model configuration for optimization...")
        model_config = config.get("model")
        if not model_config:
            raise ValueError(
                "Configuration file must contain a 'model' section for optimization."
            )

        # Get optional rounding parameter
        decimal_digits = config.get("experiment", {}).get("decimal_digits")

        # Determine filename base tag
        is_stemmed = "stemmed" in exp_name.lower()
        if is_stemmed:
            tag = "stemmed"
        elif args.remove_rep_stopwords:
            tag = "remove_rep_stopwords"
        else:
            tag = "keep_rep_stopwords"

        fn_base = exp_name if tag in exp_name else f"{exp_name}_{tag}"
        if args.model is not None:
            fn_base = f"{fn_base}_m{args.model}"

        # Check for existing results to resume if --resume is passed
        start_index = 0
        results_path = None

        if args.resume:
            patterns = [
                f"{fn_base}-*-{primary_random_state}.csv",
                f"{exp_name}-*-{primary_random_state}.csv",
            ]
            matching_files = sorted(
                {f for pat in patterns for f in RESULTS_DIR.glob(pat)}
            )
            if matching_files:
                target_n_obs = len(text)
                selected_file = None
                for file in reversed(matching_files):
                    try:
                        df = pl.read_csv(file, infer_schema_length=None)
                        # Ensure we only resume files belonging to the current campaign
                        if "campaign_id" in df.columns and len(df) > 0:
                            if (
                                df["campaign_id"][0]
                                != run_provenance.DEFAULT_CAMPAIGN_ID
                            ):
                                continue
                        else:
                            # Pre-correction legacy run without campaign_id;
                            # do not resume/append
                            continue

                        if "n_observations" in df.columns and len(df) > 0:
                            if df["n_observations"][0] == target_n_obs:
                                selected_file = file
                                existing_df = df
                                break
                        else:
                            selected_file = file
                            existing_df = df
                            break
                    except Exception:
                        continue

                if selected_file is not None:
                    start_index = len(existing_df)
                    results_path = selected_file
                    logger.info(
                        f"Found existing results file: {selected_file}. "
                        f"Resuming from index {start_index}."
                    )
                else:
                    logger.warning(
                        "No existing results file matching the current "
                        "dataset size was found. Starting from scratch."
                    )
            else:
                logger.info(
                    "No existing results file found for resumption. "
                    "Starting from scratch."
                )

        if results_path is None:
            file_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            results_filename = f"{fn_base}-{file_timestamp}-{primary_random_state}"
            results_path = RESULTS_DIR / f"{results_filename}.csv"
        else:
            # Extract timestamp from existing filename
            # (format: exp_name-timestamp-random_state.csv)
            # timestamp has a dash in it (e.g., 20260408-123456)
            parts = results_path.stem.split("-")
            if len(parts) >= 3:
                # The last part is random_state, the two parts before it are
                # the timestamp
                file_timestamp = f"{parts[-3]}-{parts[-2]}"
            else:
                file_timestamp = "unknown"

        # Initialize and run optimizer
        optimizer = Optimizer(
            texts=text,
            embeddings=embeddings,
            scaled_metadata=scaled_metadata,
            model_config=model_config,
            experiment_config=config,
            experiment_id=exp_name,
            random_state=random_state,
            file_timestamp=file_timestamp,
            remove_rep_stopwords=args.remove_rep_stopwords,
            prepared_data=prepared,
            assignment_output_dir=PROJECT_ROOT / "output" / "document_assignments",
        )

        target_index = args.model - 1 if args.model is not None else None
        optimizer.run(start_index=start_index, target_index=target_index)

        # Save Results
        optimizer.save_results(results_path, decimal_digits=decimal_digits)

        logger.info(
            f"Optimization session finished. Results saved/updated at {results_path}"
        )

        # Generate and save LaTeX table if there are any results in the final file
        if results_path.exists():
            final_results_df = pl.read_csv(results_path, infer_schema_length=None)
            if not final_results_df.is_empty():
                latex_table = make_table.generate_latex_table(final_results_df)
                table_filename = f"{results_path.stem}.tex"
                table_path = TABLES_DIR / table_filename
                with open(table_path, "w") as f:
                    f.write(latex_table)
                logger.info(f"Latex table saved at {table_path}")
        else:
            logger.warning("No results file found, skipping LaTeX table creation.")

        failed = [m for m in optimizer.run_manifests if m.get("status") == "failure"]
        if failed:
            logger.error(f"{len(failed)} run(s) failed; see the errors above.")
            return 1
        return 0

    except Exception as e:
        if logger:
            logger.error(f"Pipeline crashed: {e}", exc_info=True)
        else:
            # If logger setup fails, print to stderr
            print(f"Pipeline crashed before logger was configured: {e}")
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
