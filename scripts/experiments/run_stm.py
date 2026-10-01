# -*- coding: utf-8 -*-
"""Run STM experiments using R training and Python evaluation."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import polars as pl
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.evaluation as evaluation
import src.logger_config as logger_config
import src.utils as utils

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

STEMMED_TEXT_COL = "clean_text_stemmed"


def resolve_stm_inputs(experiment_config: dict) -> tuple[str, Path, Path]:
    """Resolves the dataset name and STM input files for an experiment.

    The stemmed and unstemmed variants of a dataset share one embeddings file,
    so the preprocessing level is taken from ``text_col``. STM must train on
    the same preprocessing level as the models it is compared against.

    Args:
        experiment_config: The ``experiment`` section of a loaded config.

    Returns:
        A tuple of (dataset_name, rds_path, bow_path).
    """
    dataset_name = Path(experiment_config["dataset_path"]).stem.replace(
        "_embeddings", ""
    )
    is_stemmed = experiment_config.get("text_col") == STEMMED_TEXT_COL
    file_prefix = f"{dataset_name}_stemmed" if is_stemmed else dataset_name
    rds_path = PROJECT_ROOT / f"data/processed/{file_prefix}_stm_data.rds"
    bow_path = PROJECT_ROOT / f"data/processed/{file_prefix}_bow.parquet"
    return dataset_name, rds_path, bow_path


def main():
    parser = argparse.ArgumentParser(description="Run STM experiments.")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Name of the experiment yaml file (e.g., anes_stm)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Override the sample size specified in the config file.",
    )
    parser.add_argument(
        "--start_from",
        type=str,
        help="Model ID to start execution from (skips previous models)",
    )
    args = parser.parse_args()

    try:
        # 1. Setup
        config = utils.load_config(args.exp, EXPERIMENTS_DIR)

        # Override sample size if requested
        if args.sample is not None:
            config["experiment"]["sample_size"] = args.sample

        exp_name = config["experiment"]["name"]

        # Adjust experiment name if sampling is applied to distinguish files
        sample_size = config["experiment"].get("sample_size")
        if sample_size:
            exp_name = f"{exp_name}_s{sample_size}"

        random_state = utils.get_random_state(config["experiment"]["random_state"])
        primary_random_state = (
            random_state[0] if isinstance(random_state, list) else random_state
        )

        logger = logger_config.setup_logging(exp_name, LOG_DIR)
        logger.info(f"Starting STM experiment: {exp_name}")
        logger.info(f"Random state: {random_state}")

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

        # 2. Dataset Path and Metadata
        dataset_name, rds_path, bow_path = resolve_stm_inputs(config["experiment"])

        logger.info(f"Dataset name: {dataset_name}")
        logger.info(f"Text column: {config['experiment'].get('text_col')}")
        logger.info(f"RDS path: {rds_path}")
        logger.info(f"BoW path: {bow_path}")

        if not rds_path.exists():
            logger.error(
                f"RDS file not found: {rds_path}. "
                "Run scripts/r_scripts/build_bow.R first."
            )
            return

        # 3. Handle Sampling
        sample_indices_path = None
        n_observations = None

        # We need to load the full bow data to know the total N and to get text for coherence
        logger.info(f"Loading BoW data from {bow_path}...")
        bow_df = pl.read_parquet(bow_path)
        logger.info(f"Loaded BoW data with {len(bow_df)} rows.")

        # Log metadata columns
        # We exclude obvious identifiers and text columns from the "metadata" log
        exclude_cols = [
            "bow_text",
            "index",
            "id",
            "text",
            "clean_text",
            "clean_text_lower",
            "clean_text_lower_punctless",
            "token_count",
        ]
        meta_cols = [c for c in bow_df.columns if c not in exclude_cols]
        logger.info(f"Available metadata columns (potential covariates): {meta_cols}")

        sample_size = config["experiment"].get("sample_size")
        if sample_size:
            logger.info(f"Sampling {sample_size} observations...")
            # Use established sampling logic
            if len(bow_df) > sample_size:
                sampled_df = bow_df.sample(n=sample_size, seed=primary_random_state)
                # Save indices for R to use
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(sampled_df["index"].to_list(), f)
                    sample_indices_path = f.name
                n_observations = sample_size
                # For coherence we use the sampled text
                eval_texts = sampled_df["bow_text"].to_list()
            else:
                logger.warning(
                    f"Sample size {sample_size} >= dataset size {len(bow_df)}. No sampling applied."
                )
                n_observations = len(bow_df)
                eval_texts = bow_df["bow_text"].to_list()
        else:
            n_observations = len(bow_df)
            eval_texts = bow_df["bow_text"].to_list()

        logger.info(f"Total observations for training: {n_observations}")

        # Tokenize eval_texts for coherence (simple split as they are already BoW)
        tokenized_texts = [t.split() for t in eval_texts if t]

        start_timestamp = datetime.datetime.now().isoformat()
        file_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        results = []
        qualitative_dfs = []

        # 4. Iterate over models
        models_config = config.get("models", [])
        logger.info(f"Running {len(models_config)} model configurations...")

        prevalence_formula = config["experiment"].get("prevalence_formula")
        if prevalence_formula:
            logger.info(f"Using prevalence formula: {prevalence_formula}")
        else:
            logger.warning(
                "No prevalence formula provided. Running vanilla STM (no metadata)."
            )

        skip_models = False
        if args.start_from:
            skip_models = True

        # Determine all seeds
        seeds = random_state if isinstance(random_state, list) else [random_state]

        for m_conf in tqdm(models_config, desc="Running STM models"):
            m_id = m_conf.get("id", "Unknown")
            k = m_conf.get("parameters", {}).get("k")

            if skip_models:
                if m_id == args.start_from:
                    skip_models = False
                    logger.info(f"Resuming execution from model: {m_id}")
                else:
                    logger.info(f"Skipping model: {m_id}")
                    continue

            if not k:
                logger.error(f"Model {m_id} missing parameter 'k'. Skipping.")
                continue

            for seed in seeds:
                if len(seeds) > 1:
                    run_id = f"{m_id}_seed{seed}"
                else:
                    run_id = m_id

                logger.info(f"--- Running model: {run_id} (K={k}) with seed {seed} ---")

                # Temp output dir for R results
                with tempfile.TemporaryDirectory() as tmp_output:
                    model_filename = f"stm_{dataset_name}_{run_id}_{file_timestamp}.rds"
                    model_path = MODELS_DIR / model_filename

                    # Run R script
                    cmd = [
                        "Rscript",
                        "scripts/r_scripts/train_stm.R",
                        "--rds_path",
                        str(rds_path),
                        "--k",
                        str(k),
                        "--output_dir",
                        tmp_output,
                        "--seed",
                        str(seed),
                        "--model_path",
                        str(model_path),
                    ]
                    if sample_indices_path:
                        cmd.extend(["--indices_path", sample_indices_path])

                    if prevalence_formula:
                        cmd.extend(["--prevalence_formula", prevalence_formula])

                    logger.info(f"[{run_id}] Running R training script...")
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, encoding="utf-8"
                    )

                    if result.returncode != 0:
                        logger.error(f"[{run_id}] R training failed:\n{result.stderr}")
                        continue

                    # Parse R output for extra info
                    for line in result.stdout.splitlines():
                        if any(
                            x in line
                            for x in [
                                "Loaded RDS data",
                                "Applying sampling",
                                "Keeping",
                                "Metadata columns:",
                                "Documents:",
                                "Vocab size:",
                                "Generated topics:",
                                "Prevalence formula:",
                                "Formula stored in model:",
                            ]
                        ):
                            logger.info(f"[{run_id}] R: {line}")

                    logger.info(f"[{run_id}] R training successful.")

                    # 5. Load R outputs for Evaluation
                    beta = pl.read_parquet(Path(tmp_output) / "beta.parquet").to_numpy()
                    theta = pl.read_parquet(
                        Path(tmp_output) / "theta.parquet"
                    ).to_numpy()
                    with open(
                        Path(tmp_output) / "vocab.txt", "r", encoding="utf-8"
                    ) as f:
                        vocab = [line.strip() for line in f]
                    with open(
                        Path(tmp_output) / "duration.txt", "r", encoding="utf-8"
                    ) as f:
                        duration = float(f.read().strip())

                    logger.info(
                        f"[{run_id}] Duration: {duration:.2f}s, "
                        f"Vocab size: {len(vocab)}, "
                        f"Beta shape: {beta.shape}"
                    )
                    top_words = evaluation.get_top_words_from_beta(beta, vocab)
                    octis_output = evaluation.topic_words_to_octis(top_words)

                    metrics = {
                        "model_name": run_id,
                        "duration_seconds": duration,
                        "n_topics": k,
                        "outliers": 0,  # STM doesn't really have outliers like HDBSCAN
                        **evaluation.topic_diagnostics(
                            octis_output, texts=tokenized_texts
                        ),
                        "evaluation_protocol": evaluation.EVALUATION_PROTOCOL,
                    }

                    # Coherence Loop
                    for cm in config["experiment"]["coherence_metrics"]:
                        metrics[cm] = evaluation.compute_coherence(
                            model_output=octis_output, texts=tokenized_texts, measure=cm
                        )

                    # Diversity Loop
                    for dm in config["experiment"]["diversity_metrics"]:
                        metrics[dm] = evaluation.compute_diversity(
                            dm, model_output=octis_output
                        )

                    # Add Run Metadata
                    run_metadata = {
                        "experiment_id": exp_name,
                        "random_state": seed,
                        "clustering_algo": "STM",
                        "dim_red_algo": "None",
                        "n_observations": n_observations,
                        "timestamp": start_timestamp,
                        "file_timestamp": file_timestamp,
                        "dataset_name": dataset_name,
                        "k": k,
                    }
                    metrics.update(run_metadata)
                    results.append(metrics)

                    # 7. Extract Qualitative Data
                    qual_df = utils.extract_stm_qualitative_data(
                        theta=theta,
                        beta=beta,
                        vocab=vocab,
                        documents=eval_texts,
                        model_id=run_id,
                        metadata=run_metadata,
                    )
                    qualitative_dfs.append(qual_df)

                    # Incremental Save
                    if results:
                        inc_results_filename = (
                            f"{exp_name}-{file_timestamp}-"
                            f"{primary_random_state}_incremental"
                        )
                        inc_results_path = RESULTS_DIR / f"{inc_results_filename}.csv"
                        pl.DataFrame(results).write_csv(inc_results_path)
                        logger.info(
                            f"[{run_id}] Incremental results saved to "
                            f"{inc_results_path}"
                        )

                    if qualitative_dfs:
                        inc_output_path = OUTPUT_DIR / f"{inc_results_filename}.json"
                        inc_consolidated_qual_df = pl.concat(
                            qualitative_dfs, how="diagonal"
                        )
                        json_str = inc_consolidated_qual_df.write_json()
                        parsed_json = json.loads(json_str)
                        with open(inc_output_path, "w", encoding="utf-8") as f:
                            json.dump(parsed_json, f, indent=4)
                        logger.info(
                            f"[{run_id}] Incremental qualitative data saved "
                            f"to {inc_output_path}"
                        )

        # 8. Save Final Results
        if results:
            results_df = pl.DataFrame(results)
            results_filename = f"{exp_name}-{file_timestamp}-{primary_random_state}"
            results_path = RESULTS_DIR / f"{results_filename}.csv"
            results_df.write_csv(results_path)
            logger.info(f"Results saved to {results_path}")

        if qualitative_dfs:
            consolidated_qual_df = pl.concat(qualitative_dfs, how="diagonal")
            output_path = OUTPUT_DIR / f"{results_filename}.json"

            json_str = consolidated_qual_df.write_json()
            parsed_json = json.loads(json_str)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=4)
            logger.info(f"Qualitative data saved to {output_path}")

        # Cleanup
        if sample_indices_path and os.path.exists(sample_indices_path):
            os.remove(sample_indices_path)

    except Exception:
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
