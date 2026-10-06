# -*- coding: utf-8 -*-
"""Run STM experiments: R trains, Python evaluates.

`scripts/r_scripts/train_stm.R` fits each model and exports its topic-word
(beta) and document-topic (theta) matrices. R runs either from the local
installation or inside the lightweight STM image (`--r-runner docker`), which
holds only R, so evaluation always happens here, on the host.

STM's bag-of-words is tokenized with the analyzer BERTopic variants use for
topic words (see scripts/data_prep/build_bow_tokens.py), so its documents
double as the coherence reference corpus the neural models are scored on.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import polars as pl
from gensim.corpora.dictionary import Dictionary

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.evaluation as evaluation
import src.logger_config as logger_config
import src.metadata_alignment as metadata_alignment
import src.run_provenance as run_provenance
import src.utils as utils
from src.document_assignments import atomic_json, file_checksum

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
ASSIGNMENTS_DIR = OUTPUT_DIR / "document_assignments"
# R exchange files live inside the project so a container mount can reach them.
EXCHANGE_DIR = PROJECT_ROOT / ".stm_tmp"

STEMMED_TEXT_COL = "clean_text_stemmed"
DEFAULT_STM_IMAGE = "cast:stm-lite-v0.2.0"
CONTAINER_WORKDIR = "/work"
# Separates these runs from STM fitted on the earlier quanteda bag-of-words.
STM_CAMPAIGN_ID = "stm_shared_vocab_v1"

# R output lines worth repeating in the experiment log.
R_LOG_MARKERS = (
    "Loaded RDS data",
    "Keeping",
    "Documents:",
    "Vocab size:",
    "Generated topics:",
    "Prevalence formula:",
    "Training finished",
)


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


def expand_runs(config: dict, seeds: list[int]) -> list[tuple[dict, int]]:
    """Lists every (model config, seed) run in the order `--model` indexes.

    Models vary slowest, matching `run_optimizer.py`, whose `--model` index
    also enumerates configurations before seeds.
    """
    return [(model, seed) for model in config.get("models", []) for seed in seeds]


def covariate_columns(experiment_config: dict) -> list[str]:
    """Returns the raw covariates the neural models receive, in config order."""
    covariates = experiment_config.get("covariates") or {}
    if isinstance(covariates, list):
        return covariates
    return (
        covariates.get("numerical", [])
        + covariates.get("categorical", [])
        + covariates.get("binary", [])
    )


def build_r_command(
    r_args: list[str], r_runner: str, image: str, docker_cmd: str
) -> list[str]:
    """Wraps `train_stm.R` arguments for the local R or the STM container.

    Paths in `r_args` must be relative to the project root, which is the
    working directory in both cases.
    """
    script = ["scripts/r_scripts/train_stm.R", *r_args]
    if r_runner == "local":
        return ["Rscript", *script]
    threads = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    return [
        docker_cmd,
        "run",
        "--rm",
        "-v",
        f"{PROJECT_ROOT}:{CONTAINER_WORKDIR}",
        "-w",
        CONTAINER_WORKDIR,
        "-e",
        f"OMP_NUM_THREADS={threads}",
        "-e",
        f"OPENBLAS_NUM_THREADS={threads}",
        image,
        # The repository .Rprofile activates renv, which the image does not use.
        "Rscript",
        "--no-init-file",
        *script,
    ]


def read_r_outputs(output_dir: Path) -> dict:
    """Loads the files `train_stm.R` exports."""

    def read_matrix(name):
        return np.loadtxt(output_dir / name, delimiter=",", skiprows=1, ndmin=2)

    def read_lines(name):
        text = (output_dir / name).read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines()]

    return {
        "beta": read_matrix("beta.csv.gz"),
        "theta": read_matrix("theta.csv.gz"),
        "vocab": read_lines("vocab.txt"),
        "doc_index": [int(float(i)) for i in read_lines("doc_index.txt")],
        "duration": float(read_lines("duration.txt")[0]),
    }


def export_assignments(
    run_metadata: dict,
    documents: pl.DataFrame,
    theta: np.ndarray,
    covariates: pl.DataFrame,
    config: dict,
) -> dict:
    """Writes the document-topic assignments and their metadata alignment.

    STM gives each document a topic mixture. Like the hard-clustering models,
    each document is assigned its most probable topic; the proportion of that
    topic is kept as its strength.

    Returns:
        The run links (``run_uid`` and paths) and ``meta_ami_mean``.
    """
    run_uid = uuid.uuid4().hex
    directory = ASSIGNMENTS_DIR / run_metadata["dataset_name"] / run_uid
    directory.mkdir(parents=True, exist_ok=False)
    relative = directory.relative_to(PROJECT_ROOT)
    links = {
        "run_uid": run_uid,
        "assignment_manifest_path": (relative / "manifest.json").as_posix(),
        "assignments_path": (relative / "assignments.parquet").as_posix(),
    }

    labels = theta.argmax(axis=1)
    assignments = documents.select("index").with_columns(
        pl.Series("input_position", np.arange(len(labels))),
        pl.Series("topic_id", labels),
        pl.Series("topic_strength", theta.max(axis=1)),
        pl.lit(run_uid).alias("run_uid"),
    )
    assignments.write_parquet(directory / "assignments.parquet")

    alignment = metadata_alignment.metadata_alignment(labels, covariates)
    atomic_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            **run_metadata,
            **links,
            "resolved_config": config,
            "assignment_semantics": "argmax_theta",
            "strength_kind": "theta_max",
            "metadata_alignment": {**alignment, "method": metadata_alignment.METHOD},
            "document_count": len(assignments),
            "status": "success",
            "artifacts": {
                "assignments.parquet": {
                    "path": "assignments.parquet",
                    "sha256": file_checksum(directory / "assignments.parquet"),
                    "row_count": len(assignments),
                }
            },
        },
    )
    return {**links, "meta_ami_mean": alignment["meta_ami_mean"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run STM experiments.")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Name of the experiment yaml file (e.g., fed/fed_standard_stm)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Override the sample size specified in the config file.",
    )
    parser.add_argument(
        "--model",
        type=int,
        help="Run only the n-th (model, seed) run (1-indexed), as run_optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Run on a single specific seed, overriding seeds in the config.",
    )
    parser.add_argument(
        "--r-runner",
        choices=("local", "docker"),
        default="local",
        help="Run R from the local installation or in the STM image.",
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("STM_IMAGE", DEFAULT_STM_IMAGE),
        help="STM image for --r-runner docker (default: $STM_IMAGE or %(default)s).",
    )
    parser.add_argument(
        "--docker-cmd",
        default="docker",
        help="Container CLI for --r-runner docker (e.g. docker, podman).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Runs the selected STM models; returns non-zero if any run fails."""
    args = parse_args(argv)

    config = utils.load_config(args.exp, EXPERIMENTS_DIR)
    experiment = config["experiment"]
    if args.sample is not None:
        experiment["sample_size"] = args.sample

    exp_name = experiment["name"]
    sample_size = experiment.get("sample_size")
    if sample_size:
        exp_name = f"{exp_name}_s{sample_size}"

    random_state = utils.get_random_state(experiment["random_state"])
    seeds = random_state if isinstance(random_state, list) else [random_state]
    if args.seed is not None:
        seeds = [args.seed]

    logger = logger_config.setup_logging(exp_name, LOG_DIR)
    logger.info(f"Starting STM experiment: {exp_name} (R runner: {args.r_runner})")

    runs = expand_runs(config, seeds)
    if args.model is not None:
        if not 1 <= args.model <= len(runs):
            logger.error(f"--model {args.model} is out of range (1-{len(runs)}).")
            return 1
        selected = [runs[args.model - 1]]
    else:
        selected = runs

    dataset_name, rds_path, bow_path = resolve_stm_inputs(experiment)
    logger.info(f"Dataset: {dataset_name} | RDS: {rds_path} | BoW: {bow_path}")
    for path in (rds_path, bow_path):
        if not path.exists():
            logger.error(
                f"Input not found: {path}. Run scripts/data_prep/build_bow_tokens.py "
                "and scripts/r_scripts/build_bow.R first."
            )
            return 1

    # The BoW parquet holds the RDS documents, in the same order.
    documents = pl.read_parquet(bow_path)
    sample_seed = seeds[0]
    if sample_size and len(documents) > sample_size:
        sampled = documents.sample(n=sample_size, seed=sample_seed)["index"]
        # Keep the RDS order, which is the order R returns theta in.
        documents = documents.filter(pl.col("index").is_in(sampled.implode()))
        logger.info(f"Sampled {len(documents)} documents with seed {sample_seed}.")
    elif sample_size:
        logger.warning(f"Sample size {sample_size} >= {len(documents)}; not sampled.")
    logger.info(f"Documents for training: {len(documents)}")

    tokenized_texts = [text.split() for text in documents["bow_text"]]
    dictionary = Dictionary(tokenized_texts)
    covariates = documents.select(covariate_columns(experiment))

    prevalence_formula = experiment.get("prevalence_formula")
    if not prevalence_formula:
        logger.warning("No prevalence formula provided. Running vanilla STM.")

    is_stemmed = "stemmed" in exp_name.lower()
    tag = "stemmed" if is_stemmed else "remove_rep_stopwords"
    fn_base = exp_name if tag in exp_name else f"{exp_name}_{tag}"
    if args.model is not None:
        fn_base = f"{fn_base}_m{args.model}"
    file_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_filename = f"{fn_base}-{file_timestamp}-{seeds[0]}"
    start_timestamp = datetime.datetime.now().isoformat()
    git_rev, git_dirty = run_provenance.get_git_info()

    EXCHANGE_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    results, qualitative_dfs, failures = [], [], 0

    for model_config, seed in selected:
        m_id = model_config.get("id", "stm")
        k = model_config.get("parameters", {}).get("k")
        run_id = f"{m_id}_seed{seed}" if len(seeds) > 1 else m_id
        if not k:
            logger.error(f"[{run_id}] Missing parameter 'k'. Skipping.")
            failures += 1
            continue
        logger.info(f"--- Running model: {run_id} (K={k}) with seed {seed} ---")

        exchange = EXCHANGE_DIR / uuid.uuid4().hex
        exchange.mkdir()
        try:
            model_path = (
                MODELS_DIR / f"stm_{dataset_name}_{run_id}_{file_timestamp}.rds"
            )
            r_args = [
                "--rds_path",
                rds_path.relative_to(PROJECT_ROOT).as_posix(),
                "--k",
                str(k),
                "--output_dir",
                exchange.relative_to(PROJECT_ROOT).as_posix(),
                "--seed",
                str(seed),
                "--model_path",
                model_path.relative_to(PROJECT_ROOT).as_posix(),
            ]
            if sample_size:
                indices_path = exchange / "indices.txt"
                indices_path.write_text(
                    "\n".join(str(i) for i in documents["index"]), encoding="utf-8"
                )
                r_args += [
                    "--indices_path",
                    indices_path.relative_to(PROJECT_ROOT).as_posix(),
                ]
            if prevalence_formula:
                r_args += ["--prevalence_formula", prevalence_formula]

            cmd = build_r_command(r_args, args.r_runner, args.image, args.docker_cmd)
            logger.info(f"[{run_id}] Running R: {' '.join(cmd[:3])} ...")
            result = subprocess.run(
                cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8"
            )
            if result.returncode != 0:
                logger.error(
                    f"[{run_id}] R training failed (exit {result.returncode}):\n"
                    f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
                )
                failures += 1
                continue
            for line in result.stdout.splitlines():
                if any(marker in line for marker in R_LOG_MARKERS):
                    logger.info(f"[{run_id}] R: {line}")

            outputs = read_r_outputs(exchange)
            if outputs["doc_index"] != documents["index"].to_list():
                raise ValueError("R documents are not aligned with the BoW parquet.")
            beta, theta = outputs["beta"], outputs["theta"]
            logger.info(
                f"[{run_id}] Duration: {outputs['duration']:.2f}s, "
                f"vocab: {len(outputs['vocab'])}, beta: {beta.shape}"
            )

            top_words = evaluation.get_top_words_from_beta(beta, outputs["vocab"])
            octis_output = evaluation.topic_words_to_octis(top_words)
            metrics = {
                "model_name": run_id,
                "duration_seconds": outputs["duration"],
                "n_topics": k,
                "outliers": 0,
                **evaluation.topic_diagnostics(octis_output, dictionary=dictionary),
                "evaluation_protocol": evaluation.EVALUATION_PROTOCOL,
            }
            for cm in experiment["coherence_metrics"]:
                metrics[cm] = evaluation.compute_coherence(
                    model_output=octis_output,
                    texts=tokenized_texts,
                    measure=cm,
                    dictionary=dictionary,
                )
            for dm in experiment["diversity_metrics"]:
                metrics[dm] = evaluation.compute_diversity(
                    dm, model_output=octis_output
                )

            run_metadata = {
                "experiment_id": exp_name,
                "random_state": seed,
                "clustering_algo": "STM",
                "dim_red_algo": "None",
                "n_observations": len(documents),
                "timestamp": start_timestamp,
                "file_timestamp": file_timestamp,
                "dataset_name": dataset_name,
                "requested_topics": k,
                "k": k,
                "text_column": experiment.get("text_col"),
                "stopword_removal": tag,
            }
            assignment = export_assignments(
                {"model_id": run_id, "seed": seed, **run_metadata},
                documents,
                theta,
                covariates,
                config,
            )
            metrics.update(run_metadata)
            metrics.update(assignment)
            metrics.update(
                {
                    "result_schema_version": run_provenance.PROVENANCE_SCHEMA_VERSION,
                    "campaign_id": STM_CAMPAIGN_ID,
                    "run_status": "success",
                    "resolved_config_hash": run_provenance.compute_config_hash(
                        model_config
                    ),
                    "run_manifest_path": assignment["assignment_manifest_path"],
                    "code_revision": git_rev,
                    "code_dirty": git_dirty,
                    "dependency_lock_hash": run_provenance.get_dependency_lock_hash(),
                    "r_runner": args.r_runner,
                    "stm_image": args.image if args.r_runner == "docker" else None,
                }
            )
            results.append(metrics)
            qualitative_dfs.append(
                utils.extract_stm_qualitative_data(
                    theta=theta,
                    beta=beta,
                    vocab=outputs["vocab"],
                    documents=documents["bow_text"].to_list(),
                    model_id=run_id,
                    metadata=run_metadata,
                    top_words=top_words,
                )
            )

            # Saved after every run so a later failure keeps finished ones.
            pl.DataFrame(results).write_csv(RESULTS_DIR / f"{results_filename}.csv")
            output_path = OUTPUT_DIR / f"{results_filename}.json"
            qualitative = pl.concat(qualitative_dfs, how="diagonal")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json.loads(qualitative.write_json()), f, indent=4)
            logger.info(f"[{run_id}] Saved results to {results_filename}.csv/.json")
        except Exception:
            logger.exception(f"[{run_id}] Evaluation failed.")
            failures += 1
        finally:
            shutil.rmtree(exchange, ignore_errors=True)

    logger.info(f"Finished: {len(results)} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
