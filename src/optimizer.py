import copy
import itertools
import json
import logging
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl

import src.models as models
import src.run_provenance as run_provenance
import src.training as training
from src.document_assignments import AssignmentRun, requested_topic_setting


def collect_hyperparameters(
    model_config: Dict[str, Any],
) -> Tuple[List[List[str]], List[List[Any]]]:
    """
    Collects hyperparameter search spaces from the model configuration.

    This function identifies hyperparameters to be tuned by looking for lists
    of values or range specifications (dict with start, stop, step) within
    the 'dimensionality_reduction' and 'clustering' parameter sections of
    the model config.

    Args:
        model_config: The model configuration dictionary.

    Returns:
        A tuple containing two lists:
        - param_paths: A list of paths to the hyperparameters in the config dict.
        - param_values: A list of lists, where each inner list contains the
          values to be tested for the corresponding hyperparameter.
    """
    param_paths = []
    param_values = []

    def collect_from_component(component_name: str):
        component = model_config.get(component_name, {})
        params = component.get("params") or {}
        # Sort keys to ensure deterministic order of combinations
        for key in sorted(params.keys()):
            value = params[key]
            path = [component_name, "params", key]

            # Atomic sequence parameters should not be expanded as grids unless nested
            if (
                key
                in (
                    "view_metrics",
                    "view_weights",
                    "view_affinities",
                    "weights",
                    "metric",
                    "keyword_ngram_range",
                )
                and isinstance(value, list)
                and (len(value) == 0 or not isinstance(value[0], (list, tuple)))
            ):
                continue

            # A list of values is considered a hyperparameter to vary
            if isinstance(value, list) and len(value) > 1:
                param_paths.append(path)
                param_values.append(value)

            # A dictionary with start/stop is a range
            elif isinstance(value, dict) and "start" in value and "stop" in value:
                start = value["start"]
                stop = value["stop"]
                step = value.get("step", 1)

                # Use np.arange for float support and consistency
                generated_values = np.arange(start, stop, step).tolist()

                param_paths.append(path)
                param_values.append(generated_values)

    collect_from_component("dimensionality_reduction")
    collect_from_component("clustering")
    collect_from_component("bertopic")
    collect_from_component("tritopic")
    collect_from_component("fast_tritopic")

    # Also collect from top-level params if present
    params = model_config.get("params") or {}
    for key in sorted(params.keys()):
        value = params[key]
        path = ["params", key]
        # A flat n-gram pair is one setting; nested pairs remain a search grid.
        if (
            key == "keyword_ngram_range"
            and isinstance(value, list)
            and (not value or not isinstance(value[0], (list, tuple)))
        ):
            continue
        if isinstance(value, list) and len(value) > 1:
            param_paths.append(path)
            param_values.append(value)

    return param_paths, param_values


def generate_hyperparameter_combinations(
    model_config: Dict[str, Any],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Generates all possible hyperparameter combinations for the search.
    """
    param_paths, param_values = collect_hyperparameters(model_config)

    if not param_paths:
        return [(model_config, {})]

    combinations = []
    # Create the Cartesian product of all hyperparameter values.
    # For each resulting combination, create a new model configuration.
    for value_combination in itertools.product(*param_values):
        new_config = copy.deepcopy(model_config)
        varied_params = {}
        for path, value in zip(param_paths, value_combination):
            # Set the specific hyperparameter value in the new config copy
            curr = new_config
            for p in path[:-1]:
                curr = curr[p]
            curr[path[-1]] = value
            # Keep track of the parameters that were varied for this run
            varied_params[".".join(path)] = value
        combinations.append((new_config, varied_params))

    return combinations


def clean_varied_params(varied_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans up parameter names for reporting and results.
    """
    return {
        key.replace("clustering.params.", "")
        .replace("dimensionality_reduction.params.", "")
        .replace("bertopic.params.", "")
        .replace("tritopic.params.", "")
        .replace("fast_tritopic.params.", "")
        .replace("params.", ""): value
        for key, value in varied_params.items()
    }


class Optimizer:
    """
    Orchestrates a hyperparameter search by training and evaluating multiple
    BERTopic models based on a single model architecture with multiple
    hyperparameter options.
    """

    def __init__(
        self,
        texts: list[str],
        embeddings: Any,
        scaled_metadata: Any,
        model_config: Dict[str, Any],
        experiment_config: Dict[str, Any],
        experiment_id: str,
        random_state: int,
        file_timestamp: str,
        remove_rep_stopwords: bool = False,
        prepared_data=None,
        assignment_output_dir=None,
    ):
        """
        Initializes the Optimizer.
        Args:
            texts: A list of strings with the texts to be analyzed
            embeddings: A np.ndarray with the document embeddings
            scaled_metadata: A np.ndarray with the document metadata
            model_config: A dictionary with the model architecture and
                hyperparameters. Hyperparameters with multiple values
                should be in a list.
            experiment_config: The global experiment configuration.
            experiment_id: The identifier for the experiment.
            random_state: The random seed used for the experiment.
            file_timestamp: The timestamp used in the results filename.
            remove_rep_stopwords: If True, removes English stop words from
                c-TF-IDF topic representations using CountVectorizer.
            prepared_data: Source-aligned loader result. Required for assignment
                export; legacy array-only callers retain their existing behavior.
            assignment_output_dir: Root of the per-execution assignment subtree.
        """
        self.texts = texts
        self.embeddings = embeddings
        self.scaled_metadata = scaled_metadata
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.experiment_id = experiment_id
        self.random_state = random_state
        self.file_timestamp = file_timestamp
        self.remove_rep_stopwords = remove_rep_stopwords
        self.prepared_data = prepared_data
        self.assignment_output_dir = (
            pathlib.Path(assignment_output_dir)
            if assignment_output_dir is not None
            else run_provenance.PROJECT_ROOT / "output" / "document_assignments"
        )
        if prepared_data is not None:
            # Use the exact materialized inputs whose identities were captured.
            self.texts, self.embeddings, self.scaled_metadata = prepared_data
        self.results = []
        self.qualitative_results = []
        self.run_manifests = []
        self.logger = logging.getLogger("pipeline")

    def run(self, start_index: int = 0, target_index: int | None = None) -> None:
        """
        Executes the optimization process: iterates through model configs,
        trains each one, and stores the evaluation metrics.

        Args:
            start_index: The index of the first configuration to evaluate.
                         Allows resuming interrupted runs.
            target_index: If provided, only this specific configuration index
                          will be executed.
        """
        import datetime

        import src.utils as utils

        if self.prepared_data is None:
            self.logger.warning(
                "Assignment export unavailable: Optimizer received no prepared_data. "
                "Use run_optimizer.py or supply source-aligned prepared inputs."
            )

        hyperparameter_combinations = generate_hyperparameter_combinations(
            self.model_config
        )

        # Determine all seeds
        seeds = (
            self.random_state
            if isinstance(self.random_state, list)
            else [self.random_state]
        )

        # Generate a flat list of runs: (combo_idx, model_config, varied_params, seed)
        all_runs = []
        for combo_idx, (model_config, varied_params) in enumerate(
            hyperparameter_combinations
        ):
            for seed in seeds:
                all_runs.append(
                    (
                        combo_idx,
                        copy.deepcopy(model_config),
                        copy.deepcopy(varied_params),
                        seed,
                    )
                )

        num_runs = len(all_runs)

        if target_index is not None:
            if target_index < 0 or target_index >= num_runs:
                self.logger.error(
                    f"Target index {target_index + 1} is out of range (1-{num_runs})."
                )
                return
            run_runs = [all_runs[target_index]]
            self.logger.info(
                f"Running specific model configuration index {target_index + 1} "
                f"of {num_runs}."
            )
        else:
            if start_index >= num_runs:
                self.logger.info(
                    f"Start index {start_index} is beyond total combinations "
                    f"{num_runs}. Nothing to do."
                )
                return

            run_runs = all_runs[start_index:]
            if start_index == 0:
                self.logger.info(
                    f"Starting hyperparameter optimization for {num_runs} models."
                )
            else:
                self.logger.info(
                    "Resuming hyperparameter optimization for "
                    f"{num_runs} models "
                    f"(starting at index {start_index + 1})."
                )

        # New Metadata Capture
        start_timestamp = datetime.datetime.now().isoformat()
        dataset_name = pathlib.Path(
            self.experiment_config["experiment"]["dataset_path"]
        ).stem.replace("_embeddings", "")
        n_observations = len(self.texts)

        try:
            for run_idx, (combo_idx, model_config, varied_params, seed) in enumerate(
                run_runs
            ):
                model_id = self.model_config.get("id", "model")
                if len(seeds) > 1:
                    run_id = f"{model_id}_{combo_idx + 1}_seed{seed}"
                else:
                    run_id = f"{model_id}_{combo_idx + 1}"

                # Clean up param names for reporting
                cleaned_varied_params = clean_varied_params(varied_params)
                self.logger.info(f"--- Training model [{run_id}] with seed {seed} ---")
                self.logger.info(f"Varied Parameters: {cleaned_varied_params}")

                # 1. Create Model Instance
                topic_model = models.create_topic_model_instance(
                    model_config=model_config,
                    scaled_metadata=self.scaled_metadata,
                    random_state=seed,
                    remove_rep_stopwords=self.remove_rep_stopwords,
                )

                # 2. Train and Evaluate
                assignment_run = None
                try:
                    train = training.train_and_evaluate
                    if self.prepared_data is not None:
                        regime = (
                            "stemmed"
                            if "stemmed" in self.experiment_id.lower()
                            else "remove_rep_stopwords"
                            if self.remove_rep_stopwords
                            else "keep_rep_stopwords"
                        )
                        assignment_run = AssignmentRun(
                            self.assignment_output_dir,
                            dataset_name,
                            self.prepared_data,
                            model_config,
                            {
                                "model_id": run_id,
                                "experiment_id": self.experiment_id,
                                "dataset_name": dataset_name,
                                "seed": seed,
                                "random_state": seed,
                                "stopword_removal": regime,
                                "file_timestamp": self.file_timestamp,
                                **cleaned_varied_params,
                            },
                            self.experiment_config,
                        )
                        train = assignment_run.execute
                    metrics, trained_model = train(
                        topic_model=topic_model,
                        model_id=run_id,
                        text=self.texts,
                        embeddings=self.embeddings,
                        config=self.experiment_config,
                        scaled_metadata=self.scaled_metadata,
                    )

                    # 3. Store results, including the varied hyperparameters
                    # and metadata
                    m_type = model_config.get("type", "tritopic")
                    clustering_algo = (
                        model_config.get("clustering", {}).get("type") or m_type
                    )
                    dim_red_algo = (
                        model_config.get("dimensionality_reduction", {}).get("type")
                        or f"{m_type}_internal"
                    )

                    run_metadata = {
                        "experiment_id": self.experiment_id,
                        "random_state": seed,
                        "clustering_algo": clustering_algo,
                        "dim_red_algo": dim_red_algo,
                        "n_observations": n_observations,
                        "timestamp": start_timestamp,
                        "file_timestamp": self.file_timestamp,
                        "dataset_name": dataset_name,
                        "requested_topics": requested_topic_setting(model_config),
                    }
                    if assignment_run is not None:
                        run_metadata.update(assignment_run.links)
                    # Measured values win over same-named grid parameters:
                    # TriTopic's `n_topics` parameter would otherwise replace
                    # the realized topic count. `requested_topics` keeps the
                    # requested value.
                    reported_params = {
                        key: value
                        for key, value in cleaned_varied_params.items()
                        if key not in metrics
                    }
                    metrics.update(run_metadata)
                    metrics.update(reported_params)

                    # 3b. Collect Provenance from fitted model and config
                    provenance = run_provenance.collect_run_provenance(
                        topic_model=trained_model,
                        model_config=model_config,
                        run_id=run_id,
                        run_status="success",
                        run_manifest_path=(
                            assignment_run.links["assignment_manifest_path"]
                            if assignment_run is not None
                            else None
                        ),
                    )
                    metrics.update(provenance)
                    self.results.append(metrics)

                    self.run_manifests.append(
                        {
                            "run_id": run_id,
                            "model_name": run_id,
                            "seed": seed,
                            "status": "success",
                            **(assignment_run.links if assignment_run else {}),
                            "provenance": provenance,
                            "varied_params": cleaned_varied_params,
                            "model_config": model_config,
                        }
                    )

                    # 4. Extract Qualitative Data
                    qual_metadata = run_metadata.copy()
                    qual_metadata.update(cleaned_varied_params)
                    self.qualitative_results.append(
                        utils.extract_qualitative_data(
                            trained_model, run_id, qual_metadata
                        )
                    )

                except Exception as e:
                    self.logger.error(
                        f"Failed to train model [{run_id}] with params "
                        f"{cleaned_varied_params} and seed {seed}: {e}"
                    )
                    self.run_manifests.append(
                        {
                            "run_id": run_id,
                            "model_name": run_id,
                            "seed": seed,
                            "status": "failure",
                            **(assignment_run.links if assignment_run else {}),
                            "error": str(e),
                            "varied_params": cleaned_varied_params,
                            "model_config": model_config,
                        }
                    )
                    continue
        except KeyboardInterrupt:
            self.logger.warning(
                "Optimization interrupted by user. Cleaning up and saving results..."
            )

        self.logger.info("Finished hyperparameter optimization phase.")

    def save_results(
        self, filepath: str | pathlib.Path, decimal_digits: int | None = None
    ) -> None:
        """
        Saves the collected evaluation metrics to a CSV file.
        Args:
            filepath: The path to the output CSV file.
            decimal_digits: Deprecated storage parameter. Full floating-point
                precision is preserved in CSV files. Retained for API compatibility.
        """
        if not self.results:
            self.logger.warning("No results to save. Run the optimization first.")
            return

        save_path = pathlib.Path(filepath)
        relative_manifest_path = f"logs/manifests/{save_path.stem}_manifest.json"

        # Update run_manifest_path in results if not set
        for res in self.results:
            if not res.get("run_manifest_path"):
                res["run_manifest_path"] = relative_manifest_path

        df = pl.DataFrame(self.results)

        # Reorder columns based on user's desired output format
        all_cols = df.columns

        core_stats_cols = [
            "experiment_id",
            "random_state",
            "file_timestamp",
            "model_name",
            "dataset_name",
            "timestamp",
            "n_observations",
            "clustering_algo",
            "dim_red_algo",
            "duration_seconds",
            "requested_topics",
            "n_topics",
            "outliers",
            "n_topics_short",
            "n_topics_unscored",
            "n_keywords_oov",
        ]

        # Calculated metrics from the experiment config
        exp_metrics = []
        if "experiment" in self.experiment_config:
            exp_metrics.extend(
                self.experiment_config["experiment"].get("coherence_metrics", [])
            )
            exp_metrics.extend(
                self.experiment_config["experiment"].get("diversity_metrics", [])
            )
        # Descriptive topic–metadata alignment (see src/metadata_alignment.py).
        exp_metrics.append("meta_ami_mean")
        exp_metrics.append("evaluation_protocol")

        provenance_cols = run_provenance.PROVENANCE_COLUMNS

        # Varied parameter columns are what's left over
        param_cols = [
            col
            for col in all_cols
            if col not in core_stats_cols
            and col not in exp_metrics
            and col not in provenance_cols
        ]

        # New order: core stats, then params, then calculated metrics, then provenance
        final_order = (
            [c for c in core_stats_cols if c in all_cols]
            + [p for p in param_cols if p in all_cols]
            + [m for m in exp_metrics if m in all_cols]
            + [pr for pr in provenance_cols if pr in all_cols]
        )
        final_order += [c for c in all_cols if c not in final_order]
        df = df.select(final_order)

        # Handle appending/merging if the file already exists
        if save_path.exists():
            try:
                existing_df = pl.read_csv(save_path, infer_schema_length=None)
                existing_campaign = (
                    existing_df["campaign_id"][0]
                    if "campaign_id" in existing_df.columns and len(existing_df) > 0
                    else None
                )
                current_campaign = (
                    df["campaign_id"][0]
                    if "campaign_id" in df.columns and len(df) > 0
                    else None
                )
                if existing_campaign != current_campaign:
                    self.logger.warning(
                        f"Existing results file {save_path} has incompatible "
                        f"campaign '{existing_campaign}' (current: "
                        f"'{current_campaign}'). Saving to a separate file to "
                        "prevent cross-campaign contamination."
                    )
                    save_path = save_path.with_name(
                        f"{save_path.stem}_{current_campaign or 'v2'}{save_path.suffix}"
                    )
                else:
                    df = pl.concat([existing_df, df], how="diagonal")
            except Exception as e:
                self.logger.error(
                    f"Failed to merge with existing results file: {e}. "
                    "Saving to a new file with suffix."
                )
                save_path = save_path.with_name(
                    f"{save_path.stem}_v2{save_path.suffix}"
                )

        # Stage 6: Preserve full precision Float64, do not truncate with float_precision
        df.write_csv(save_path)
        self.logger.info(f"Results saved to {save_path}")

        # Save Qualitative Data
        output_dir = save_path.parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.qualitative_results:
            consolidated_qual_df = pl.concat(self.qualitative_results, how="diagonal")
            output_path = output_dir / f"{save_path.stem}.json"

            # Handle merging if the qualitative file already exists
            if output_path.exists():
                try:
                    existing_qual_df = pl.read_json(
                        output_path, infer_schema_length=None
                    )
                    consolidated_qual_df = pl.concat(
                        [
                            existing_qual_df,
                            consolidated_qual_df.cast(existing_qual_df.schema),
                        ],
                        how="vertical",
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to merge with existing qualitative results file: {e}"
                    )

            # Serialize to JSON string and then pretty-print using the
            # standard json library
            json_str = consolidated_qual_df.write_json()
            parsed_json = json.loads(json_str)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=4)

            self.logger.info(f"Qualitative topic data saved at {output_path}")

        # Save Run Manifest
        manifest_dir = save_path.parent.parent / "logs" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{save_path.stem}_manifest.json"
        git_rev, git_dirty = run_provenance.get_git_info()
        manifest_payload = {
            "campaign_id": run_provenance.DEFAULT_CAMPAIGN_ID,
            "experiment_id": self.experiment_id,
            "file_timestamp": self.file_timestamp,
            "code_revision": git_rev,
            "code_dirty": git_dirty,
            "dependency_lock_hash": run_provenance.get_dependency_lock_hash(),
            "runs": self.run_manifests,
        }
        run_provenance.save_run_manifest(manifest_path, manifest_payload)
        self.logger.info(f"Run manifest saved at {manifest_path}")
