"""Experiment queue domain logic for selecting and submitting experiments.

This module provides pure and unit-testable experiment-queue logic, including:
- parsing run and model index specifications (e.g., 1..15, 1-5, 1,2,5);
- resolving datasets and models (including categories and substring expansion);
- applying model exclusions;
- constructing the job list for SLURM submission (split vs standard).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("pipeline")

PROJECT_NAME = "ca_bertopic"

DEFAULT_DATASETS = ("anes", "fed", "gadarian", "yelp")

DEFAULT_MEM = "32G"
DEFAULT_CPUS = 4
DEFAULT_TIME = "24:00:00"

STM_MEM = "16G"
STM_CPUS = 4
STM_TIME = "24:00:00"

ALL_MODELS = (
    "aligned_umap",
    "aligned_umap_mv_k_means",
    "aligned_umap_mv_spherical_k_means",
    "append_umap",
    "append_umap_mv_co_reg_spectral",
    "append_umap_mv_k_means",
    "append_umap_mv_spectral",
    "append_umap_mv_spectral_info0",
    "append_umap_mv_spherical_k_means",
    "append_umap_w000",
    "append_umap_w005",
    "append_umap_w010",
    "append_umap_w020",
    "append_umap_w030",
    "append_umap_w050",
    "baseline",
    "feature_stacking_hdbscan",
    "fast_tritopic",
    "k_means",
    "mv_co_reg_spectral",
    "mv_hdbscan",
    "mv_k_means",
    "mv_spectral",
    "mv_spectral_info0",
    "mv_spherical_k_means",
    "pca_k_means",
    "pca_mv_co_reg_spectral",
    "pca_mv_k_means",
    "pca_mv_spectral",
    "pca_mv_spherical_k_means",
    "stm",
    "tritopic",
    "umap_spectral",
)

DEFAULT_MODEL_INDICES = tuple(range(1, 16))


@dataclass(frozen=True)
class SlurmJobConfig:
    """Configuration for a single resolved SLURM experiment job."""

    dataset: str
    model: str
    exp_dir: str
    job_dataset: str
    job_name: str
    model_idx: int | None = None
    keep_rep_stopwords: bool = False
    mem: str = DEFAULT_MEM
    cpus: int = DEFAULT_CPUS
    time_limit: str = DEFAULT_TIME
    project_name: str = PROJECT_NAME
    reservation: str | None = None

    @property
    def rep_flag(self) -> str:
        """Return the representation stopwords CLI flag."""
        return (
            "--keep-rep-stopwords"
            if self.keep_rep_stopwords
            else "--remove-rep-stopwords"
        )

    @property
    def run_command(self) -> str:
        """Construct the Python optimizer run command."""
        exp_target = f"{self.exp_dir}/{self.dataset}_standard_{self.model}"
        if self.model_idx is not None:
            return (
                f"uv run python scripts/experiments/run_optimizer.py "
                f"--exp {exp_target} --model {self.model_idx} {self.rep_flag}"
            )
        return (
            f"uv run python scripts/experiments/run_optimizer.py "
            f"--exp {exp_target} {self.rep_flag}"
        )

    @property
    def data_copy_command(self) -> str:
        """Construct the rsync data copy command for this dataset."""
        if self.dataset == "yelp":
            src_file = (
                f"$HOME/{self.project_name}/data/processed/"
                "yelp_s10000_embeddings.parquet"
            )
            dst_file = "data/processed/yelp_embeddings.parquet"
            return f"rsync -a {src_file} {dst_file}"

        src_file = (
            f"$HOME/{self.project_name}/data/processed/"
            f"{self.dataset}_embeddings.parquet"
        )
        return f"rsync -a {src_file} data/processed/"

    @property
    def sbatch_args(self) -> list[str]:
        """Construct the list of sbatch CLI options for this job."""
        args = [
            f"--job-name={self.job_name}",
            "--partition=cidia",
            "--nodes=1",
            "--ntasks=1",
            f"--mem={self.mem}",
            f"--cpus-per-task={self.cpus}",
            f"--time={self.time_limit}",
            "--output=slurm_log/%x_%j.out",
            "--error=slurm_log/%x_%j.err",
        ]
        if self.reservation:
            args.append(f"--reservation={self.reservation}")
        return args

    @property
    def worker_args(self) -> list[str]:
        """Construct positional arguments passed to slurm_job.sh."""
        args = [
            self.dataset,
            f"{self.exp_dir}/{self.dataset}_standard_{self.model}",
            self.rep_flag,
        ]
        if self.model_idx is not None:
            args.append(str(self.model_idx))
        return args

    def full_sbatch_command(
        self, worker_script_path: str = "scripts/experiments/slurm_job.sh"
    ) -> list[str]:
        """Construct the complete sbatch command invocation."""
        return ["sbatch", *self.sbatch_args, worker_script_path, *self.worker_args]


@dataclass(frozen=True)
class QueuePlan:
    """Complete plan of datasets, models, and jobs to execute."""

    datasets: tuple[str, ...]
    models: tuple[str, ...]
    split: bool
    model_indices: tuple[int, ...]
    use_stemmed: bool
    keep_rep_stopwords: bool
    excludes: tuple[str, ...]
    dry_run: bool
    jobs: tuple[SlurmJobConfig, ...]
    reservation: str | None = None

    @property
    def total_jobs(self) -> int:
        """Return the total number of jobs to be submitted."""
        return len(self.jobs)


def parse_run_indices(raw_runs: str | None) -> list[int]:
    """Parse run or model index specifications.

    Accepts formats such as:
    - '1..15'
    - '1-5'
    - '1,2,5'
    - '1..3,7,9-10'

    Args:
        raw_runs: Comma-separated list or ranges of indices.

    Returns:
        List of integer indices.
    """
    if raw_runs is None or not raw_runs.strip():
        return list(DEFAULT_MODEL_INDICES)

    indices: list[int] = []
    parts = raw_runs.split(",")
    for part in parts:
        part_clean = part.strip()
        if not part_clean:
            continue

        range_match = re.match(r"^([0-9]+)(?:\.\.|\-)([0-9]+)$", part_clean)
        if range_match:
            start_idx = int(range_match.group(1))
            end_idx = int(range_match.group(2))
            if start_idx <= end_idx:
                indices.extend(range(start_idx, end_idx + 1))
        elif re.match(r"^[0-9]+$", part_clean):
            indices.append(int(part_clean))
        else:
            logger.warning(
                "Unrecognized run index or range '%s'. Ignoring.", part_clean
            )

    return indices


def resolve_datasets(raw_datasets: str | None, is_test: bool = False) -> list[str]:
    """Resolve target datasets based on user input or defaults.

    Args:
        raw_datasets: Comma-separated dataset names.
        is_test: Whether test mode is enabled.

    Returns:
        List of target dataset names in order.
    """
    if raw_datasets is not None and raw_datasets.strip():
        datasets = [d.strip() for d in raw_datasets.split(",") if d.strip()]
        return datasets
    if is_test:
        return ["fed"]
    return list(DEFAULT_DATASETS)


def resolve_models(
    raw_models: str | None = None,
    is_test: bool = False,
    raw_exact_models: str | None = None,
) -> list[str]:
    """Resolve model list expanding categories, matching, or exact models.

    Preserves legacy category mappings:
    - kmeans/k_means: matches any model with 'k_means' in name (including spherical)
    - spherical: matches any model with 'spherical' in name
    - pca: matches any model with 'pca' in name
    - spectral: matches any model with 'spectral' in name
    - umap: matches any model with 'umap' in name
    - tritopic: matches any model with 'tritopic' in name

    For any non-category token, matches exact name or substring in ALL_MODELS.
    When raw_exact_models is specified, models are matched strictly 1:1.

    Args:
        raw_models: Comma-separated model names or categories.
        is_test: Whether test mode is enabled.
        raw_exact_models: Comma-separated exact model names (no expansion).

    Returns:
        List of unique models preserving resolution order.

    Raises:
        ValueError: If an exact model name is unknown.
    """
    exact_models: list[str] = []
    if raw_exact_models is not None and raw_exact_models.strip():
        for item in raw_exact_models.split(","):
            item_clean = item.strip()
            if not item_clean:
                continue
            if item_clean not in ALL_MODELS:
                valid_models_str = ", ".join(ALL_MODELS)
                raise ValueError(
                    f"Error: Unknown exact model '{item_clean}'. "
                    f"Valid models are: {valid_models_str}"
                )
            if item_clean not in exact_models:
                exact_models.append(item_clean)

    if not (raw_models and raw_models.strip()):
        if exact_models:
            return exact_models
        if is_test:
            return ["baseline", "stm"]
        return list(ALL_MODELS)

    if raw_models is not None and raw_models.strip():
        initial_models: list[str] = []
        for item in raw_models.split(","):
            item_clean = item.strip()
            if not item_clean:
                continue

            matched = False
            if item_clean in ("kmeans", "k_means"):
                for m in ALL_MODELS:
                    if "k_means" in m:
                        initial_models.append(m)
                        matched = True
            elif item_clean == "spherical":
                for m in ALL_MODELS:
                    if "spherical" in m:
                        initial_models.append(m)
                        matched = True
            elif item_clean == "pca":
                for m in ALL_MODELS:
                    if "pca" in m:
                        initial_models.append(m)
                        matched = True
            elif item_clean == "spectral":
                for m in ALL_MODELS:
                    if "spectral" in m:
                        initial_models.append(m)
                        matched = True
            elif item_clean == "umap":
                for m in ALL_MODELS:
                    if "umap" in m:
                        initial_models.append(m)
                        matched = True
            elif item_clean == "tritopic":
                for m in ALL_MODELS:
                    if "tritopic" in m:
                        initial_models.append(m)
                        matched = True

            if not matched:
                for m in ALL_MODELS:
                    if m == item_clean or item_clean in m:
                        initial_models.append(m)
                        matched = True

            if not matched:
                logger.warning(
                    "Warning: Model or category '%s' did not match any known model.",
                    item_clean,
                )

        # Deduplicate preserving order of first appearance
        target_models: list[str] = []
        for m in initial_models:
            if m not in target_models:
                target_models.append(m)
        for m in exact_models:
            if m not in target_models:
                target_models.append(m)
        return target_models

    if exact_models:
        return exact_models
    return list(ALL_MODELS)


def apply_exclusions(models: list[str], raw_excludes: str | None) -> list[str]:
    """Filter out models matching exclusion patterns.

    Args:
        models: Initial list of resolved models.
        raw_excludes: Comma-separated exclusion keywords.

    Returns:
        Filtered list of models.

    Raises:
        ValueError: If no models remain after applying exclusions.
    """
    exclude_list: list[str] = []
    if raw_excludes is not None and raw_excludes.strip():
        for ex in raw_excludes.split(","):
            ex_clean = ex.strip()
            if ex_clean:
                if ex_clean == "kmeans":
                    ex_clean = "k_means"
                exclude_list.append(ex_clean)

    final_models: list[str] = []
    for m in models:
        excluded = any(ex in m for ex in exclude_list)
        if not excluded:
            final_models.append(m)

    if not final_models:
        raise ValueError(
            "Error: No models selected after applying filters and exclusions."
        )

    return final_models


def build_jobs(
    datasets: list[str],
    models: list[str],
    split: bool,
    model_indices: list[int],
    use_stemmed: bool,
    keep_rep_stopwords: bool,
    mem: str = DEFAULT_MEM,
    cpus: int = DEFAULT_CPUS,
    time_limit: str = DEFAULT_TIME,
    project_name: str = PROJECT_NAME,
    reservation: str | None = None,
) -> list[SlurmJobConfig]:
    """Build list of SlurmJobConfig objects for all dataset and model combinations.

    Note that STM models are currently skipped with a warning, matching
    legacy behavior.

    Args:
        datasets: Target datasets.
        models: Target models.
        split: Whether to split into per-model-index runs.
        model_indices: Model indices to run when split is True.
        use_stemmed: Whether to use stemmed dataset variant.
        keep_rep_stopwords: Whether to keep representation stopwords.
        mem: SLURM memory allocation.
        cpus: SLURM CPU allocation.
        time_limit: SLURM time allocation.
        project_name: Name of project repository.
        reservation: Optional SLURM reservation name.

    Returns:
        List of SlurmJobConfig objects.
    """
    jobs: list[SlurmJobConfig] = []

    for dataset in datasets:
        exp_dir = f"{dataset}_stemmed" if use_stemmed else dataset
        job_dataset = f"{dataset}_stemmed" if use_stemmed else dataset

        for model in models:
            if model == "stm":
                logger.warning(
                    "Warning: Model 'stm' requested for dataset '%s', "
                    "but STM jobs are currently disabled. Skipping.",
                    dataset,
                )
                continue

            if split:
                for model_idx in model_indices:
                    job_name = f"{job_dataset}_{model}_m{model_idx}"
                    jobs.append(
                        SlurmJobConfig(
                            dataset=dataset,
                            model=model,
                            exp_dir=exp_dir,
                            job_dataset=job_dataset,
                            job_name=job_name,
                            model_idx=model_idx,
                            keep_rep_stopwords=keep_rep_stopwords,
                            mem=mem,
                            cpus=cpus,
                            time_limit=time_limit,
                            project_name=project_name,
                            reservation=reservation,
                        )
                    )
            else:
                job_name = f"{job_dataset}_{model}"
                jobs.append(
                    SlurmJobConfig(
                        dataset=dataset,
                        model=model,
                        exp_dir=exp_dir,
                        job_dataset=job_dataset,
                        job_name=job_name,
                        model_idx=None,
                        keep_rep_stopwords=keep_rep_stopwords,
                        mem=mem,
                        cpus=cpus,
                        time_limit=time_limit,
                        project_name=project_name,
                        reservation=reservation,
                    )
                )

    return jobs


def create_queue_plan(
    raw_datasets: str | None,
    raw_models: str | None,
    raw_excludes: str | None,
    raw_runs: str | None,
    split: bool,
    use_stemmed: bool,
    keep_rep_stopwords: bool,
    is_test: bool = False,
    dry_run: bool = False,
    mem: str = DEFAULT_MEM,
    cpus: int = DEFAULT_CPUS,
    time_limit: str = DEFAULT_TIME,
    project_name: str = PROJECT_NAME,
    allow_empty_datasets: bool = False,
    raw_exact_models: str | None = None,
    reservation: str | None = None,
) -> QueuePlan:
    """Create a fully resolved QueuePlan.

    Args:
        raw_datasets: User-specified datasets string.
        raw_models: User-specified models string.
        raw_excludes: User-specified exclusion string.
        raw_runs: User-specified run indices string.
        split: Whether split mode is requested.
        use_stemmed: Whether stemmed variant is requested.
        keep_rep_stopwords: Whether representation stopwords should be kept.
        is_test: Whether test mode is active.
        dry_run: Whether dry-run mode is active.
        mem: SLURM memory allocation.
        cpus: SLURM CPU allocation.
        time_limit: SLURM time allocation.
        project_name: Project name.
        allow_empty_datasets: For internal testing of dataset validation.
        raw_exact_models: User-specified exact model names.
        reservation: Optional SLURM reservation name.

    Returns:
        Resolved QueuePlan.

    Raises:
        ValueError: If datasets or models resolve to empty sets.
    """
    if allow_empty_datasets:
        datasets: list[str] = []
    else:
        datasets = resolve_datasets(raw_datasets, is_test=is_test)

    if not datasets:
        raise ValueError("Error: No datasets selected.")

    indices = parse_run_indices(raw_runs)

    initial_models = resolve_models(
        raw_models, is_test=is_test, raw_exact_models=raw_exact_models
    )
    final_models = apply_exclusions(initial_models, raw_excludes)

    excludes_tuple: tuple[str, ...] = ()
    if raw_excludes and raw_excludes.strip():
        excludes_tuple = tuple(e.strip() for e in raw_excludes.split(",") if e.strip())

    clean_reservation = (
        reservation.strip() if reservation and reservation.strip() else None
    )

    jobs = build_jobs(
        datasets=datasets,
        models=final_models,
        split=split,
        model_indices=indices,
        use_stemmed=use_stemmed,
        keep_rep_stopwords=keep_rep_stopwords,
        mem=mem,
        cpus=cpus,
        time_limit=time_limit,
        project_name=project_name,
        reservation=clean_reservation,
    )

    return QueuePlan(
        datasets=tuple(datasets),
        models=tuple(final_models),
        split=split,
        model_indices=tuple(indices),
        use_stemmed=use_stemmed,
        keep_rep_stopwords=keep_rep_stopwords,
        excludes=excludes_tuple,
        dry_run=dry_run,
        jobs=tuple(jobs),
        reservation=clean_reservation,
    )
