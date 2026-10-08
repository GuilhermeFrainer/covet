"""Experiment queue domain logic for selecting and submitting experiments.

This module provides pure and unit-testable experiment-queue logic, including:
- parsing run and model index specifications (e.g., 1..15, 1-5, 1,2,5);
- resolving datasets and models (including categories and substring expansion);
- applying model exclusions;
- constructing the job list for SLURM submission (split vs standard);
- packing several jobs into one SLURM job that runs them in sequence.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("pipeline")

PROJECT_NAME = "ca_bertopic"
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"

DEFAULT_DATASETS = ("anes", "fed", "gadarian", "yelp")

DEFAULT_MEM = "32G"
DEFAULT_CPUS = 4
DEFAULT_TIME = "24:00:00"

# STM trains in R on one core, inside the lightweight image built from
# Dockerfile.stm and loaded on each node from $HOME/docker_images/.
STM_MODEL = "stm"
STM_MEM = "16G"
STM_CPUS = 1
STM_TIME = "24:00:00"
STM_IMAGE = "cast:stm-lite-v0.2.0"

# Separates the worker invocations slurm_pack_job.sh runs in sequence.
PACK_SEPARATOR = "::"

_MEM_UNITS_MB = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}

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


def stm_input_prefix(dataset: str, use_stemmed: bool) -> str:
    """Returns the prefix of a dataset's STM inputs in data/processed.

    Yelp STM configs train on the aligned 10k sample, as the neural Yelp
    runs do.
    """
    base = "yelp_s10000" if dataset == "yelp" else dataset
    return f"{base}_stemmed" if use_stemmed else base


def parse_time_limit(value: str) -> int:
    """Converts a SLURM time limit to seconds.

    Accepts the sbatch formats MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM and
    D-HH:MM:SS.

    Raises:
        ValueError: If the value is not a SLURM time limit.
    """
    text = value.strip()
    try:
        days = 0
        clock = text
        if "-" in text:
            day_part, clock = text.split("-", 1)
            days = int(day_part)
        parts = [int(p) for p in clock.split(":")]
        if not 1 <= len(parts) <= 3:
            raise ValueError
        if "-" in text:
            hours, minutes, seconds = parts + [0] * (3 - len(parts))
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            hours = 0
            minutes, seconds = parts + [0] * (2 - len(parts))
    except ValueError:
        raise ValueError(f"Error: Invalid SLURM time limit '{value}'.") from None
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def format_time_limit(seconds: int) -> str:
    """Formats seconds as a SLURM time limit (D-HH:MM:SS or HH:MM:SS)."""
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}-{clock}" if days else clock


def parse_mem(value: str) -> float:
    """Converts a SLURM memory request such as 32G or 16000 to megabytes.

    Raises:
        ValueError: If the value is not a SLURM memory request.
    """
    match = re.fullmatch(r"([0-9]+)([KMGT]?)B?", value.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Error: Invalid SLURM memory request '{value}'.")
    return int(match.group(1)) * _MEM_UNITS_MB[(match.group(2) or "M").upper()]


def _sbatch_args(
    job_name: str, mem: str, cpus: int, time_limit: str, reservation: str | None
) -> list[str]:
    """Constructs the sbatch CLI options shared by single and packed jobs."""
    args = [
        f"--job-name={job_name}",
        "--partition=cidia",
        "--nodes=1",
        "--ntasks=1",
        f"--mem={mem}",
        f"--cpus-per-task={cpus}",
        f"--time={time_limit}",
        "--output=slurm_log/%x_%j.out",
        "--error=slurm_log/%x_%j.err",
    ]
    if reservation:
        args.append(f"--reservation={reservation}")
    return args


def count_config_runs(exp_target: str) -> int:
    """Counts the (model, seed) runs a model-list config defines.

    This is the range of `--model` indices `run_stm.py` accepts.
    """
    # Imported here so queueing other models needs only the standard library.
    from src.utils import get_random_state, load_config

    experiment = load_config(exp_target, EXPERIMENTS_DIR)
    seeds = get_random_state(experiment["experiment"]["random_state"])
    n_seeds = len(seeds) if isinstance(seeds, list) else 1
    return len(experiment.get("models", [])) * n_seeds


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
    def is_stm(self) -> bool:
        """Whether this job trains STM, which has its own worker."""
        return self.model == STM_MODEL

    @property
    def exp_target(self) -> str:
        """Return the experiment config path relative to experiments/."""
        return f"{self.exp_dir}/{self.dataset}_standard_{self.model}"

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
        """Construct the Python run command the worker executes."""
        model_arg = f" --model {self.model_idx}" if self.model_idx is not None else ""
        if self.is_stm:
            return (
                f"uv run python scripts/experiments/run_stm.py "
                f"--exp {self.exp_target}{model_arg} --r-runner docker"
            )
        return (
            f"uv run python scripts/experiments/run_optimizer.py "
            f"--exp {self.exp_target}{model_arg} {self.rep_flag}"
        )

    @property
    def stm_input_prefix(self) -> str:
        """Return the prefix of this job's STM RDS and BoW files."""
        return stm_input_prefix(self.dataset, self.job_dataset.endswith("_stemmed"))

    @property
    def data_copy_command(self) -> str:
        """Construct the rsync data copy command for this dataset."""
        if self.is_stm:
            src_dir = f"$HOME/{self.project_name}/data/processed"
            prefix = self.stm_input_prefix
            return (
                f"rsync -a {src_dir}/{prefix}_stm_data.rds "
                f"{src_dir}/{prefix}_bow.parquet data/processed/"
            )
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
        return _sbatch_args(
            self.job_name, self.mem, self.cpus, self.time_limit, self.reservation
        )

    @property
    def worker_args(self) -> list[str]:
        """Construct positional arguments for slurm_job.sh or slurm_stm_job.sh."""
        if self.is_stm:
            args = [self.stm_input_prefix, self.exp_target, STM_IMAGE]
        else:
            args = [self.dataset, self.exp_target, self.rep_flag]
        if self.model_idx is not None:
            args.append(str(self.model_idx))
        return args

    def full_sbatch_command(
        self, worker_script_path: str = "scripts/experiments/slurm_job.sh"
    ) -> list[str]:
        """Construct the complete sbatch command invocation."""
        return ["sbatch", *self.sbatch_args, worker_script_path, *self.worker_args]

    def worker_command(
        self, worker_script_path: str, stm_worker_script_path: str
    ) -> list[str]:
        """Construct this job's worker invocation inside a packed job."""
        script = stm_worker_script_path if self.is_stm else worker_script_path
        return [script, *self.worker_args]


@dataclass(frozen=True)
class SlurmPackConfig:
    """Several jobs that one SLURM job runs one after another.

    The packed job requests the largest memory and CPU count among its
    members, so each member gets at least what it would have alone.
    """

    jobs: tuple[SlurmJobConfig, ...]
    time_limit: str

    @property
    def job_name(self) -> str:
        """Return the first member's name, suffixed with the others' count."""
        first = self.jobs[0].job_name
        return f"{first}+{len(self.jobs) - 1}" if len(self.jobs) > 1 else first

    @property
    def mem(self) -> str:
        """Return the largest memory request among the members."""
        return max((job.mem for job in self.jobs), key=parse_mem)

    @property
    def cpus(self) -> int:
        """Return the largest CPU request among the members."""
        return max(job.cpus for job in self.jobs)

    @property
    def reservation(self) -> str | None:
        """Return the reservation the members share."""
        return self.jobs[0].reservation

    @property
    def sbatch_args(self) -> list[str]:
        """Construct the list of sbatch CLI options for the packed job."""
        return _sbatch_args(
            self.job_name, self.mem, self.cpus, self.time_limit, self.reservation
        )

    def full_sbatch_command(
        self,
        pack_script_path: str,
        worker_script_path: str,
        stm_worker_script_path: str,
    ) -> list[str]:
        """Construct the sbatch command that runs every member in sequence."""
        command = ["sbatch", *self.sbatch_args, pack_script_path]
        for i, job in enumerate(self.jobs):
            if i:
                command.append(PACK_SEPARATOR)
            command.extend(
                job.worker_command(worker_script_path, stm_worker_script_path)
            )
        return command


def pack_jobs(
    jobs: list[SlurmJobConfig] | tuple[SlurmJobConfig, ...],
    pack_size: int,
    time_limit: str | None = None,
) -> list[SlurmPackConfig]:
    """Groups consecutive jobs into packs of at most `pack_size` members.

    Args:
        jobs: Jobs in submission order.
        pack_size: Maximum number of jobs per pack.
        time_limit: Time limit of each packed job; None sums the members'.

    Returns:
        Packs in submission order.

    Raises:
        ValueError: If `pack_size` is below 1 or `time_limit` is invalid.
    """
    if pack_size < 1:
        raise ValueError("Error: --pack must be at least 1.")
    if time_limit is not None:
        parse_time_limit(time_limit)

    packs: list[SlurmPackConfig] = []
    for start in range(0, len(jobs), pack_size):
        members = tuple(jobs[start : start + pack_size])
        limit = time_limit or format_time_limit(
            sum(parse_time_limit(job.time_limit) for job in members)
        )
        packs.append(SlurmPackConfig(jobs=members, time_limit=limit))
    return packs


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
    packs: tuple[SlurmPackConfig, ...] = ()

    @property
    def total_jobs(self) -> int:
        """Return the total number of jobs (runs) in the plan."""
        return len(self.jobs)

    @property
    def total_submissions(self) -> int:
        """Return the number of SLURM jobs sbatch receives."""
        return len(self.packs) if self.packs else len(self.jobs)


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
    mem: str | None = None,
    cpus: int | None = None,
    time_limit: str | None = None,
    project_name: str = PROJECT_NAME,
    reservation: str | None = None,
    count_runs: Callable[[str], int] = count_config_runs,
) -> list[SlurmJobConfig]:
    """Build list of SlurmJobConfig objects for all dataset and model combinations.

    STM jobs default to the STM resources, and split mode submits only the
    run indices their config defines. STM has no bag-of-words that keeps
    stopwords, so it is skipped when representation stopwords are kept.

    Args:
        datasets: Target datasets.
        models: Target models.
        split: Whether to split into per-model-index runs.
        model_indices: Model indices to run when split is True.
        use_stemmed: Whether to use stemmed dataset variant.
        keep_rep_stopwords: Whether to keep representation stopwords.
        mem: SLURM memory allocation; None uses the model's default.
        cpus: SLURM CPU allocation; None uses the model's default.
        time_limit: SLURM time allocation; None uses the model's default.
        project_name: Name of project repository.
        reservation: Optional SLURM reservation name.
        count_runs: Returns the number of runs an experiment target defines.

    Returns:
        List of SlurmJobConfig objects.
    """
    jobs: list[SlurmJobConfig] = []

    for dataset in datasets:
        exp_dir = f"{dataset}_stemmed" if use_stemmed else dataset
        job_dataset = f"{dataset}_stemmed" if use_stemmed else dataset

        for model in models:
            is_stm = model == STM_MODEL
            indices = list(model_indices)
            if is_stm:
                if keep_rep_stopwords:
                    logger.warning(
                        "Skipping STM for dataset '%s': its bag-of-words only "
                        "exists with stopwords removed.",
                        dataset,
                    )
                    continue
                if split:
                    exp_target = f"{exp_dir}/{dataset}_standard_{model}"
                    n_runs = count_runs(exp_target)
                    indices = [i for i in indices if i <= n_runs]
                    if len(indices) < len(model_indices):
                        logger.info(
                            "%s defines %d runs; skipping run indices above it.",
                            exp_target,
                            n_runs,
                        )

            resources = {
                "mem": mem or (STM_MEM if is_stm else DEFAULT_MEM),
                "cpus": cpus or (STM_CPUS if is_stm else DEFAULT_CPUS),
                "time_limit": time_limit or (STM_TIME if is_stm else DEFAULT_TIME),
            }
            runs = indices if split else [None]
            for model_idx in runs:
                suffix = f"_m{model_idx}" if model_idx is not None else ""
                jobs.append(
                    SlurmJobConfig(
                        dataset=dataset,
                        model=model,
                        exp_dir=exp_dir,
                        job_dataset=job_dataset,
                        job_name=f"{job_dataset}_{model}{suffix}",
                        model_idx=model_idx,
                        keep_rep_stopwords=keep_rep_stopwords,
                        project_name=project_name,
                        reservation=reservation,
                        **resources,
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
    mem: str | None = None,
    cpus: int | None = None,
    time_limit: str | None = None,
    project_name: str = PROJECT_NAME,
    allow_empty_datasets: bool = False,
    raw_exact_models: str | None = None,
    reservation: str | None = None,
    pack_size: int = 1,
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
        mem: SLURM memory allocation; None uses each model's default.
        cpus: SLURM CPU allocation; None uses each model's default.
        time_limit: SLURM time allocation; None uses each model's default.
        project_name: Project name.
        allow_empty_datasets: For internal testing of dataset validation.
        raw_exact_models: User-specified exact model names.
        reservation: Optional SLURM reservation name.
        pack_size: Jobs per SLURM job, run in sequence; 1 disables packing.
            When packing, `time_limit` applies to each packed job, and None
            sums its members' default limits.

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
        time_limit=None if pack_size > 1 else time_limit,
        project_name=project_name,
        reservation=clean_reservation,
    )
    packs = pack_jobs(jobs, pack_size, time_limit) if pack_size != 1 else []

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
        packs=tuple(packs),
    )
