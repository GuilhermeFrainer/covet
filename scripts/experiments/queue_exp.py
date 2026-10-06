"""Master CLI script to select and queue experiments on SLURM.

Provides command-line argument parsing and orchestration, delegating domain
logic to src.experiment_queue and worker execution to scripts/experiments/slurm_job.sh
(STM: scripts/experiments/slurm_stm_job.sh).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.logger_config as logger_config
from src.experiment_queue import QueuePlan, create_queue_plan

LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "slurm_job.sh"
STM_WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "slurm_stm_job.sh"


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line parser for queue_exp."""
    parser = argparse.ArgumentParser(
        description="Master script to queue standard experiments on SLURM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  uv run python scripts/experiments/queue_exp.py -d fed
  uv run python scripts/experiments/queue_exp.py -d yelp -m mv_spectral -b
  uv run python scripts/experiments/queue_exp.py -d fed --split --runs 1..5
  uv run python scripts/experiments/queue_exp.py -d fed --stemmed
  uv run python scripts/experiments/queue_exp.py -d fed --test
  uv run python scripts/experiments/queue_exp.py -x pca,k_means
  uv run python scripts/experiments/queue_exp.py -d gadarian,anes \\
      -m baseline,stm,mv_spectral -b -n
""",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        "--datasets",
        dest="dataset",
        type=str,
        default=None,
        help=(
            "Comma-separated list of datasets to run. "
            "Available: anes, fed, gadarian, yelp. Default: all 4 datasets."
        ),
    )

    parser.add_argument(
        "-m",
        "--model",
        "--models",
        dest="model",
        type=str,
        default=None,
        help=(
            "Comma-separated list of models or model categories to run. "
            "Categories: baseline, stm, spectral, kmeans, spherical, "
            "pca, umap, tritopic. Exact models include mv_hdbscan and "
            "feature_stacking_hdbscan. Default: all standard models."
        ),
    )

    parser.add_argument(
        "-e",
        "--exact-model",
        "--exact-models",
        dest="exact_model",
        type=str,
        default=None,
        help=(
            "Comma-separated list of exact model names to run. "
            "Disables category expansion and substring matching."
        ),
    )

    parser.add_argument(
        "-x",
        "--exclude",
        dest="exclude",
        type=str,
        default=None,
        help=(
            "Comma-separated keywords or categories to exclude. "
            "Example: -x pca,k_means (excludes PCA & K-Means models)."
        ),
    )

    parser.add_argument(
        "-b",
        "--split",
        "--breakdown",
        dest="split",
        action="store_true",
        help=(
            "Split each experiment into separate Slurm jobs for each topic-count "
            "and seed combination (default: 15 runs per model)."
        ),
    )

    parser.add_argument(
        "--runs",
        "--model-idx",
        dest="runs",
        type=str,
        default=None,
        help=(
            "Comma-separated list or range of model configuration indices to run "
            "when split is active (default: 1-15). "
            "Examples: --runs 1..15, --runs 1,2,5."
        ),
    )

    parser.add_argument(
        "-s",
        "--stemmed",
        dest="stemmed",
        action="store_true",
        help="Run experiments on stemmed dataset versions (e.g. fed_stemmed).",
    )

    parser.add_argument(
        "--keep-rep-stopwords",
        dest="keep_rep_stopwords",
        action="store_true",
        help="Keep English stop words in topic representations (default: removed).",
    )

    parser.add_argument(
        "-t",
        "--test",
        dest="test",
        action="store_true",
        help=(
            "Run a minimal test set (baseline & stm). "
            "Defaults to 'fed' dataset if -d is not specified."
        ),
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Preview jobs to be submitted without executing sbatch.",
    )

    parser.add_argument(
        "-l",
        "--list",
        dest="list",
        action="store_true",
        help="List resolved dataset & model combinations and exit.",
    )

    parser.add_argument(
        "-y",
        "--yes",
        dest="yes",
        action="store_true",
        help="Skip confirmation prompt when submitting >10 jobs.",
    )

    parser.add_argument(
        "--mem",
        dest="mem",
        type=str,
        default=None,
        help="Override memory per job (default: 32G; STM: 16G).",
    )

    parser.add_argument(
        "--cpus",
        dest="cpus",
        type=int,
        default=None,
        help="Override CPUs per task (default: 4; STM: 1).",
    )

    parser.add_argument(
        "--time",
        dest="time",
        type=str,
        default=None,
        help="Override time limit per job (default: 24:00:00).",
    )

    parser.add_argument(
        "--worker-script",
        dest="worker_script",
        type=str,
        default=None,
        help=(
            "Path to SLURM worker script for non-STM models "
            "(default: scripts/experiments/slurm_job.sh)."
        ),
    )

    parser.add_argument(
        "--reservation",
        dest="reservation",
        type=str,
        default=None,
        help="SLURM reservation name to run jobs within.",
    )

    return parser


def format_plan_summary(plan: QueuePlan) -> str:
    """Format the experiment submission plan banner."""
    lines = [
        "=================================================================",
        " Experiment Submission Plan",
        "=================================================================",
        f" Datasets ({len(plan.datasets)}):  {' '.join(plan.datasets)}",
        f" Models ({len(plan.models)}):    {' '.join(plan.models)}",
    ]

    if plan.split:
        indices_str = " ".join(str(idx) for idx in plan.model_indices)
        lines.append(
            f" Mode:         SPLIT / BREAKDOWN "
            f"({len(plan.model_indices)} runs per model: {indices_str})"
        )
    else:
        lines.append(" Mode:         STANDARD (1 job per model)")

    lines.append(f" Total Jobs:   {plan.total_jobs}")

    if plan.use_stemmed:
        lines.append(" Variant:      STEMMED (using clean_text_stemmed)")

    if plan.excludes:
        lines.append(f" Exclusions:   {','.join(plan.excludes)}")

    if plan.reservation:
        lines.append(f" Reservation:  {plan.reservation}")

    if plan.dry_run:
        lines.append(" Dry Run:      YES (no jobs will be submitted)")

    lines.append("=================================================================")
    return "\n".join(lines)


def format_list_jobs(plan: QueuePlan) -> str:
    """Format the resolved jobs list for --list mode."""
    lines = ["", "Resolved Jobs List:"]
    for job in plan.jobs:
        if job.model_idx is not None:
            lines.append(
                f" - Dataset: {job.dataset} (Config dir: {job.exp_dir}) | "
                f"Model: {job.model} | Run: #{job.model_idx}"
            )
        else:
            lines.append(
                f" - Dataset: {job.dataset} (Config dir: {job.exp_dir}) | "
                f"Model: {job.model} (All runs)"
            )
    return "\n".join(lines)


def confirm_submission(total_jobs: int) -> bool:
    """Prompt user for confirmation when submitting more than 10 jobs."""
    try:
        reply = (
            input(
                f"You are about to submit {total_jobs} jobs to SLURM. Continue? [y/N] "
            )
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        return False
    return reply in ("y", "yes")


def submit_jobs(
    plan: QueuePlan,
    worker_script: Path | str,
    runner: Callable = subprocess.run,
    stm_worker_script: Path | str = STM_WORKER_SCRIPT,
) -> None:
    """Execute or simulate submission of all jobs in the plan.

    STM jobs go to `stm_worker_script`, every other job to `worker_script`.
    """
    Path("slurm_log").mkdir(parents=True, exist_ok=True)

    for job_count, job in enumerate(plan.jobs, 1):
        res_suffix = f" | Res: {job.reservation}" if job.reservation else ""
        if plan.dry_run:
            if job.model_idx is not None:
                print(
                    f"[{job_count}/{plan.total_jobs}] [DRY RUN] Job: {job.job_name} | "
                    f"Dataset: {job.dataset} | Model: {job.model} "
                    f"(Run #{job.model_idx}) | Mem: {job.mem} | CPUs: {job.cpus} | "
                    f"Time: {job.time_limit}{res_suffix}"
                )
            else:
                print(
                    f"[{job_count}/{plan.total_jobs}] [DRY RUN] Job: {job.job_name} | "
                    f"Dataset: {job.dataset} | Model: {job.model} | "
                    f"Mem: {job.mem} | CPUs: {job.cpus} | "
                    f"Time: {job.time_limit}{res_suffix}"
                )
        else:
            if job.model_idx is not None:
                print(
                    f"[{job_count}/{plan.total_jobs}] Queuing job for "
                    f"Dataset: {job.dataset} | Model: {job.model} | "
                    f"Run: #{job.model_idx}"
                )
            else:
                print(
                    f"[{job_count}/{plan.total_jobs}] Queuing job for "
                    f"Dataset: {job.dataset} | Model: {job.model}"
                )

            script = stm_worker_script if job.is_stm else worker_script
            sbatch_cmd = job.full_sbatch_command(str(script))
            runner(sbatch_cmd, check=True)

    print("------------------------------------------------")
    if plan.dry_run:
        print(f"Dry run complete ({plan.total_jobs} jobs simulated).")
    else:
        print(f"All {plan.total_jobs} jobs have been dispatched to the scheduler.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for experiment queue manager."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Initialize repository logger
    logger_config.setup_logging("queue_exp", LOG_DIR)

    split = args.split or (args.runs is not None)
    worker_script = (
        Path(args.worker_script) if args.worker_script else DEFAULT_WORKER_SCRIPT
    )

    try:
        plan = create_queue_plan(
            raw_datasets=args.dataset,
            raw_models=args.model,
            raw_exact_models=args.exact_model,
            raw_excludes=args.exclude,
            raw_runs=args.runs,
            split=split,
            use_stemmed=args.stemmed,
            keep_rep_stopwords=args.keep_rep_stopwords,
            is_test=args.test,
            dry_run=args.dry_run,
            mem=args.mem,
            cpus=args.cpus,
            time_limit=args.time,
            reservation=args.reservation,
        )
    except ValueError as e:
        print(f"{e}", file=sys.stderr)
        return 1

    print(format_plan_summary(plan))

    if args.list:
        print(format_list_jobs(plan))
        return 0

    if not plan.dry_run and not args.yes and plan.total_jobs > 10:
        if not confirm_submission(plan.total_jobs):
            print("Aborted.")
            return 0

    submit_jobs(plan, worker_script=worker_script)
    return 0


if __name__ == "__main__":
    sys.exit(main())
