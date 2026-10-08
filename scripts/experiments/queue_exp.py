"""Master CLI script to select and queue experiments on SLURM.

Provides command-line argument parsing and orchestration, delegating domain
logic to src.experiment_queue and worker execution to scripts/experiments/slurm_job.sh
(STM: scripts/experiments/slurm_stm_job.sh). Packed jobs run several of those
workers in sequence through scripts/experiments/slurm_pack_job.sh.
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.logger_config as logger_config
from src.experiment_queue import QueuePlan, SlurmJobConfig, create_queue_plan

LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "slurm_job.sh"
STM_WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "slurm_stm_job.sh"
PACK_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "slurm_pack_job.sh"

# The cluster's default QOS ("normal") lets a user hold 10 jobs, pending
# and running together, of which 5 run at once.
DEFAULT_MAX_QUEUED = 10
DEFAULT_POLL_SECONDS = 60
# Text sbatch prints when a QOS refuses a job over the submit limit.
SUBMIT_LIMIT_REASON = "MaxSubmitJob"


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
        help=(
            "Override time limit per job (default: 24:00:00). With --pack, "
            "this is the limit of each packed job (default: the sum of its "
            "runs' limits)."
        ),
    )

    parser.add_argument(
        "-p",
        "--pack",
        dest="pack",
        type=int,
        default=1,
        help=(
            "Run this many consecutive jobs of the plan one after another in a "
            "single SLURM job, which requests the largest memory and CPUs "
            "among them (default: 1, no packing)."
        ),
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
        "--max-queued",
        dest="max_queued",
        type=int,
        default=DEFAULT_MAX_QUEUED,
        help=(
            "Keep at most this many of your jobs in the queue, waiting for "
            "jobs to finish before submitting more. Match it to your QOS "
            f"MaxSubmitPU (default: {DEFAULT_MAX_QUEUED}; 0 disables waiting). "
            "Run large batches inside tmux so the wait survives logout."
        ),
    )

    parser.add_argument(
        "--poll-seconds",
        dest="poll_seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help=(
            "Seconds between queue checks while waiting for a free slot "
            f"(default: {DEFAULT_POLL_SECONDS})."
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

    if plan.packs:
        lines.append(
            f" Packing:      up to {len(plan.packs[0].jobs)} jobs in sequence per "
            f"SLURM job ({plan.total_submissions} SLURM jobs)"
        )

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

    def job_line(job: SlurmJobConfig, indent: str = " ") -> str:
        if job.model_idx is not None:
            return (
                f"{indent}- Dataset: {job.dataset} (Config dir: {job.exp_dir}) | "
                f"Model: {job.model} | Run: #{job.model_idx}"
            )
        return (
            f"{indent}- Dataset: {job.dataset} (Config dir: {job.exp_dir}) | "
            f"Model: {job.model} (All runs)"
        )

    if not plan.packs:
        lines.extend(job_line(job) for job in plan.jobs)
        return "\n".join(lines)

    for pack_count, pack in enumerate(plan.packs, 1):
        lines.append(
            f" SLURM job {pack_count}: {pack.job_name} | Mem: {pack.mem} | "
            f"CPUs: {pack.cpus} | Time: {pack.time_limit}"
        )
        lines.extend(job_line(job, indent="   ") for job in pack.jobs)
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


def count_queued_jobs(runner: Callable = subprocess.run) -> int:
    """Return how many of the current user's jobs SLURM holds, any state."""
    result = runner(
        ["squeue", "-h", "-u", getpass.getuser(), "-o", "%i"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sum(1 for line in result.stdout.splitlines() if line.strip())


def wait_for_slot(
    max_queued: int,
    runner: Callable = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> None:
    """Block until the user holds fewer than `max_queued` jobs."""
    announced = False
    while (queued := count_queued_jobs(runner)) >= max_queued:
        if not announced:
            print(
                f"  Queue full ({queued}/{max_queued} jobs); "
                f"checking again every {poll_seconds:g}s..."
            )
            announced = True
        sleeper(poll_seconds)


def submit_jobs(
    plan: QueuePlan,
    worker_script: Path | str,
    runner: Callable = subprocess.run,
    stm_worker_script: Path | str = STM_WORKER_SCRIPT,
    max_queued: int = 0,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Execute or simulate submission of all jobs in the plan.

    STM jobs go to `stm_worker_script`, every other job to `worker_script`.
    With `max_queued` above zero, each submission first waits until the user
    holds fewer than that many jobs. A job refused for exceeding the submit
    limit is retried after a wait; any other refusal is reported and the
    remaining jobs are still submitted.

    Returns:
        Names of the jobs sbatch refused.
    """
    Path("slurm_log").mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

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
            continue

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
        if not submit_until_accepted(
            sbatch_cmd, job.job_name, runner, max_queued, poll_seconds, sleeper
        ):
            failed.append(job.job_name)

    print_submission_summary(plan, failed)
    return failed


def submit_packs(
    plan: QueuePlan,
    worker_script: Path | str,
    runner: Callable = subprocess.run,
    stm_worker_script: Path | str = STM_WORKER_SCRIPT,
    pack_script: Path | str = PACK_SCRIPT,
    max_queued: int = 0,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Execute or simulate submission of the plan's packed jobs.

    Each pack is one SLURM job that runs its members' workers in sequence
    through `pack_script`. Waiting and retries work as in `submit_jobs`.

    Returns:
        Names of the packed jobs sbatch refused.
    """
    Path("slurm_log").mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    total = plan.total_submissions

    for pack_count, pack in enumerate(plan.packs, 1):
        n_runs = len(pack.jobs)
        if plan.dry_run:
            res_suffix = f" | Res: {pack.reservation}" if pack.reservation else ""
            print(
                f"[{pack_count}/{total}] [DRY RUN] Job: {pack.job_name} "
                f"({n_runs} in sequence) | Mem: {pack.mem} | CPUs: {pack.cpus} | "
                f"Time: {pack.time_limit}{res_suffix}"
            )
        else:
            print(
                f"[{pack_count}/{total}] Queuing job {pack.job_name} "
                f"({n_runs} in sequence)"
            )
        for job in pack.jobs:
            run = f" | Run: #{job.model_idx}" if job.model_idx is not None else ""
            print(f"    - Dataset: {job.dataset} | Model: {job.model}{run}")
        if plan.dry_run:
            continue

        sbatch_cmd = pack.full_sbatch_command(
            str(pack_script), str(worker_script), str(stm_worker_script)
        )
        if not submit_until_accepted(
            sbatch_cmd, pack.job_name, runner, max_queued, poll_seconds, sleeper
        ):
            failed.append(pack.job_name)

    print_submission_summary(plan, failed)
    return failed


def submit_until_accepted(
    sbatch_cmd: list[str],
    job_name: str,
    runner: Callable,
    max_queued: int,
    poll_seconds: float,
    sleeper: Callable[[float], None],
) -> bool:
    """Run one sbatch command, waiting while the queue is full.

    Returns:
        Whether sbatch accepted the job.
    """
    while True:
        if max_queued > 0:
            wait_for_slot(max_queued, runner, sleeper, poll_seconds)
        try:
            result = runner(sbatch_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            if SUBMIT_LIMIT_REASON in stderr:
                print(f"  Submit limit reached; retrying in {poll_seconds:g}s...")
                sleeper(poll_seconds)
                continue
            print(f"  sbatch refused {job_name}: {stderr}", file=sys.stderr)
            return False
        if isinstance(result.stdout, str) and result.stdout.strip():
            print(f"  {result.stdout.strip()}")
        return True


def print_submission_summary(plan: QueuePlan, failed: list[str]) -> None:
    """Print how many SLURM jobs were simulated, dispatched, or refused."""
    total = plan.total_submissions
    runs = f", {plan.total_jobs} runs" if plan.packs else ""
    print("------------------------------------------------")
    if plan.dry_run:
        print(f"Dry run complete ({total} jobs simulated{runs}).")
    elif failed:
        print(
            f"{total - len(failed)} of {total} jobs dispatched{runs}; "
            f"sbatch refused {len(failed)}: {' '.join(failed)}"
        )
    elif plan.packs:
        print(
            f"All {total} jobs ({plan.total_jobs} runs) have been dispatched "
            "to the scheduler."
        )
    else:
        print(f"All {total} jobs have been dispatched to the scheduler.")


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
            pack_size=args.pack,
        )
    except ValueError as e:
        print(f"{e}", file=sys.stderr)
        return 1

    print(format_plan_summary(plan))

    if args.list:
        print(format_list_jobs(plan))
        return 0

    if args.max_queued < 0 or args.poll_seconds <= 0:
        print("--max-queued must be >= 0 and --poll-seconds > 0.", file=sys.stderr)
        return 1

    if not plan.dry_run and not args.yes and plan.total_submissions > 10:
        if not confirm_submission(plan.total_submissions):
            print("Aborted.")
            return 0

    submit = submit_packs if plan.packs else submit_jobs
    failed = submit(
        plan,
        worker_script=worker_script,
        max_queued=args.max_queued,
        poll_seconds=args.poll_seconds,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
