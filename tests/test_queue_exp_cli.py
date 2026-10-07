"""Tests for scripts/experiments/queue_exp.py CLI and orchestration."""

import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.experiments.queue_exp import (
    build_parser,
    format_list_jobs,
    format_plan_summary,
    main,
    submit_jobs,
)
from src.experiment_queue import STM_IMAGE, create_queue_plan


@pytest.fixture(autouse=True)
def cleanup_pipeline_logger():
    yield
    logger = logging.getLogger("pipeline")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = True


def test_cli_imports_without_third_party_packages():
    # The login-node wrapper must not depend on the project's scientific stack.
    code = (
        "import sys\n"
        "for name in ('numpy', 'polars', 'yaml'):\n"
        "    sys.modules[name] = None\n"
        "import scripts.experiments.queue_exp\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class TestQueueExpCLIParser:
    """Test CLI argument parsing."""

    def test_default_args(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.dataset is None
        assert args.model is None
        assert args.exclude is None
        assert args.split is False
        assert args.runs is None
        assert args.stemmed is False
        assert args.keep_rep_stopwords is False
        assert args.test is False
        assert args.dry_run is False
        assert args.list is False
        assert args.yes is False
        assert args.mem is None
        assert args.cpus is None
        assert args.time is None
        assert args.reservation is None

    def test_reservation_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--reservation", "cluster_node_gpu"])
        assert args.reservation == "cluster_node_gpu"

    def test_aliases_dataset(self):
        parser = build_parser()
        assert parser.parse_args(["-d", "fed"]).dataset == "fed"
        assert parser.parse_args(["--dataset", "fed"]).dataset == "fed"
        assert parser.parse_args(["--datasets", "fed"]).dataset == "fed"

    def test_aliases_model(self):
        parser = build_parser()
        assert parser.parse_args(["-m", "baseline"]).model == "baseline"
        assert parser.parse_args(["--model", "baseline"]).model == "baseline"
        assert parser.parse_args(["--models", "baseline"]).model == "baseline"

    def test_aliases_exact_model(self):
        parser = build_parser()
        assert parser.parse_args(["-e", "tritopic"]).exact_model == "tritopic"
        res_long = parser.parse_args(["--exact-model", "tritopic"])
        assert res_long.exact_model == "tritopic"
        res_plural = parser.parse_args(["--exact-models", "tritopic"])
        assert res_plural.exact_model == "tritopic"

    def test_aliases_split_and_runs(self):
        parser = build_parser()
        assert parser.parse_args(["-b"]).split is True
        assert parser.parse_args(["--split"]).split is True
        assert parser.parse_args(["--breakdown"]).split is True
        assert parser.parse_args(["--runs", "1..5"]).runs == "1..5"
        assert parser.parse_args(["--model-idx", "1..5"]).runs == "1..5"

    def test_aliases_flags(self):
        parser = build_parser()
        assert parser.parse_args(["-s"]).stemmed is True
        assert parser.parse_args(["--stemmed"]).stemmed is True
        assert parser.parse_args(["--keep-rep-stopwords"]).keep_rep_stopwords is True
        assert parser.parse_args(["-t"]).test is True
        assert parser.parse_args(["--test"]).test is True
        assert parser.parse_args(["-n"]).dry_run is True
        assert parser.parse_args(["--dry-run"]).dry_run is True
        assert parser.parse_args(["-l"]).list is True
        assert parser.parse_args(["--list"]).list is True
        assert parser.parse_args(["-y"]).yes is True
        assert parser.parse_args(["--yes"]).yes is True

    def test_resource_overrides(self):
        parser = build_parser()
        args = parser.parse_args(["--mem", "64G", "--cpus", "8", "--time", "12:00:00"])
        assert args.mem == "64G"
        assert args.cpus == 8
        assert args.time == "12:00:00"


class TestFormatting:
    """Test output formatting for summary and list."""

    def test_format_plan_summary_standard(self):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            dry_run=True,
        )
        summary = format_plan_summary(plan)
        assert "Experiment Submission Plan" in summary
        assert "Datasets (1):  fed" in summary
        assert "Models (1):    baseline" in summary
        assert "Mode:         STANDARD (1 job per model)" in summary
        assert "Total Jobs:   1" in summary
        assert "Dry Run:      YES (no jobs will be submitted)" in summary

    def test_format_plan_summary_split(self):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs="1..3",
            split=True,
            use_stemmed=True,
            keep_rep_stopwords=True,
            dry_run=False,
        )
        summary = format_plan_summary(plan)
        assert "Mode:         SPLIT / BREAKDOWN (3 runs per model: 1 2 3)" in summary
        assert "Variant:      STEMMED (using clean_text_stemmed)" in summary

    def test_format_plan_summary_reservation(self):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            dry_run=True,
            reservation="gpu_cluster_res",
        )
        summary = format_plan_summary(plan)
        assert "Reservation:  gpu_cluster_res" in summary

    def test_format_list_jobs(self):
        plan_std = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
        )
        list_std = format_list_jobs(plan_std)
        assert "Resolved Jobs List:" in list_std
        assert (
            " - Dataset: fed (Config dir: fed) | Model: baseline (All runs)" in list_std
        )

        plan_split = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs="1",
            split=True,
            use_stemmed=False,
            keep_rep_stopwords=False,
        )
        list_split = format_list_jobs(plan_split)
        assert (
            " - Dataset: fed (Config dir: fed) | Model: baseline | Run: #1"
            in list_split
        )


class TestSubmissionAndMain:
    """Test job submission runner and main execution."""

    def test_submit_jobs_dry_run(self, capsys):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            dry_run=True,
        )
        mock_runner = MagicMock()
        submit_jobs(
            plan, worker_script="scripts/experiments/slurm_job.sh", runner=mock_runner
        )
        # In dry run, runner should not be called
        mock_runner.assert_not_called()
        captured = capsys.readouterr().out
        assert (
            "[1/1] [DRY RUN] Job: fed_baseline | Dataset: fed | Model: baseline"
            in captured
        )
        assert "Dry run complete (1 jobs simulated)." in captured

    def test_submit_jobs_live_mocked(self, capsys):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            dry_run=False,
        )
        mock_runner = MagicMock()
        submit_jobs(
            plan, worker_script="scripts/experiments/slurm_job.sh", runner=mock_runner
        )
        assert mock_runner.call_count == 1
        call_args = mock_runner.call_args[0][0]
        assert call_args[0] == "sbatch"
        assert "scripts/experiments/slurm_job.sh" in call_args
        captured = capsys.readouterr().out
        assert "[1/1] Queuing job for Dataset: fed | Model: baseline" in captured
        assert "All 1 jobs have been dispatched to the scheduler." in captured

    def test_main_list_mode(self, capsys):
        exit_code = main(["-d", "fed", "-m", "baseline", "-l"])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Resolved Jobs List:" in captured
        assert (
            "- Dataset: fed (Config dir: fed) | Model: baseline (All runs)" in captured
        )

    def test_main_dry_run(self, capsys):
        exit_code = main(["-d", "fed", "-m", "baseline", "-n"])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Experiment Submission Plan" in captured
        assert "Dry run complete" in captured

    def test_main_exact_model_dry_run(self, capsys):
        exit_code = main(["-d", "fed", "-e", "tritopic", "-n"])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Models (1):    tritopic" in captured
        assert "Job: fed_tritopic" in captured
        assert "fast_tritopic" not in captured
        assert "Dry run complete (1 jobs simulated)." in captured

    def test_submit_jobs_routes_stm_to_its_worker(self, capsys):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline,stm",
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            dry_run=False,
        )
        mock_runner = MagicMock()
        submit_jobs(
            plan,
            worker_script="slurm_job.sh",
            runner=mock_runner,
            stm_worker_script="slurm_stm_job.sh",
        )
        baseline_cmd, stm_cmd = (call[0][0] for call in mock_runner.call_args_list)
        assert "slurm_job.sh" in baseline_cmd
        assert "slurm_stm_job.sh" not in baseline_cmd
        assert stm_cmd[-4:] == [
            "slurm_stm_job.sh",
            "fed",
            "fed/fed_standard_stm",
            STM_IMAGE,
        ]
        assert "--cpus-per-task=1" in stm_cmd

    def test_submit_jobs_dry_run_with_reservation(self, capsys):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            dry_run=True,
            reservation="my_special_res",
        )
        mock_runner = MagicMock()
        submit_jobs(
            plan, worker_script="scripts/experiments/slurm_job.sh", runner=mock_runner
        )
        mock_runner.assert_not_called()
        captured = capsys.readouterr().out
        assert "Job: fed_baseline" in captured
        assert "| Res: my_special_res" in captured

    def test_submit_jobs_live_with_reservation(self, capsys):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            dry_run=False,
            reservation="my_special_res",
        )
        mock_runner = MagicMock()
        submit_jobs(
            plan, worker_script="scripts/experiments/slurm_job.sh", runner=mock_runner
        )
        assert mock_runner.call_count == 1
        call_args = mock_runner.call_args[0][0]
        assert "--reservation=my_special_res" in call_args

    def test_main_with_reservation(self, capsys):
        exit_code = main(
            ["-d", "fed", "-m", "baseline", "--reservation", "my_res", "-n"]
        )
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Reservation:  my_res" in captured
        assert "| Res: my_res" in captured

    def test_confirm_submission(self, monkeypatch):
        from scripts.experiments.queue_exp import confirm_submission

        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert confirm_submission(20) is True

        monkeypatch.setattr("builtins.input", lambda _: "n")
        assert confirm_submission(20) is False

        monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(EOFError))
        assert confirm_submission(20) is False

    def test_main_aborted_when_prompt_rejected(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        # 4 datasets x baseline = 4 jobs, but let's test with split to exceed 10 jobs
        exit_code = main(["-d", "fed", "-m", "baseline", "--runs", "1..15"])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Aborted." in captured
