"""Tests for experiment queue pure logic in src.experiment_queue."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.experiments.queue_exp import PACK_SCRIPT
from scripts.experiments.run_stm import resolve_stm_inputs
from src.experiment_queue import (
    ALL_MODELS,
    DEFAULT_CPUS,
    DEFAULT_DATASETS,
    DEFAULT_MEM,
    DEFAULT_MODEL_INDICES,
    PACK_SEPARATOR,
    STM_CPUS,
    STM_IMAGE,
    STM_MEM,
    apply_exclusions,
    build_jobs,
    count_config_runs,
    create_queue_plan,
    format_time_limit,
    parse_mem,
    parse_run_indices,
    parse_time_limit,
    resolve_datasets,
    resolve_models,
)
from src.utils import load_config

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


class TestParseRunIndices:
    """Test parsing of run/model index specifications."""

    def test_default_when_none_or_empty(self):
        assert parse_run_indices(None) == list(DEFAULT_MODEL_INDICES)
        assert parse_run_indices("") == list(DEFAULT_MODEL_INDICES)
        assert parse_run_indices("   ") == list(DEFAULT_MODEL_INDICES)

    def test_range_with_dots(self):
        assert parse_run_indices("1..5") == [1, 2, 3, 4, 5]
        assert parse_run_indices("1..1") == [1]

    def test_range_with_dash(self):
        assert parse_run_indices("1-5") == [1, 2, 3, 4, 5]
        assert parse_run_indices("10-12") == [10, 11, 12]

    def test_comma_separated_indices(self):
        assert parse_run_indices("1,2,5") == [1, 2, 5]
        assert parse_run_indices(" 1 , 2 , 5 ") == [1, 2, 5]

    def test_mixed_ranges_and_singles(self):
        assert parse_run_indices("1..3,7,9-10") == [1, 2, 3, 7, 9, 10]

    def test_reversed_range_produces_empty(self):
        # Bash `for ((idx=5; idx<=1; idx++))` produces no elements
        assert parse_run_indices("5..1") == []
        assert parse_run_indices("5-1") == []

    def test_invalid_tokens_ignored_with_warning(self, caplog):
        # Invalid tokens should be ignored and trigger a warning
        res = parse_run_indices("1,foo,3,1..a,-5")
        assert res == [1, 3]
        assert "Unrecognized run index or range" in caplog.text


class TestResolveDatasets:
    """Test resolution of dataset specifications."""

    def test_default_datasets(self):
        assert resolve_datasets(None) == list(DEFAULT_DATASETS)
        assert resolve_datasets("") == list(DEFAULT_DATASETS)

    def test_explicit_datasets(self):
        assert resolve_datasets("fed") == ["fed"]
        assert resolve_datasets("fed,anes") == ["fed", "anes"]
        assert resolve_datasets("  gadarian ,  yelp  ") == ["gadarian", "yelp"]

    def test_test_mode_default_dataset(self):
        assert resolve_datasets(None, is_test=True) == ["fed"]

    def test_test_mode_with_explicit_dataset(self):
        assert resolve_datasets("anes", is_test=True) == ["anes"]


class TestResolveModels:
    """Test model resolution and category expansion."""

    def test_default_models(self):
        assert resolve_models(None) == list(ALL_MODELS)
        assert resolve_models("") == list(ALL_MODELS)

    def test_test_mode_models(self):
        assert resolve_models(None, is_test=True) == ["baseline", "stm"]

    def test_category_kmeans_and_alias(self):
        models_kmeans = resolve_models("kmeans")
        models_k_means = resolve_models("k_means")
        assert models_kmeans == models_k_means
        # Matches any model with "k_means" in name (including spherical)
        assert all("k_means" in m for m in models_kmeans)
        assert "k_means" in models_kmeans
        assert "aligned_umap_mv_k_means" in models_kmeans
        assert "mv_spherical_k_means" in models_kmeans

    def test_category_spherical(self):
        models = resolve_models("spherical")
        assert all("spherical" in m for m in models)
        assert "mv_spherical_k_means" in models
        assert "pca_mv_spherical_k_means" in models

    def test_category_pca(self):
        models = resolve_models("pca")
        assert all("pca" in m for m in models)
        assert "pca_k_means" in models
        assert "pca_mv_spectral" in models

    def test_category_spectral(self):
        models = resolve_models("spectral")
        assert all("spectral" in m for m in models)
        assert "mv_spectral" in models
        assert "umap_spectral" in models

    def test_category_umap(self):
        models = resolve_models("umap")
        assert all("umap" in m for m in models)
        assert "aligned_umap" in models
        assert "append_umap" in models
        assert "umap_spectral" in models

    def test_category_tritopic(self):
        models = resolve_models("tritopic")
        assert models == ["fast_tritopic", "tritopic"]

    def test_explicit_model_and_substring_fallback(self):
        assert resolve_models("baseline") == ["baseline"]
        assert resolve_models("stm") == ["stm"]
        # Substring matching for non-category items
        mv_spectral_models = resolve_models("mv_spectral")
        assert "mv_spectral" in mv_spectral_models
        assert "append_umap_mv_spectral" in mv_spectral_models
        assert "pca_mv_spectral" in mv_spectral_models

    def test_unknown_model_warning(self, caplog):
        res = resolve_models("unknown_model_xyz")
        assert res == []
        expected_msg = (
            "Warning: Model or category 'unknown_model_xyz' did not match any"
            " known model."
        )
        assert expected_msg in caplog.text

    def test_deduplication_preserving_order(self):
        res = resolve_models("baseline,baseline,stm,baseline")
        assert res == ["baseline", "stm"]

    def test_exact_model_tritopic(self):
        res = resolve_models(raw_exact_models="tritopic")
        assert res == ["tritopic"]
        assert "fast_tritopic" not in res

    def test_exact_model_mv_spectral(self):
        res = resolve_models(raw_exact_models="mv_spectral")
        assert res == ["mv_spectral"]
        assert "append_umap_mv_spectral" not in res
        assert "pca_mv_spectral" not in res

    def test_exact_model_multiple(self):
        res = resolve_models(raw_exact_models="tritopic, baseline")
        assert res == ["tritopic", "baseline"]

    def test_exact_model_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown exact model 'invalid_model'"):
            resolve_models(raw_exact_models="invalid_model")

    def test_exact_model_combined_with_category(self):
        res = resolve_models(raw_models="baseline", raw_exact_models="tritopic")
        assert res == ["baseline", "tritopic"]


class TestApplyExclusions:
    """Test applying exclusions to models."""

    def test_exclude_category_and_alias(self):
        models = ["k_means", "pca_k_means", "mv_spectral", "baseline"]
        assert apply_exclusions(models, "kmeans") == ["mv_spectral", "baseline"]
        assert apply_exclusions(models, "k_means") == ["mv_spectral", "baseline"]

    def test_exclude_multiple(self):
        models = ["pca_k_means", "pca_mv_spectral", "mv_spectral", "baseline"]
        assert apply_exclusions(models, "pca,baseline") == ["mv_spectral"]

    def test_no_exclusions(self):
        models = ["baseline", "stm"]
        assert apply_exclusions(models, None) == ["baseline", "stm"]
        assert apply_exclusions(models, "") == ["baseline", "stm"]

    def test_exclude_all_raises_error(self):
        models = ["baseline"]
        with pytest.raises(ValueError, match="No models selected"):
            apply_exclusions(models, "baseline")


class TestJobConstruction:
    """Test job generation and SlurmJobConfig properties."""

    def test_standard_job_construction(self):
        jobs = build_jobs(
            datasets=["fed"],
            models=["baseline", "stm"],
            split=False,
            model_indices=[1],
            use_stemmed=False,
            keep_rep_stopwords=False,
        )
        assert [job.model for job in jobs] == ["baseline", "stm"]
        job = jobs[0]
        assert job.dataset == "fed"
        assert job.model == "baseline"
        assert job.job_name == "fed_baseline"
        assert job.exp_dir == "fed"
        assert job.model_idx is None
        assert "--remove-rep-stopwords" in job.run_command
        assert (
            "scripts/experiments/run_optimizer.py --exp fed/fed_standard_baseline"
            in job.run_command
        )
        assert "--model" not in job.run_command

    def test_split_job_construction(self):
        jobs = build_jobs(
            datasets=["fed", "anes"],
            models=["baseline"],
            split=True,
            model_indices=[1, 2],
            use_stemmed=False,
            keep_rep_stopwords=False,
        )
        assert len(jobs) == 4
        assert jobs[0].job_name == "fed_baseline_m1"
        assert jobs[0].model_idx == 1
        assert "--model 1" in jobs[0].run_command
        assert jobs[1].job_name == "fed_baseline_m2"
        assert jobs[1].model_idx == 2
        assert "--model 2" in jobs[1].run_command

    def test_stemmed_dataset(self):
        jobs = build_jobs(
            datasets=["fed"],
            models=["baseline"],
            split=False,
            model_indices=[1],
            use_stemmed=True,
            keep_rep_stopwords=False,
        )
        assert len(jobs) == 1
        assert jobs[0].exp_dir == "fed_stemmed"
        assert jobs[0].job_dataset == "fed_stemmed"
        assert jobs[0].job_name == "fed_stemmed_baseline"
        assert "--exp fed_stemmed/fed_standard_baseline" in jobs[0].run_command

    def test_keep_rep_stopwords(self):
        jobs = build_jobs(
            datasets=["fed"],
            models=["baseline"],
            split=False,
            model_indices=[1],
            use_stemmed=False,
            keep_rep_stopwords=True,
        )
        assert "--keep-rep-stopwords" in jobs[0].run_command
        assert "--remove-rep-stopwords" not in jobs[0].run_command

    def test_yelp_data_copy_command(self):
        yelp_jobs = build_jobs(
            datasets=["yelp"],
            models=["baseline"],
            split=False,
            model_indices=[1],
            use_stemmed=False,
            keep_rep_stopwords=False,
        )
        fed_jobs = build_jobs(
            datasets=["fed"],
            models=["baseline"],
            split=False,
            model_indices=[1],
            use_stemmed=False,
            keep_rep_stopwords=False,
        )
        assert (
            "yelp_s10000_embeddings.parquet data/processed/yelp_embeddings.parquet"
            in yelp_jobs[0].data_copy_command
        )
        assert "fed_embeddings.parquet data/processed/" in fed_jobs[0].data_copy_command

    @pytest.mark.parametrize("use_stemmed", [False, True])
    def test_trump_sample_keeps_its_own_data_and_configs(self, use_stemmed):
        job = build_jobs(
            datasets=["trump_s25000"],
            models=["baseline"],
            split=False,
            model_indices=[1],
            use_stemmed=use_stemmed,
            keep_rep_stopwords=False,
        )[0]
        exp_dir = "trump_s25000_stemmed" if use_stemmed else "trump_s25000"
        exp_target = f"{exp_dir}/trump_s25000_standard_baseline"

        assert f"--exp {exp_target} " in job.run_command
        assert (EXPERIMENTS_DIR / f"{exp_target}.yaml").exists()
        assert job.data_copy_command.endswith(
            "data/processed/trump_s25000_embeddings.parquet data/processed/"
        )

    def test_resource_overrides(self):
        jobs = build_jobs(
            datasets=["fed"],
            models=["baseline"],
            split=False,
            model_indices=[1],
            use_stemmed=False,
            keep_rep_stopwords=False,
            mem="64G",
            cpus=8,
            time_limit="12:00:00",
        )
        job = jobs[0]
        assert job.mem == "64G"
        assert job.cpus == 8
        assert job.time_limit == "12:00:00"
        sbatch_args = job.sbatch_args
        assert "--mem=64G" in sbatch_args or (
            "--mem" in sbatch_args
            and sbatch_args[sbatch_args.index("--mem") + 1] == "64G"
        )
        assert "--cpus-per-task=8" in sbatch_args or (
            "--cpus-per-task" in sbatch_args
            and sbatch_args[sbatch_args.index("--cpus-per-task") + 1] == "8"
        )
        assert "--time=12:00:00" in sbatch_args or (
            "--time" in sbatch_args
            and sbatch_args[sbatch_args.index("--time") + 1] == "12:00:00"
        )

    def test_worker_args_and_sbatch_command(self):
        jobs_split = build_jobs(
            datasets=["fed"],
            models=["baseline"],
            split=True,
            model_indices=[3],
            use_stemmed=False,
            keep_rep_stopwords=False,
            mem="16G",
            cpus=2,
            time_limit="10:00:00",
        )
        job = jobs_split[0]
        assert job.worker_args == [
            "fed",
            "fed/fed_standard_baseline",
            "--remove-rep-stopwords",
            "3",
        ]
        cmd = job.full_sbatch_command("scripts/experiments/slurm_job.sh")
        assert cmd[0] == "sbatch"
        assert "--job-name=fed_baseline_m3" in cmd
        assert "--mem=16G" in cmd
        assert "--cpus-per-task=2" in cmd
        assert "--time=10:00:00" in cmd
        assert cmd[-5:] == [
            "scripts/experiments/slurm_job.sh",
            "fed",
            "fed/fed_standard_baseline",
            "--remove-rep-stopwords",
            "3",
        ]

    def test_standard_vs_split_job_counts(self):
        n_jobs = len(DEFAULT_DATASETS) * len(ALL_MODELS)
        plan_standard = create_queue_plan(
            raw_datasets=None,
            raw_models=None,
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
        )
        assert plan_standard.total_jobs == n_jobs

        plan_split = create_queue_plan(
            raw_datasets=None,
            raw_models=None,
            raw_excludes=None,
            raw_runs="1..15",
            split=True,
            use_stemmed=False,
            keep_rep_stopwords=False,
        )
        stm_runs = sum(
            count_config_runs(f"{dataset}/{dataset}_standard_stm")
            for dataset in DEFAULT_DATASETS
        )
        neural_jobs = len(DEFAULT_DATASETS) * (len(ALL_MODELS) - 1)
        assert plan_split.total_jobs == (
            neural_jobs * len(DEFAULT_MODEL_INDICES) + stm_runs
        )


class TestCreateQueuePlan:
    """Test end-to-end plan creation."""

    def test_create_queue_plan_test_mode(self):
        plan = create_queue_plan(
            raw_datasets=None,
            raw_models=None,
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            is_test=True,
            dry_run=True,
        )
        assert plan.datasets == ("fed",)
        assert plan.models == ("baseline", "stm")
        assert [job.model for job in plan.jobs] == ["baseline", "stm"]
        assert plan.total_jobs == 2
        assert plan.dry_run is True

    def test_reservation_job_construction(self):
        jobs = build_jobs(
            datasets=["fed"],
            models=["baseline"],
            split=False,
            model_indices=[1],
            use_stemmed=False,
            keep_rep_stopwords=False,
            reservation="my_reservation",
        )
        assert len(jobs) == 1
        job = jobs[0]
        assert job.reservation == "my_reservation"
        assert "--reservation=my_reservation" in job.sbatch_args
        assert "--reservation=my_reservation" in job.full_sbatch_command()

    def test_reservation_none_by_default(self):
        jobs = build_jobs(
            datasets=["fed"],
            models=["baseline"],
            split=False,
            model_indices=[1],
            use_stemmed=False,
            keep_rep_stopwords=False,
        )
        assert jobs[0].reservation is None
        assert not any("--reservation" in arg for arg in jobs[0].sbatch_args)

    def test_create_queue_plan_with_reservation(self):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            reservation="  cluster_node_1  ",
        )
        assert plan.reservation == "cluster_node_1"
        assert plan.jobs[0].reservation == "cluster_node_1"
        assert "--reservation=cluster_node_1" in plan.jobs[0].sbatch_args

    def test_create_queue_plan_empty_reservation_is_none(self):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            reservation="   ",
        )
        assert plan.reservation is None
        assert plan.jobs[0].reservation is None


class TestStmJobs:
    """STM jobs run run_stm.py through their own worker and image."""

    def _stm_job(self, dataset="fed", **kwargs):
        options = {
            "split": False,
            "model_indices": [1],
            "use_stemmed": False,
            "keep_rep_stopwords": False,
        }
        options.update(kwargs)
        return build_jobs(datasets=[dataset], models=["stm"], **options)

    def test_standard_stm_job(self):
        (job,) = self._stm_job()
        assert job.is_stm
        assert job.job_name == "fed_stm"
        assert job.run_command == (
            "uv run python scripts/experiments/run_stm.py "
            "--exp fed/fed_standard_stm --r-runner docker"
        )
        assert job.worker_args == ["fed", "fed/fed_standard_stm", STM_IMAGE]
        assert job.mem == STM_MEM
        assert job.cpus == STM_CPUS

    def test_neural_jobs_keep_default_resources(self):
        jobs = build_jobs(
            datasets=["fed"],
            models=["baseline", "stm"],
            split=False,
            model_indices=[1],
            use_stemmed=False,
            keep_rep_stopwords=False,
        )
        assert (jobs[0].mem, jobs[0].cpus) == (DEFAULT_MEM, DEFAULT_CPUS)
        assert (jobs[1].mem, jobs[1].cpus) == (STM_MEM, STM_CPUS)

    def test_resource_overrides_apply_to_stm(self):
        (job,) = self._stm_job(mem="64G", cpus=2, time_limit="02:00:00")
        assert (job.mem, job.cpus, job.time_limit) == ("64G", 2, "02:00:00")

    def test_split_stops_at_the_config_run_count(self):
        jobs = self._stm_job(
            split=True, model_indices=list(range(1, 16)), count_runs=lambda _: 5
        )
        assert [job.model_idx for job in jobs] == [1, 2, 3, 4, 5]
        assert jobs[2].worker_args == ["fed", "fed/fed_standard_stm", STM_IMAGE, "3"]
        assert "--model 3 --r-runner docker" in jobs[2].run_command

    def test_skipped_when_rep_stopwords_are_kept(self):
        assert self._stm_job(keep_rep_stopwords=True) == []

    def test_stemmed_and_yelp_inputs(self):
        (stemmed,) = self._stm_job(use_stemmed=True)
        assert stemmed.worker_args[:2] == [
            "fed_stemmed",
            "fed_stemmed/fed_standard_stm",
        ]
        (yelp,) = self._stm_job(dataset="yelp")
        assert yelp.stm_input_prefix == "yelp_s10000"
        assert "yelp_s10000_stm_data.rds" in yelp.data_copy_command
        assert "yelp_s10000_bow.parquet" in yelp.data_copy_command

    @pytest.mark.parametrize("use_stemmed", [False, True])
    @pytest.mark.parametrize("dataset", [*DEFAULT_DATASETS, "trump", "trump_s25000"])
    def test_worker_inputs_match_run_stm(self, dataset, use_stemmed):
        (job,) = self._stm_job(dataset=dataset, use_stemmed=use_stemmed)
        config = load_config(job.exp_target, EXPERIMENTS_DIR)
        _, rds_path, bow_path = resolve_stm_inputs(config["experiment"])
        assert rds_path.name == f"{job.stm_input_prefix}_stm_data.rds"
        assert bow_path.name == f"{job.stm_input_prefix}_bow.parquet"

    def test_active_stm_configs_use_one_seed(self):
        for path in EXPERIMENTS_DIR.glob("*/*_standard_stm.yaml"):
            if "archive" in path.parts:
                continue
            exp_target = f"{path.parent.name}/{path.stem}"
            config = load_config(exp_target, EXPERIMENTS_DIR)
            assert len(config["experiment"]["random_state"]) == 1, exp_target
            assert count_config_runs(exp_target) == len(config["models"])


class TestSlurmLimits:
    """Parsing and formatting of SLURM time limits and memory requests."""

    @pytest.mark.parametrize(
        ("value", "seconds"),
        [
            ("30", 30 * 60),
            ("30:15", 30 * 60 + 15),
            ("24:00:00", 86400),
            ("1-12", 86400 + 12 * 3600),
            ("2-01:30", 2 * 86400 + 3600 + 30 * 60),
            ("1-00:00:05", 86405),
        ],
    )
    def test_parse_time_limit(self, value, seconds):
        assert parse_time_limit(value) == seconds

    @pytest.mark.parametrize("value", ["", "abc", "1:2:3:4", "1-2:3:4:5", "x-01"])
    def test_parse_time_limit_rejects_invalid(self, value):
        with pytest.raises(ValueError, match="Invalid SLURM time limit"):
            parse_time_limit(value)

    def test_format_time_limit(self):
        assert format_time_limit(86400 * 4) == "4-00:00:00"
        assert format_time_limit(3600 + 61) == "01:01:01"

    def test_parse_mem(self):
        assert parse_mem("32G") == 32768
        assert parse_mem("16000") == 16000
        assert parse_mem("1t") == 1024 * 1024
        with pytest.raises(ValueError, match="Invalid SLURM memory"):
            parse_mem("lots")


class TestPacking:
    """Packing several jobs into SLURM jobs that run them in sequence."""

    def _plan(self, pack_size, time_limit=None, models="baseline,stm,mv_k_means"):
        return create_queue_plan(
            raw_datasets="fed",
            raw_models=None,
            raw_exact_models=models,
            raw_excludes=None,
            raw_runs=None,
            split=False,
            use_stemmed=False,
            keep_rep_stopwords=False,
            time_limit=time_limit,
            pack_size=pack_size,
        )

    def test_no_packing_by_default(self):
        plan = self._plan(1)
        assert plan.packs == ()
        assert plan.total_submissions == plan.total_jobs == 3

    def test_packs_consecutive_jobs(self):
        plan = self._plan(2)
        assert plan.total_jobs == 3
        assert plan.total_submissions == 2
        assert [len(p.jobs) for p in plan.packs] == [2, 1]
        assert plan.packs[0].jobs == plan.jobs[:2]
        assert plan.packs[0].job_name == "fed_baseline+1"
        assert plan.packs[1].job_name == "fed_mv_k_means"

    def test_pack_requests_largest_resources(self):
        pack = self._plan(2).packs[0]  # baseline (32G, 4 CPUs) + STM (16G, 1)
        assert pack.mem == DEFAULT_MEM
        assert pack.cpus == DEFAULT_CPUS

    def test_pack_time_sums_members_by_default(self):
        plan = self._plan(3)
        assert plan.packs[0].time_limit == "3-00:00:00"

    def test_time_override_applies_to_the_pack(self):
        plan = self._plan(3, time_limit="2-00:00:00")
        assert plan.packs[0].time_limit == "2-00:00:00"
        assert "--time=2-00:00:00" in plan.packs[0].sbatch_args
        # Members keep their defaults; only the packed job is submitted.
        assert all(job.time_limit == "24:00:00" for job in plan.jobs)

    def test_invalid_pack_arguments(self):
        with pytest.raises(ValueError, match="--pack must be at least 1"):
            self._plan(0)
        with pytest.raises(ValueError, match="Invalid SLURM time limit"):
            self._plan(2, time_limit="soon")

    def test_sbatch_command_separates_workers(self):
        pack = self._plan(3).packs[0]
        cmd = pack.full_sbatch_command("pack.sh", "worker.sh", "stm.sh")
        script_at = cmd.index("pack.sh")
        assert cmd[0] == "sbatch"
        assert "--job-name=fed_baseline+2" in cmd[:script_at]
        runs = " ".join(cmd[script_at + 1 :]).split(f" {PACK_SEPARATOR} ")
        assert runs == [
            "worker.sh fed fed/fed_standard_baseline --remove-rep-stopwords",
            f"stm.sh fed fed/fed_standard_stm {STM_IMAGE}",
            "worker.sh fed fed/fed_standard_mv_k_means --remove-rep-stopwords",
        ]

    def test_split_runs_keep_their_indices(self):
        plan = create_queue_plan(
            raw_datasets="fed",
            raw_models="baseline",
            raw_excludes=None,
            raw_runs="1..5",
            split=True,
            use_stemmed=False,
            keep_rep_stopwords=False,
            pack_size=4,
        )
        assert [p.job_name for p in plan.packs] == [
            "fed_baseline_m1+3",
            "fed_baseline_m5",
        ]
        cmd = plan.packs[0].full_sbatch_command("p.sh", "w.sh", "s.sh")
        assert cmd.count(PACK_SEPARATOR) == 3
        assert cmd[-1] == "4"


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="runs the SLURM pack script with POSIX bash",
)
class TestPackScript:
    """slurm_pack_job.sh runs every worker in order and reports failures."""

    def _worker(self, tmp_path, name, exit_code):
        script = tmp_path / name
        script.write_text(
            "#!/bin/bash\n"
            f'echo "{name} $*" >> "{tmp_path}/calls.log"\n'
            f"exit {exit_code}\n"
        )
        return str(script)

    def test_runs_all_workers_and_fails_if_any_failed(self, tmp_path):
        ok = self._worker(tmp_path, "ok.sh", 0)
        bad = self._worker(tmp_path, "bad.sh", 3)
        result = subprocess.run(
            ["bash", str(PACK_SCRIPT), ok, "a", "b c", "::", bad, "x", "::", ok, "z"],
            capture_output=True,
            text=True,
        )
        calls = (tmp_path / "calls.log").read_text().splitlines()
        assert calls == ["ok.sh a b c", "bad.sh x", "ok.sh z"]
        assert result.returncode == 1
        assert "2/3 runs succeeded" in result.stdout
        assert "Failed run 2" in result.stderr

    def test_succeeds_when_all_workers_succeed(self, tmp_path):
        ok = self._worker(tmp_path, "ok.sh", 0)
        result = subprocess.run(
            ["bash", str(PACK_SCRIPT), ok, "1", "::", ok, "2"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "2/2 runs succeeded" in result.stdout
