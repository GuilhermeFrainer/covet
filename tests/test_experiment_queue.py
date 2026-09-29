"""Tests for experiment queue pure logic in src.experiment_queue."""

import pytest

from src.experiment_queue import (
    ALL_MODELS,
    DEFAULT_DATASETS,
    DEFAULT_MODEL_INDICES,
    apply_exclusions,
    build_jobs,
    create_queue_plan,
    parse_run_indices,
    resolve_datasets,
    resolve_models,
)


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
        # stm is skipped because STM jobs are disabled
        assert len(jobs) == 1
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
        # STM jobs are disabled, so every model except stm yields one job
        n_jobs = len(DEFAULT_DATASETS) * (len(ALL_MODELS) - 1)
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
        assert plan_split.total_jobs == n_jobs * len(DEFAULT_MODEL_INDICES)


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
        # stm skipped in jobs, only baseline remains
        assert len(plan.jobs) == 1
        assert plan.total_jobs == 1
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
