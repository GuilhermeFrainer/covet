"""Unit tests for completeness verification and partial model detection."""

import pathlib
import sys

import polars as pl
import pytest

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.merge_results import merge_files  # noqa: E402
from src.verification import (  # noqa: E402
    CompletenessReport,
    verify_dataset_completeness,
)


@pytest.fixture
def mock_experiments_dir(tmp_path):
    exp_dir = tmp_path / "experiments"
    dataset_dir = exp_dir / "testds"
    dataset_dir.mkdir(parents=True)

    # 1. Model A config: 2 topic combos x 2 seeds = 4 runs
    yaml_a = dataset_dir / "testds_standard_model_a.yaml"
    yaml_a.write_text(
        """
experiment:
  name: "testds_standard_model_a"
  random_state:
    - 101
    - 102

model:
  id: "model_a"
  bertopic:
    params:
      nr_topics:
        - 10
        - 20
"""
    )

    # 2. Model B config: 1 combo x 2 seeds = 2 runs
    yaml_b = dataset_dir / "testds_standard_model_b.yaml"
    yaml_b.write_text(
        """
experiment:
  name: "testds_standard_model_b"
  random_state:
    - 101
    - 102

model:
  id: "model_b"
"""
    )

    return exp_dir


def test_verify_completeness_complete_and_unrun(mock_experiments_dir):
    # Model A is complete (4 runs with seeds 101, 102), Model B is unrun (0 runs)
    df = pl.DataFrame(
        {
            "dataset_name": ["testds"] * 4,
            "model_name": [
                "model_a_1_seed101",
                "model_a_2_seed101",
                "model_a_1_seed102",
                "model_a_2_seed102",
            ],
            "random_state": [101, 101, 102, 102],
            "u_mass": [-0.5, -0.6, -0.55, -0.65],
            "c_v": [0.6, 0.62, 0.58, 0.61],
        }
    )

    report: CompletenessReport = verify_dataset_completeness(
        dataset_name="testds",
        dataset_type="standard",
        df=df,
        experiments_dir=mock_experiments_dir,
    )

    assert len(report.complete_models) == 1
    assert report.complete_models[0].model_name == "model_a"
    assert len(report.unrun_models) == 1
    assert report.unrun_models[0].model_name == "model_b"
    assert len(report.partial_models) == 0
    assert not report.has_partial_models
    assert report.is_valid_for_merge


def test_verify_completeness_partial_model(mock_experiments_dir):
    # Model A is partial (only seed 101 completed, seed 102 missing)
    df = pl.DataFrame(
        {
            "dataset_name": ["testds"] * 2,
            "model_name": ["model_a_1_seed101", "model_a_2_seed101"],
            "random_state": [101, 101],
            "u_mass": [-0.5, -0.6],
            "c_v": [0.6, 0.62],
        }
    )

    report: CompletenessReport = verify_dataset_completeness(
        dataset_name="testds",
        dataset_type="standard",
        df=df,
        experiments_dir=mock_experiments_dir,
    )

    assert len(report.partial_models) == 1
    partial = report.partial_models[0]
    assert partial.model_name == "model_a"
    assert partial.found_runs == 2
    assert partial.expected_runs == 4
    assert partial.missing_seeds == [102]
    assert report.has_partial_models
    assert not report.is_valid_for_merge

    slurm_cmd = report.slurm_rerun_command()
    assert slurm_cmd is not None
    assert "-d testds -m model_a" in slurm_cmd


def test_verify_completeness_null_metrics(mock_experiments_dir):
    df = pl.DataFrame(
        {
            "dataset_name": ["testds"] * 4,
            "model_name": [
                "model_a_1_seed101",
                "model_a_2_seed101",
                "model_a_1_seed102",
                "model_a_2_seed102",
            ],
            "random_state": [101, 101, 102, 102],
            "u_mass": [-0.5, None, -0.55, -0.65],  # null metric
            "c_v": [0.6, 0.62, 0.58, 0.61],
        }
    )

    report: CompletenessReport = verify_dataset_completeness(
        dataset_name="testds",
        dataset_type="standard",
        df=df,
        experiments_dir=mock_experiments_dir,
    )

    assert len(report.null_metric_runs) == 1
    assert not report.is_valid_for_merge


def test_merge_files_aborts_on_partial_models(
    tmp_path, mock_experiments_dir, monkeypatch
):
    # Point global EXPERIMENTS_DIR to mock directory
    monkeypatch.setattr("src.verification.EXPERIMENTS_DIR", mock_experiments_dir)

    out_csv = tmp_path / "testds_standard_merged.csv"
    f_partial = tmp_path / "testds_standard_model_a-20260810-100000-101.csv"

    # Only 2 of 4 runs
    pl.DataFrame(
        {
            "dataset_name": ["testds"] * 2,
            "model_name": ["model_a_1_seed101", "model_a_2_seed101"],
            "random_state": [101, 101],
            "u_mass": [-0.5, -0.6],
            "c_v": [0.6, 0.62],
        }
    ).write_csv(f_partial)

    # 1. By default, merge should abort (return False) and not write out_csv
    ok = merge_files(
        [f_partial],
        out_csv,
        dry_run=False,
        force=True,
        allow_partial=False,
    )
    assert not ok
    assert not out_csv.exists()

    # 2. With allow_partial=True, merge should succeed
    ok_override = merge_files(
        [f_partial],
        out_csv,
        dry_run=False,
        force=True,
        allow_partial=True,
    )
    assert ok_override
    assert out_csv.exists()


def test_check_missing_results_reads_files_with_mixed_column_types(
    tmp_path, monkeypatch, capsys
):
    """A column inferred as Int64 in one file and String in another still merges."""
    import scripts.analysis.check_missing_results as cmr

    results = tmp_path / "results"
    results.mkdir()
    pl.DataFrame(
        {
            "experiment_id": ["fed_standard_baseline"],
            "model_name": ["baseline_1"],
            "random_state": [36201624],
            "n_clusters": [10],
        }
    ).write_csv(results / "fed_standard_merged.csv")
    pl.DataFrame(
        {
            "experiment_id": ["fed_standard_baseline"],
            "model_name": ["baseline_2"],
            "random_state": [36201624],
            "n_clusters": [""],
        }
    ).write_csv(results / "fed_standard_baseline_m2-20261009-120000-36201624.csv")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_missing_results.py",
            "--dataset",
            "fed",
            "--result-type",
            "standard",
            "--results-dir",
            str(results),
        ],
    )

    with pytest.raises(SystemExit):
        cmr.main()

    assert "baseline" in capsys.readouterr().out
