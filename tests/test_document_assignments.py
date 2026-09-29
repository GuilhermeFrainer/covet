import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.data import load_and_prep_data
from src.document_assignments import (
    AssignmentRun,
    atomic_json,
    inspect_run,
    representative_identities,
    validate_assignments,
)
from src.metadata_alignment import metadata_alignment


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source.parquet"
    pl.DataFrame(
        {
            "index": [3367, 5129, 20, 21, 22, 23],
            "id": [1, 1, 2, 2, 3, 3],
            "text": [
                "same words",
                "same words",
                "other words",
                " ",
                None,
                "last words",
            ],
            "embedding": [[float(i), 1.0] for i in range(6)],
            "rate": [1.0, 5.0, 2.0, 3.0, 4.0, 6.0],
            "kind": ["A", "B", "A", "B", "A", "B"],
        }
    ).write_parquet(source)
    config = {
        "experiment": {
            "dataset_path": str(source),
            "covariates": {"numerical": ["rate"], "categorical": ["kind"]},
            "coherence_metrics": [],
            "diversity_metrics": [],
        }
    }
    return load_and_prep_data(config, 42, return_prepared=True), config


class FittedModel:
    topics_ = [0, 0, -1, 1]
    hdbscan_model = SimpleNamespace(labels_=np.array([8, 8, -1, 9]))

    def get_topic_info(self):
        return pd.DataFrame(
            {
                "Topic": [-1, 0, 1],
                "Count": [1, 2, 1],
                "Representation": [["other"], ["same"], ["last"]],
                "Representative_Docs": [
                    ["other words"],
                    ["same words"],
                    ["last words"],
                ],
            }
        )

    def fit_transform(self, **kwargs):
        return self.topics_, None

    def get_topics(self):
        return {-1: [("other", 1.0)], 0: [("same", 1.0)], 1: [("last", 1.0)]}


def make_run(tmp_path, prepared, enabled=True):
    data, config = prepared
    return AssignmentRun(
        tmp_path / "output" / "document_assignments",
        "fed",
        data,
        {"id": "baseline", "bertopic": {"params": {"nr_topics": 50}}},
        {"model_id": "baseline", "seed": 42},
        config,
        enabled,
    )


def test_filter_sample_and_duplicate_identity(prepared):
    data, config = prepared
    assert data.documents["source_row_ordinal"].to_list() == [0, 1, 2, 5]
    assert data.documents["source_document_key"].n_unique() == 4
    assert data.documents["id"].to_list() == [1, 1, 2, 3]
    config["experiment"]["sample_size"] = 3
    sampled = load_and_prep_data(config, 7, return_prepared=True)
    for position, ordinal in enumerate(sampled.documents["source_row_ordinal"]):
        assert sampled.embeddings[position][0] == ordinal
        assert (
            sampled.text[position]
            == ["same words", "same words", "other words", " ", None, "last words"][
                ordinal
            ]
        )
    source_rates = [1.0, 5.0, 2.0, 3.0, 4.0, 6.0]
    rates = [source_rates[r] for r in sampled.documents["source_row_ordinal"]]
    assert sampled.metadata["rate"].to_list() == [
        (r - min(rates)) / (max(rates) - min(rates)) for r in rates
    ]
    assert len(load_and_prep_data(config, 7)) == 3  # legacy callers


def test_export_roundtrip_noise_remapping_and_representatives(tmp_path, prepared):
    run = make_run(tmp_path, prepared)
    run.after_fit(FittedModel())
    assignments = pl.read_parquet(run.directory / "assignments.parquet")
    assert assignments["topic_id"].to_list() == [0, 0, -1, 1]
    summary = inspect_run(run.path)
    assert [r["topic_id"] for r in summary["selected_documents"]] == [0, 0]
    assert summary["topics"]["0"]["raw_metadata"]["rate"]["mean"] == 3.0
    reps = json.loads((run.directory / "representative_documents.json").read_text())
    assert reps[1]["resolution"] == "ambiguous"
    assert reps[1]["source_document_key"] is None
    assert len(reps[1]["candidate_source_keys"]) == 2
    assert run.payload["strength_kind"] is None
    other = make_run(tmp_path, prepared)
    other.after_fit(FittedModel())
    assert other.uid != run.uid
    assert run.path.exists()


@pytest.mark.parametrize(
    "labels", [[0], [[0], [0], [-1], [1]], [0, None, -1, 1], 0, [0, 1, -1, 1]]
)
def test_invalid_labels_rejected(tmp_path, prepared, labels):
    run = make_run(tmp_path, prepared)
    model = FittedModel()
    model.topics_ = labels
    with pytest.raises(ValueError):
        run.after_fit(model)
    assert json.loads(run.path.read_text())["export_status"] == "failure"


def test_tritopic_api_and_exact_representative_indices(prepared):
    data, _ = prepared
    model = SimpleNamespace(
        topics_=[SimpleNamespace(topic_id=0)], labels_=np.array([0, 0, -1, 1])
    )
    table = pl.DataFrame({"topic_id": [0, -1, 1], "count": [2, 1, 1]})
    assignments = validate_assignments(model, data.documents, table, is_tritopic=True)
    model.get_topic_info = lambda: pd.DataFrame(
        {"Topic": [0], "Representative_Docs": [[1]]}
    )
    reps = representative_identities(model, assignments, data.text)
    assert reps[0]["input_position"] == 1
    assert reps[0]["resolution"] == "unique"


def test_evaluation_failure_retains_fit(tmp_path, prepared, monkeypatch):
    from src import evaluation

    def fail(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(evaluation, "bertopic_output_to_octis", fail)
    data, config = prepared
    run = make_run(tmp_path, prepared)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        run.execute(
            topic_model=FittedModel(),
            model_id="baseline",
            text=data.text,
            embeddings=data.embeddings,
            config=config,
        )
    manifest = json.loads(run.path.read_text())
    assert manifest["fit_status"] == "success"
    assert manifest["export_status"] == "success"
    assert manifest["evaluation_status"] == "failure"
    assert manifest["status"] == "failure"
    assert inspect_run(run.path)["topics"]["0"]["document_count"] == 2


def test_interrupted_atomic_write_preserves_previous_json(tmp_path, monkeypatch):
    import src.document_assignments as module

    path = tmp_path / "manifest.json"
    atomic_json(path, {"status": "pending"})

    def fail(*args):
        raise OSError("interrupted replace")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        atomic_json(path, {"status": "success"})
    assert json.loads(path.read_text()) == {"status": "pending"}


def test_source_changes_are_rejected(tmp_path, prepared):
    run = make_run(tmp_path, prepared)
    run.after_fit(FittedModel())
    source = prepared[1]["experiment"]["dataset_path"]
    pl.DataFrame({"text": ["replacement"]}).write_parquet(source)
    with pytest.raises(ValueError, match="checksum"):
        inspect_run(run.path)


def test_parquet_failure_does_not_commit(tmp_path, prepared, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    run = make_run(tmp_path, prepared)
    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail)
    with pytest.raises(OSError):
        run.after_fit(FittedModel())
    manifest = json.loads(run.path.read_text())
    assert manifest["export_status"] == "failure"
    assert manifest["status"] != "success"
    assert "assignments.parquet" not in manifest["artifacts"]


def test_archive_preserves_execution_artifacts(tmp_path, prepared):
    from scripts.analysis.merge_results import archive_files, group_files, merge_files

    run = make_run(tmp_path, prepared)
    run.after_fit(FittedModel())
    output = run.directory.parents[2]
    raw = output / "fed_standard_baseline-20260921-120000-42.json"
    topics = pl.read_json(run.directory / "topics.json").with_columns(
        pl.lit("fed").alias("dataset_name"),
        pl.lit("remove_rep_stopwords").alias("stopword_removal"),
    )
    topics.write_json(raw)
    grouped = group_files(output, ".json", ignore_suffix="_merged")
    assert grouped[("fed", "standard")] == [raw]
    merged = output / "fed_standard_merged.json"
    merge_files([raw], merged, dry_run=False, force=True, allow_partial=True)
    archive_files(
        [raw],
        output / "archive",
        "fed",
        "standard",
        merged,
        dry_run=False,
        keep_originals=False,
    )
    assert not raw.exists()
    assert pl.read_json(merged)["run_uid"].unique().to_list() == [run.uid]
    assert inspect_run(run.path)["run_uid"] == run.uid


@pytest.mark.parametrize("name", ["baseline", "mv_spectral", "umap_spectral"])
def test_real_smoke_fit(tmp_path, name):
    """Small synthetic fits, distinct from the full-corpus scientific reruns."""
    from pathlib import Path

    from src.models import create_topic_model_instance
    from src.utils import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(f"fed/fed_qualitative_k50_{name}", root / "experiments")
    model_config = config["model"]
    setting = (
        model_config["bertopic"]["params"]
        if name == "baseline"
        else model_config["clustering"]["params"]
    )
    setting["nr_topics" if name == "baseline" else "n_clusters"] = 3
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(3, 16)) * 4
    embeddings = np.vstack(
        [centers[i % 3] + rng.normal(size=16) * 0.1 for i in range(90)]
    )
    vocab = [
        "apple banana orange fruit orchard",
        "train road truck car transport",
        "bank money market interest finance",
    ]
    source = tmp_path / "smoke.parquet"
    pl.DataFrame(
        {
            "text": [vocab[i % 3] for i in range(90)],
            "embedding": pl.Series(embeddings, dtype=pl.Array(pl.Float64, 16)),
            "rate": [float(i % 3) for i in range(90)],
        }
    ).write_parquet(source)
    config["experiment"].update(
        dataset_path=str(source),
        text_col="text",
        embedding_col="embedding",
        covariates={"numerical": ["rate"]},
        coherence_metrics=[],
        diversity_metrics=[],
    )
    prepared = load_and_prep_data(config, 42, return_prepared=True)
    model = create_topic_model_instance(
        model_config, prepared.metadata, 42, remove_rep_stopwords=True
    )
    run = AssignmentRun(
        tmp_path / "output",
        "synthetic_smoke",
        prepared,
        model_config,
        {"model_id": name, "seed": 42},
        config,
    )
    run.execute(
        topic_model=model,
        model_id=name,
        text=prepared.text,
        embeddings=prepared.embeddings,
        scaled_metadata=prepared.metadata,
        config=config,
    )
    assert run.payload["status"] == "success"
    assert pl.read_parquet(run.directory / "assignments.parquet").height == 90


def test_rerun_configs_preserve_scientific_parameters():
    import copy
    from pathlib import Path

    from src.utils import load_config

    root = Path(__file__).resolve().parents[1] / "experiments"
    for name in ("baseline", "mv_spectral", "umap_spectral"):
        original = load_config(f"fed/fed_standard_{name}", root)
        rerun = load_config(f"fed/fed_qualitative_k50_{name}", root)
        expected = copy.deepcopy(original["model"])
        if name == "baseline":
            expected["bertopic"]["params"]["nr_topics"] = 50
        else:
            expected["clustering"]["params"]["n_clusters"] = 50
        assert rerun["model"] == expected
        assert rerun["experiment"]["random_state"] == [36201624, 62613654, 57116123]
        assert rerun["experiment"]["text_col"] == "clean_text"
        assert rerun["experiment"].get("sample_size") is None


@pytest.mark.parametrize("target_index", [None, 1])
def test_optimizer_grid_and_split_export(tmp_path, prepared, monkeypatch, target_index):
    from src import evaluation, models
    from src.optimizer import Optimizer

    data, config = prepared
    monkeypatch.setattr(
        models, "create_topic_model_instance", lambda **kwargs: FittedModel()
    )
    monkeypatch.setattr(evaluation, "bertopic_output_to_octis", lambda *args: {})
    optimizer = Optimizer(
        texts=data.text,
        embeddings=data.embeddings,
        scaled_metadata=data.metadata,
        prepared_data=data,
        assignment_output_dir=tmp_path / "output" / "document_assignments",
        model_config={
            "id": "spectral",
            "clustering": {"params": {"n_clusters": [2, 3]}},
        },
        experiment_config=config,
        experiment_id="fed_export_grid",
        random_state=[42, 43],
        file_timestamp="20260921-120000",
        remove_rep_stopwords=True,
    )
    optimizer.run(target_index=target_index)
    expected = 4 if target_index is None else 1
    assert len(optimizer.results) == expected
    assert len({row["run_uid"] for row in optimizer.results}) == expected
    for row in optimizer.results:
        from pathlib import Path

        manifest = json.loads(Path(row["assignment_manifest_path"]).read_text())
        assert manifest["status"] == "success"
        assert manifest["input"]["sampling_seed"] == 42
        assert manifest["requested_topic_setting"] in (2, 3)
        assert manifest["seed"] == row["random_state"]
        assert (
            inspect_run(row["assignment_manifest_path"])["topics"]["0"][
                "document_count"
            ]
            == 2
        )
    if target_index == 1:
        assert optimizer.results[0]["model_name"] == "spectral_1_seed43"
        assert optimizer.results[0]["random_state"] == 43
    (tmp_path / "results").mkdir()
    optimizer.save_results(tmp_path / "results" / "grid.csv")
    assert (
        pl.read_csv(tmp_path / "results" / "grid.csv")["run_uid"].n_unique() == expected
    )
    assert (
        pl.read_json(tmp_path / "output" / "grid.json")["run_uid"].n_unique()
        == expected
    )


def test_optimizer_retains_assignments_after_evaluation_failure(
    tmp_path, prepared, monkeypatch
):
    from src import evaluation, models
    from src.optimizer import Optimizer

    def fail(*args):
        raise RuntimeError("metric failure")

    data, config = prepared
    monkeypatch.setattr(
        models, "create_topic_model_instance", lambda **kwargs: FittedModel()
    )
    monkeypatch.setattr(evaluation, "bertopic_output_to_octis", fail)
    optimizer = Optimizer(
        texts=data.text,
        embeddings=data.embeddings,
        scaled_metadata=data.metadata,
        prepared_data=data,
        assignment_output_dir=tmp_path / "assignments",
        model_config={"id": "baseline"},
        experiment_config=config,
        experiment_id="fed_export_failure",
        random_state=42,
        file_timestamp="test",
    )
    optimizer.run()
    assert not optimizer.results
    failure = optimizer.run_manifests[0]
    assert failure["status"] == "failure"
    from pathlib import Path

    manifest = json.loads(Path(failure["assignment_manifest_path"]).read_text())
    assert manifest["fit_status"] == "success"
    assert manifest["export_status"] == "success"
    assert manifest["evaluation_status"] == "failure"
    assert (
        inspect_run(failure["assignment_manifest_path"])["run_uid"]
        == failure["run_uid"]
    )


@pytest.mark.parametrize("model_id", ["baseline", "mv_spectral"])
def test_optimizer_cli_passes_prepared_inputs(
    tmp_path, prepared, monkeypatch, model_id
):
    import logging

    from scripts.experiments import run_optimizer as runner
    from src import evaluation, models

    _, config = prepared
    config["experiment"].update(name="fed_cli_export", random_state=[42, 43])
    config["model"] = {"id": model_id}
    monkeypatch.setattr(runner.utils, "load_config", lambda *args: config)
    monkeypatch.setattr(
        runner.logger_config,
        "setup_logging",
        lambda *args: logging.getLogger("pipeline"),
    )
    monkeypatch.setattr(
        models, "create_topic_model_instance", lambda **kwargs: FittedModel()
    )
    monkeypatch.setattr(evaluation, "bertopic_output_to_octis", lambda *args: {})
    monkeypatch.setattr(runner.make_table, "generate_latex_table", lambda *args: "test")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    for name in ("RESULTS_DIR", "LOG_DIR", "TABLES_DIR"):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(runner, name, path)
    monkeypatch.setattr(
        "sys.argv", ["run_optimizer.py", "--exp", "test", "--model", "2"]
    )
    runner.main()
    manifests = list(
        (tmp_path / "output").glob("document_assignments/*/*/manifest.json")
    )
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["seed"] == 43
    assert manifest["input"]["sampling_seed"] == 42
    assert manifest["status"] == "success"


def test_prepared_data_keeps_raw_covariates_aligned(prepared):
    data, _ = prepared
    assert data.covariates.columns == ["rate", "kind"]
    assert data.covariates.height == len(data.text)


def test_assignment_run_reports_alignment_in_csv_row_and_metrics_json(
    tmp_path, prepared, monkeypatch
):
    from src import evaluation

    # The fixture config requests no coherence/diversity metrics.
    monkeypatch.setattr(evaluation, "bertopic_output_to_octis", lambda m: {})
    data, config = prepared
    run = make_run(tmp_path, prepared)
    metrics, _ = run.execute(
        topic_model=FittedModel(),
        model_id="fed_model",
        text=data.text,
        embeddings=data.embeddings,
        config=config,
        scaled_metadata=data.metadata,
    )
    expected = metadata_alignment(FittedModel.topics_, data.covariates)
    assert metrics["meta_ami_mean"] == pytest.approx(expected["meta_ami_mean"])
    assert "meta_ami_by_covariate" not in metrics

    saved = json.loads((run.directory / "metrics.json").read_text(encoding="utf-8"))
    assert saved["meta_ami_by_covariate"] == pytest.approx(
        expected["meta_ami_by_covariate"]
    )
    manifest = json.loads(run.path.read_text(encoding="utf-8"))
    assert "method" in manifest["metadata_alignment"]


def test_backfill_reproduces_run_time_alignment(tmp_path, prepared, monkeypatch):
    import importlib.util

    from src import evaluation
    from src.document_assignments import file_checksum

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analysis"
        / "backfill_metadata_alignment.py"
    )
    spec = importlib.util.spec_from_file_location("backfill", script)
    backfill = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backfill)

    monkeypatch.setattr(evaluation, "bertopic_output_to_octis", lambda m: {})
    data, config = prepared
    run = make_run(tmp_path, prepared)
    metrics, _ = run.execute(
        topic_model=FittedModel(),
        model_id="fed_model",
        text=data.text,
        embeddings=data.embeddings,
        config=config,
    )
    source = Path(config["experiment"]["dataset_path"])
    record = backfill.score_run(run.path, {file_checksum(source): source}, {})
    assert record["skip_reason"] is None
    assert record["meta_ami_mean"] == pytest.approx(metrics["meta_ami_mean"])
    assert record["matches_run_time_value"] is True

    missing = backfill.score_run(run.path, {}, {})
    assert missing["skip_reason"] == "source dataset not found locally"
