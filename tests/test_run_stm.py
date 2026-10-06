import gzip
import json
import logging
import pathlib
import sys

import numpy as np
import polars as pl
import pytest

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.experiments.run_stm as run_stm  # noqa: E402
from scripts.experiments.run_stm import resolve_stm_inputs  # noqa: E402
from src.evaluation import representation_tokens  # noqa: E402
from src.utils import load_config  # noqa: E402

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture(autouse=True)
def cleanup_pipeline_logger():
    # run_stm.main configures the shared logger; restore it for later tests.
    yield
    logger = logging.getLogger("pipeline")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = True


def test_unstemmed_config_uses_unstemmed_inputs():
    name, rds_path, bow_path = resolve_stm_inputs(
        {
            "dataset_path": "data/processed/trump_embeddings.parquet",
            "text_col": "clean_text",
        }
    )
    assert name == "trump"
    assert rds_path == PROCESSED_DIR / "trump_stm_data.rds"
    assert bow_path == PROCESSED_DIR / "trump_bow.parquet"


def test_stemmed_config_uses_stemmed_inputs():
    name, rds_path, bow_path = resolve_stm_inputs(
        {
            "dataset_path": "data/processed/trump_embeddings.parquet",
            "text_col": "clean_text_stemmed",
        }
    )
    assert name == "trump"
    assert rds_path == PROCESSED_DIR / "trump_stemmed_stm_data.rds"
    assert bow_path == PROCESSED_DIR / "trump_stemmed_bow.parquet"


@pytest.mark.parametrize(
    ("exp_name", "expected_prefix"),
    [
        ("trump/trump_standard_stm", "trump"),
        ("trump_stemmed/trump_standard_stm", "trump_stemmed"),
        ("yelp/yelp_standard_stm", "yelp_s10000"),
        ("yelp_stemmed/yelp_standard_stm", "yelp_s10000_stemmed"),
        ("trump_s25000/trump_s25000_standard_stm", "trump_s25000"),
        ("trump_s25000_stemmed/trump_s25000_standard_stm", "trump_s25000_stemmed"),
    ],
)
def test_active_stm_configs_match_build_bow_outputs(exp_name, expected_prefix):
    config = load_config(exp_name, EXPERIMENTS_DIR)
    _, rds_path, bow_path = resolve_stm_inputs(config["experiment"])
    assert rds_path.name == f"{expected_prefix}_stm_data.rds"
    assert bow_path.name == f"{expected_prefix}_bow.parquet"


def test_runs_vary_models_before_seeds():
    config = {"models": [{"id": "a"}, {"id": "b"}]}
    runs = run_stm.expand_runs(config, [1, 2])
    assert [(m["id"], seed) for m, seed in runs] == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]


def test_local_r_command_runs_the_script_directly():
    cmd = run_stm.build_r_command(["--k", "5"], "local", "img", "docker")
    assert cmd == ["Rscript", "scripts/r_scripts/train_stm.R", "--k", "5"]


def test_container_r_command_mounts_the_project(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    cmd = run_stm.build_r_command(["--k", "5"], "docker", "cast:stm", "podman")
    assert cmd[:3] == ["podman", "run", "--rm"]
    assert f"{run_stm.PROJECT_ROOT}:/work" in cmd
    assert "OMP_NUM_THREADS=3" in cmd
    # The repository .Rprofile would activate renv inside the image.
    image_at = cmd.index("cast:stm")
    assert cmd[image_at + 1 :] == [
        "Rscript",
        "--no-init-file",
        "scripts/r_scripts/train_stm.R",
        "--k",
        "5",
    ]


def test_reads_r_outputs(tmp_path):
    with gzip.open(tmp_path / "beta.csv.gz", "wt") as f:
        f.write("V1,V2,V3\n0.2,0.5,0.3\n0.6,0.1,0.3\n")
    with gzip.open(tmp_path / "theta.csv.gz", "wt") as f:
        f.write("V1,V2\n0.9,0.1\n")
    (tmp_path / "vocab.txt").write_text("a\nb\nc\n", encoding="utf-8")
    (tmp_path / "doc_index.txt").write_text("7\n", encoding="utf-8")
    (tmp_path / "duration.txt").write_text("1.5\n", encoding="utf-8")

    outputs = run_stm.read_r_outputs(tmp_path)

    assert outputs["beta"].shape == (2, 3)
    assert outputs["theta"].shape == (1, 2)
    assert outputs["vocab"] == ["a", "b", "c"]
    assert outputs["doc_index"] == [7]
    assert outputs["duration"] == 1.5


def test_exports_argmax_assignments_and_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr(run_stm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_stm, "ASSIGNMENTS_DIR", tmp_path / "assignments")
    documents = pl.DataFrame({"index": [10, 11, 12, 13]})
    theta = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    covariates = pl.DataFrame({"party": ["a", "b", "a", "b"]})

    links = run_stm.export_assignments(
        {"model_id": "stm_k2", "dataset_name": "toy"},
        documents,
        theta,
        covariates,
        config={},
    )

    run_dir = tmp_path / links["assignment_manifest_path"]
    manifest = json.loads(run_dir.read_text(encoding="utf-8"))
    assignments = pl.read_parquet(tmp_path / links["assignments_path"])
    assert assignments["topic_id"].to_list() == [0, 1, 0, 1]
    assert assignments["index"].to_list() == [10, 11, 12, 13]
    assert manifest["assignment_semantics"] == "argmax_theta"
    assert manifest["metadata_alignment"]["meta_ami_by_covariate"]["party"] == 1.0
    assert links["meta_ami_mean"] == 1.0


def test_out_of_range_model_index_fails():
    assert (
        run_stm.main(["--exp", "gadarian/gadarian_standard_stm", "--model", "99"]) == 1
    )


@pytest.mark.parametrize("prefix", ["gadarian", "gadarian_stemmed"])
def test_bow_vocabulary_matches_neural_topic_words(prefix):
    bow_path = PROCESSED_DIR / f"{prefix}_bow.parquet"
    embeddings_path = PROCESSED_DIR / "gadarian_embeddings.parquet"
    if not (bow_path.exists() and embeddings_path.exists()):
        pytest.skip("Processed gadarian data is not available.")
    text_col = "clean_text_stemmed" if prefix.endswith("_stemmed") else "clean_text"
    texts = pl.read_parquet(embeddings_path, columns=[text_col])[text_col].to_list()
    neural_vocab = {w for doc in representation_tokens(texts) for w in doc}

    bow = pl.read_parquet(bow_path, columns=["bow_text"])["bow_text"]
    stm_vocab = {w for doc in bow for w in doc.split()}

    assert stm_vocab == neural_vocab
