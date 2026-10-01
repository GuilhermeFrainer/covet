"""In-place re-scoring of merged results under the current evaluation protocol."""

import json
import pathlib
import sys
import zipfile

import numpy as np
import polars as pl
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis import recompute_topic_metrics as recompute  # noqa: E402
from src import evaluation  # noqa: E402

RNG = np.random.default_rng(0)
VOCAB = [f"w{i}" for i in range(30)]
TEXTS = [" ".join(RNG.choice(VOCAB, size=8)) for _ in range(200)] + [
    "Economy.no jobs"
] * 5
CONFIG = {
    "experiment": {
        "dataset_path": "data/processed/toy_embeddings.parquet",
        "text_col": "clean_text",
        "coherence_metrics": ["c_npmi", "u_mass"],
        "diversity_metrics": ["topic_diversity"],
    },
    "model": {"id": "baseline"},
}
FULL_TOPICS = [VOCAB[0:10], VOCAB[10:20]]
SHORT_TOPICS = [VOCAB[0:10], ["economyno", "jobs"]]
STAMP = "20261002_000000"


def _legacy(topics):
    """The stored NPMI under the previous protocol, for unpadded topics."""
    texts = recompute.tokenize(TEXTS, ("bertopic", True, False))
    return repr(recompute.legacy_npmi(topics, texts))


def _row(name, topics_count, npmi, protocol=None):
    return {
        "experiment_id": "toy_standard_baseline",
        "model_name": name,
        "dataset_name": "toy",
        "random_state": "1",
        "file_timestamp": "20260901-000000",
        "n_topics": str(topics_count),
        "c_npmi": npmi,
        "u_mass": "-1.5",
        "topic_diversity": "0.9",
        "notes": 'keep, this "exact" text',
        "evaluation_protocol": protocol,
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A results tree with four rows and their saved topic words."""
    results, output = tmp_path / "results", tmp_path / "output"
    results.mkdir()
    output.mkdir()
    for name, value in (
        ("RESULTS_DIR", results),
        ("ARCHIVE_DIR", results / "archive"),
        ("OUTPUT_DIR", output),
        ("ASSIGNMENTS_DIR", output / "document_assignments"),
    ):
        monkeypatch.setattr(recompute, name, value)
    monkeypatch.setattr(recompute, "load_config", lambda *args: CONFIG)
    monkeypatch.setattr(recompute, "load_texts", lambda *args: TEXTS)

    rows = [
        _row("full_1", 2, _legacy(FULL_TOPICS)),  # verifiable
        _row("short_1", 2, "0.99"),  # padded originally: unverifiable
        _row("orphan_1", 2, "0.5"),  # no saved topic words
        _row("current_1", 2, "0.4", evaluation.EVALUATION_PROTOCOL),
    ]
    path = results / "toy_standard_merged.csv"
    pl.DataFrame(rows, schema={k: pl.String for k in rows[0]}).write_csv(path)
    topics = [
        {
            "dataset_name": "toy",
            "model_id": model,
            "file_timestamp": "20260901-000000",
            "topic_id": topic_id,
            "representation": words,
        }
        for model, words_list in (
            ("full_1", FULL_TOPICS),
            ("short_1", SHORT_TOPICS),
            ("current_1", FULL_TOPICS),
        )
        for topic_id, words in enumerate(words_list)
    ]
    (output / "toy_standard_merged.json").write_text(json.dumps(topics))
    return path


def _process(path, dry_run=False):
    return recompute.process_file(
        path, recompute.load_topic_words(), recompute.Corpora(), dry_run, STAMP
    )


def test_updates_in_place_and_archives_the_original(project):
    original_bytes = project.read_bytes()
    original = pl.read_csv(project, infer_schema_length=0)
    report = _process(project)
    assert report["update"] == 2 and report["current"] == 1
    assert report["topic words not found"] == 1
    assert report["verified"] == 1 and report["unverifiable"] == 1

    archived = (
        project.parent / "archive" / f"toy_standard_merged_pre_recompute_{STAMP}.zip"
    )
    with zipfile.ZipFile(archived) as bundle:
        assert bundle.read(project.name) == original_bytes

    updated = pl.read_csv(project, infer_schema_length=0)
    assert updated.columns[: original.width] == original.columns
    for column in ("experiment_id", "model_name", "n_topics", "notes"):
        assert updated[column].equals(original[column])
    by_name = {row["model_name"]: row for row in updated.to_dicts()}
    for name in ("full_1", "short_1"):
        row = by_name[name]
        assert row["evaluation_protocol"] == evaluation.EVALUATION_PROTOCOL
        assert (
            row["c_npmi_padded"]
            == original.filter(pl.col("model_name") == name)["c_npmi"][0]
        )
        assert row["u_mass_padded"] == "-1.5"
    assert by_name["short_1"]["n_topics_short"] == "1"
    assert by_name["short_1"]["n_keywords_oov"] == "0"  # "economyno" now found
    assert by_name["short_1"]["c_npmi"] != "0.99"
    for name, npmi in (("orphan_1", "0.5"), ("current_1", "0.4")):
        row = by_name[name]
        assert row["c_npmi"] == npmi and row["c_npmi_padded"] is None
    assert by_name["orphan_1"]["evaluation_protocol"] is None


def test_dry_run_writes_nothing(project):
    original = project.read_bytes()
    report = _process(project, dry_run=True)
    assert report["update"] == 2 and report["verified"] == 1
    assert project.read_bytes() == original
    assert not (project.parent / "archive").exists()


def test_rerun_is_a_no_op(project):
    _process(project)
    after_first = project.read_bytes()
    report = recompute.process_file(
        project,
        recompute.load_topic_words(),
        recompute.Corpora(),
        False,
        "20261002_000001",
    )
    assert report.get("update") is None and report["current"] == 3
    assert project.read_bytes() == after_first
    assert len(list((project.parent / "archive").iterdir())) == 1


def test_unreproduced_stored_score_leaves_file_untouched(project):
    frame = pl.read_csv(project, infer_schema_length=0).with_columns(
        pl.when(pl.col("model_name") == "full_1")
        .then(pl.lit("0.123"))
        .otherwise(pl.col("c_npmi"))
        .alias("c_npmi")
    )
    frame.write_csv(project)
    original = project.read_bytes()
    with pytest.raises(recompute.RecomputeError, match="not reproduced"):
        _process(project)
    assert project.read_bytes() == original
    assert not (project.parent / "archive").exists()


def test_topic_count_mismatch_leaves_file_untouched(project):
    frame = pl.read_csv(project, infer_schema_length=0).with_columns(
        pl.when(pl.col("model_name") == "short_1")
        .then(pl.lit("5"))
        .otherwise(pl.col("n_topics"))
        .alias("n_topics")
    )
    frame.write_csv(project)
    original = project.read_bytes()
    with pytest.raises(recompute.RecomputeError, match="do not match"):
        _process(project, dry_run=True)
    assert project.read_bytes() == original


def test_duplicate_run_keys_are_rejected(project):
    frame = pl.read_csv(project, infer_schema_length=0)
    pl.concat([frame, frame.head(1)]).write_csv(project)
    with pytest.raises(recompute.RecomputeError, match="duplicate"):
        _process(project, dry_run=True)


def test_refuses_to_run_with_unmerged_raw_files(project, monkeypatch, capsys):
    (project.parent / "toy_standard_baseline_m1-20261001.csv").write_text("a\n1\n")
    monkeypatch.setattr(sys, "argv", ["recompute", "--datasets", "toy"])
    with pytest.raises(SystemExit) as exit_info:
        recompute.main()
    assert exit_info.value.code == 1
    assert "run merge_results.py first" in capsys.readouterr().out


def test_tokenizer_follows_model_family_protocol_and_recorded_ngrams():
    row = {"stopword_removal": "remove_rep_stopwords"}
    bertopic = {"id": "baseline"}
    assert recompute.tokenizer_key(row, bertopic) == ("bertopic", True, True)
    assert recompute.tokenizer_key(row, bertopic, legacy=True) == (
        "bertopic",
        True,
        False,
    )
    tritopic = {"type": "tritopic", "params": {"keyword_ngram_range": [1, 1]}}
    assert recompute.tokenizer_key(row, tritopic) == ("tritopic", (1, 1), False)
    assert recompute.tokenizer_key(row, tritopic, legacy=True) == (
        "tritopic",
        (1, 1),
        False,
    )
    stale = {"type": "fast_tritopic", "params": {}}
    assert recompute.tokenizer_key(row, stale) == ("tritopic", (1, 2), False)
    manifest = {"model_config": stale}
    assert recompute.model_config_for(CONFIG, manifest) is stale
    assert recompute.model_config_for(CONFIG, None) is CONFIG["model"]


def test_bertopic_tokenization_sees_punctuation_stripped_words():
    current = recompute.tokenize(["Economy.no jobs"], ("bertopic", True, True))
    legacy = recompute.tokenize(["Economy.no jobs"], ("bertopic", True, False))
    assert current == [["economyno", "jobs"]]
    assert legacy == [["economy", "jobs"]]


@pytest.mark.parametrize(
    ("regime", "source", "expected"),
    [
        ("remove_rep_stopwords", "", True),
        ("keep_rep_stopwords", "", False),
        (None, "fed_standard_merged.csv", True),
        (None, "fed_no_stopword_removal_merged.csv", False),
    ],
)
def test_stopword_regime(regime, source, expected):
    row = {"stopword_removal": regime}
    assert recompute.removes_rep_stopwords(row, source) is expected


def test_topic_words_fall_back_to_assignment_export(tmp_path, monkeypatch):
    run_dir = tmp_path / "toy" / "abc"
    run_dir.mkdir(parents=True)
    (run_dir / "topics.json").write_text(
        json.dumps(
            [
                {"topic_id": -1, "representation": ["noise"]},
                {"topic_id": 0, "representation": ["war", "iraq"]},
            ]
        )
    )
    monkeypatch.setattr(recompute, "ASSIGNMENTS_DIR", tmp_path)
    row = {
        "dataset_name": "toy",
        "model_name": "m_1",
        "file_timestamp": "t",
        "run_uid": "abc",
    }
    assert recompute.topic_words_for(row, {}) == [["war", "iraq"]]


@pytest.mark.parametrize(
    ("name", "dataset"),
    [
        ("trump_standard_merged.csv", "trump"),
        ("trump_s25000_standard_merged.csv", "trump_s25000"),
        ("trump_s25000_standard_baseline_m1-20261001-1.csv", "trump_s25000"),
        ("yelp_stemmed_merged.csv", "yelp"),
        ("fed_no_stopword_removal_merged.csv", "fed"),
    ],
)
def test_file_dataset_handles_underscored_dataset_names(name, dataset):
    assert recompute.file_dataset(name) == dataset
