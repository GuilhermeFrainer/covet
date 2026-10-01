"""Offline re-scoring of saved runs under the current evaluation protocol."""

import json
import math
import pathlib
import sys

import polars as pl
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis import recompute_topic_metrics as recompute  # noqa: E402
from src import evaluation  # noqa: E402

TEXTS = ["Economy.no jobs war", "war iraq troops", "jobs economy war"] * 10
BERTOPIC_CONFIG = {
    "experiment": {
        "dataset_path": "data/processed/toy_embeddings.parquet",
        "text_col": "clean_text",
        "coherence_metrics": ["c_npmi", "u_mass"],
        "diversity_metrics": ["topic_diversity", "irbo"],
    },
    "model": {"id": "baseline"},
}
ROW = {
    "run_uid": None,
    "dataset_name": "toy",
    "experiment_id": "toy_standard_baseline",
    "model_name": "baseline_1_seed1",
    "random_state": "1",
    "file_timestamp": "20261001-000000",
    "stopword_removal": "remove_rep_stopwords",
}


@pytest.fixture
def stub_inputs(monkeypatch):
    monkeypatch.setattr(recompute, "load_config", lambda *args: BERTOPIC_CONFIG)
    monkeypatch.setattr(recompute, "load_texts", lambda *args: TEXTS)


def test_scores_saved_topic_words_without_padding(stub_inputs):
    key = (ROW["dataset_name"], ROW["model_name"], ROW["file_timestamp"])
    index = {key: [["economyno", "jobs", "", "2008"], ["war"], ["war", "iraq"]]}
    record = recompute.score_run(ROW, index, {}, dry_run=False)

    assert record["skip_reason"] is None
    assert record["evaluation_protocol"] == evaluation.EVALUATION_PROTOCOL
    # "economyno" exists only in BERTopic's punctuation-stripped text.
    assert record["n_keywords_oov"] == 0
    assert record["n_topics_short"] == 3
    assert record["n_topics_unscored"] == 1  # the single-word topic
    assert math.isfinite(record["c_npmi"]) and math.isfinite(record["u_mass"])
    assert record["topic_diversity"] == pytest.approx(4 / 30)


def test_missing_topic_words_are_skipped(stub_inputs):
    record = recompute.score_run(ROW, {}, {}, dry_run=False)
    assert record["skip_reason"] == "topic words not found"


def test_tokenizer_follows_model_family_and_recorded_ngrams():
    tritopic = {"model": {"type": "fast_tritopic", "params": {}}}
    recorded = {"model_config": {"type": "tritopic", "params": {}}}
    current = {"model": {"type": "tritopic", "params": {"keyword_ngram_range": [1, 1]}}}
    assert recompute.tokenizer_key(ROW, BERTOPIC_CONFIG, None) == ("bertopic", True)
    assert recompute.tokenizer_key(ROW, tritopic, None) == ("tritopic", (1, 2))
    # The run's own manifest wins over the current config.
    assert recompute.tokenizer_key(ROW, current, recorded) == ("tritopic", (1, 2))
    assert recompute.tokenizer_key(ROW, current, None) == ("tritopic", (1, 1))


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
    row = {"stopword_removal": regime, "source_file": source}
    assert recompute.removes_rep_stopwords(row) is expected


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
    row = {**ROW, "run_uid": "abc"}
    assert recompute.topic_words_for(row, {}) == [["war", "iraq"]]


def test_merged_rows_win_over_raw_duplicates(tmp_path, monkeypatch):
    columns = {"dataset_name": ["toy"], "model_name": ["m_1"], "file_timestamp": ["t"]}
    pl.DataFrame({**columns, "c_v": ["raw"]}).write_csv(tmp_path / "toy_raw.csv")
    pl.DataFrame({**columns, "c_v": ["merged"]}).write_csv(
        tmp_path / "toy_standard_merged.csv"
    )
    monkeypatch.setattr(recompute, "RESULTS_DIR", tmp_path)
    rows = recompute.load_result_rows()
    assert rows["c_v"].to_list() == ["merged"]
