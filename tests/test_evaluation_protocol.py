"""Unpadded topic scoring and BERTopic-aligned evaluation tokenization."""

import math
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
from bertopic import BERTopic
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import InvertedRBO, TopicDiversity
from sklearn.feature_extraction.text import CountVectorizer

from src import evaluation, training
from src.data import load_and_prep_data, load_texts

RNG = np.random.default_rng(0)
VOCAB = [f"w{i}" for i in range(30)]
TEXTS = [list(RNG.choice(VOCAB, size=8)) for _ in range(200)]
FULL = [VOCAB[0:10], VOCAB[10:20], VOCAB[20:30]]


@pytest.mark.parametrize("measure", ["c_npmi", "u_mass", "c_v"])
def test_full_length_topics_match_octis(measure):
    expected = Coherence(texts=TEXTS, topk=10, measure=measure).score({"topics": FULL})
    actual = evaluation.compute_coherence({"topics": FULL}, TEXTS, measure)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("measure", ["c_npmi", "u_mass", "c_v"])
def test_single_word_topics_are_excluded_not_scored(measure):
    with_single = {"topics": FULL + [["w0"]]}
    assert evaluation.compute_coherence(with_single, TEXTS, measure) == pytest.approx(
        evaluation.compute_coherence({"topics": FULL}, TEXTS, measure)
    )


def test_padding_no_longer_inflates_coherence():
    short = ["w0", "w1", "w2"]
    padded = short + ["w0"] * 7
    unpadded_score = evaluation.compute_coherence({"topics": [short]}, TEXTS, "c_npmi")
    # Repeated words are scored once, so pre-padded input changes nothing...
    assert evaluation.compute_coherence(
        {"topics": [padded]}, TEXTS, "c_npmi"
    ) == pytest.approx(unpadded_score)
    # ...whereas OCTIS rewards the self-pairs that padding creates.
    octis_padded = Coherence(texts=TEXTS, topk=10, measure="c_npmi").score(
        {"topics": [padded]}
    )
    assert octis_padded > unpadded_score


def test_short_first_topic_is_scored():
    topics = {"topics": [["w0", "w1", "w2"]] + FULL}
    for measure in ("c_npmi", "u_mass", "c_v"):
        assert math.isfinite(evaluation.compute_coherence(topics, TEXTS, measure))
    for measure in ("irbo", "topic_diversity"):
        assert math.isfinite(evaluation.compute_diversity(measure, topics))


def test_no_scorable_topic_returns_nan():
    topics = {"topics": [["w0"], ["absent", "missing"]]}
    assert math.isnan(evaluation.compute_coherence(topics, TEXTS, "c_npmi"))


def test_topic_diagnostics():
    topics = {"topics": FULL + [["w0", "w1"], ["w0"], ["w5", "absent"]]}
    assert evaluation.topic_diagnostics(topics, TEXTS) == {
        "n_topics_short": 3,
        "n_topics_unscored": 2,
        "n_keywords_oov": 1,
    }


@pytest.mark.parametrize(
    ("measure", "octis_metric"),
    [("irbo", InvertedRBO()), ("topic_diversity", TopicDiversity())],
)
def test_diversity_matches_octis_for_full_topics(measure, octis_metric):
    topics = {"topics": FULL + [VOCAB[5:15]]}
    assert evaluation.compute_diversity(measure, topics) == pytest.approx(
        octis_metric.score(topics)
    )


def test_topic_diversity_still_penalizes_short_topics():
    topics = {"topics": [VOCAB[0:10], ["w10", "w11"]]}
    assert evaluation.compute_diversity("topic_diversity", topics) == 12 / 20


def test_bertopic_preprocess_matches_bertopic():
    documents = np.array(["Economy.No jobs", "don't\tknow\n", "!!!", "A-B c_d 42"])
    expected = BERTopic()._preprocess_text(documents)
    assert [evaluation.bertopic_preprocess(d) for d in documents] == expected


def test_bertopic_output_is_not_padded():
    model = Mock(spec=BERTopic)
    model.get_topics.return_value = {-1: [], 0: [], 1: []}
    model.get_topic.side_effect = lambda t_id: {
        0: [("war", 0.5), ("iraq", 0.4), ("", 0.1), ("2008", 0.1)],
        1: [(f"w{i}", 0.1) for i in range(12)],
    }[t_id]
    topics = evaluation.bertopic_output_to_octis(model)["topics"]
    assert topics == [["war", "iraq"], [f"w{i}" for i in range(10)]]


def test_bertopic_evaluation_sees_punctuation_stripped_words(monkeypatch):
    model = Mock(spec=BERTopic)
    model.vectorizer_model = CountVectorizer(stop_words="english")
    model.fit_transform.return_value = ([0, 0], None)
    model.get_topics.return_value = {0: []}
    monkeypatch.setattr(
        training.evaluation, "bertopic_output_to_octis", lambda model: {}
    )
    score = Mock(return_value=0.5)
    monkeypatch.setattr(training.evaluation, "compute_coherence", score)
    config = {"experiment": {"coherence_metrics": ["c_npmi"], "diversity_metrics": []}}
    metrics, _ = training.train_and_evaluate(
        model, "m", ["Economy.no today", "Don't know"], np.zeros((2, 4)), config
    )
    assert score.call_args.kwargs["texts"] == [
        ["economyno", "today"],
        ["dont", "know"],
    ]
    assert metrics["evaluation_protocol"] == evaluation.EVALUATION_PROTOCOL


@pytest.mark.parametrize("sample_size", [None, 3])
def test_load_texts_matches_training_inputs(tmp_path, sample_size):
    path = tmp_path / "toy_embeddings.parquet"
    pl.DataFrame(
        {
            "clean_text": ["a b", "  ", "c d", None, "e f", "g h", "i j"],
            "clean_text_embedding": [[0.0, 1.0]] * 7,
            "age": [1, 2, 3, 4, 5, 6, 7],
        }
    ).write_parquet(path)
    config = {
        "experiment": {
            "dataset_path": str(path),
            "text_col": "clean_text",
            "embedding_col": "clean_text_embedding",
            "covariates": {"numerical": ["age"]},
            "sample_size": sample_size,
        }
    }
    text, _, _ = load_and_prep_data(config, random_state=7)
    assert load_texts(config, random_state=7) == text
