"""Verify coherence receives tokens from the appropriate keyword analyzer."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from src import evaluation, training
from src.models import create_topic_model_instance

TEXTS = ["The Federal Reserve, acts!", "BANK policy.", "the and"]
CONFIG = {
    "experiment": {"coherence_metrics": ["u_mass", "c_npmi"], "diversity_metrics": []}
}


def evaluate(model, monkeypatch):
    score = Mock(return_value=0.5)
    monkeypatch.setattr(training.evaluation, "compute_coherence", score)
    training.train_and_evaluate(
        model, "tokenization_test", TEXTS, np.zeros((3, 4)), CONFIG
    )
    assert score.call_count == 2
    return [call.kwargs["texts"] for call in score.call_args_list]


@pytest.mark.parametrize("model_type", ["tritopic", "fast_tritopic"])
@pytest.mark.parametrize("ngram_range", [[1, 1], [1, 2]])
def test_tritopic_uses_keyword_vectorizer_after_fit(
    model_type, ngram_range, monkeypatch
):
    model = create_topic_model_instance(
        {"type": model_type, "params": {"keyword_ngram_range": ngram_range}},
        None,
        random_state=42,
    )
    # A competing attribute must not override TriTopic's keyword analyzer.
    model.vectorizer_model = Mock()
    model.vectorizer_model.build_analyzer.side_effect = AssertionError("wrong analyzer")

    def fit(**kwargs):
        vectorizer = CountVectorizer(
            stop_words="english", ngram_range=model._keyword_extractor.ngram_range
        ).fit(kwargs["documents"])
        model._keyword_extractor._vectorizer = vectorizer
        model.labels_ = np.zeros(3, dtype=int)
        model.topics_ = [SimpleNamespace(topic_id=0, keywords=["federal", "reserve"])]

    monkeypatch.setattr(model, "fit", fit)
    tokens = evaluate(model, monkeypatch)
    expected = [["federal", "reserve", "acts"], ["bank", "policy"]]
    if ngram_range == [1, 2]:
        expected = [
            ["federal", "reserve", "acts", "federal reserve", "reserve acts"],
            ["bank", "policy", "bank policy"],
        ]
    assert tokens == [expected, expected]


def test_bertopic_keeps_its_vectorizer_analyzer(monkeypatch):
    model = Mock(spec=BERTopic)
    model.vectorizer_model = CountVectorizer(lowercase=False, ngram_range=(1, 2))
    model._keyword_extractor = Mock()
    model._keyword_extractor._vectorizer.build_analyzer.side_effect = AssertionError(
        "BERTopic must not use the TriTopic analyzer"
    )
    model.fit_transform.return_value = ([0, 0, 0], None)
    model.get_topics.return_value = {0: []}
    monkeypatch.setattr(
        training.evaluation, "bertopic_output_to_octis", lambda model: {}
    )
    expected = [
        [
            "The",
            "Federal",
            "Reserve",
            "acts",
            "The Federal",
            "Federal Reserve",
            "Reserve acts",
        ],
        ["BANK", "policy", "BANK policy"],
        ["the", "and", "the and"],
    ]
    assert evaluate(model, monkeypatch) == [expected, expected]


@pytest.mark.parametrize("model_type", ["tritopic", "fast_tritopic"])
def test_tritopic_without_vectorizer_warns_and_continues(
    model_type, monkeypatch, caplog
):
    model = create_topic_model_instance({"type": model_type}, None, random_state=42)
    monkeypatch.setattr(model, "fit", lambda **kwargs: None)
    model.labels_ = []
    model.topics_ = []
    monkeypatch.setattr(logging.getLogger("pipeline"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="pipeline"):
        tokens = evaluate(model, monkeypatch)
    expected = [text.lower().split() for text in TEXTS]
    assert tokens == [expected, expected]
    assert "no vectorizer analyzer" in caplog.text
    assert "may not match the topic keywords" in caplog.text


def test_representation_tokens_match_default_bertopic_vectorizer():
    texts = ["The economy.no grew, and it's 2020!", "the and of"]
    analyzer = evaluation.representation_vectorizer().build_analyzer()
    expected = [analyzer(evaluation.bertopic_preprocess(t)) for t in texts]

    tokens = evaluation.representation_tokens(texts)

    assert tokens == expected
    assert tokens[0] == ["economyno", "grew", "2020"]
    # A document of stopwords has no tokens and stays aligned with its text.
    assert tokens[1] == []


def test_beta_top_words_skip_numbers_like_bertopic_topics():
    vocab = ["2020", "tax", "jobs", "1999", "rate"]
    beta = np.array([[0.4, 0.3, 0.1, 0.15, 0.05]])

    assert evaluation.get_top_words_from_beta(beta, vocab, topk=2) == [["tax", "jobs"]]
