"""Topic-quality metrics: coherence and diversity over each topic's top words.

Topics are scored on the words they actually have. Topics with fewer than
`topk` words are not padded, since repeated words form self-pairs that
receive the maximum NPMI and UMass. Topics with fewer than two distinct
in-vocabulary words have no word pairs and are excluded from the coherence
averages; `topic_diagnostics` reports how many were excluded.
"""

import itertools
import logging
import re
from typing import Any

import numpy as np
from bertopic import BERTopic
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from octis.evaluation_metrics.diversity_metrics import get_word2index, rbo
from sklearn.feature_extraction.text import CountVectorizer

# Recorded with every result so rows scored under different rules are never
# pooled. Rows without it were scored with padded topics and raw-text
# tokenization for BERTopic variants.
EVALUATION_PROTOCOL = "unpadded_2026_10"

# Inverted RBO weight, matching OCTIS's InvertedRBO default.
RBO_WEIGHT = 0.9


def topic_words_to_octis(topic_words: list[list[str]]) -> dict[str, list[list[str]]]:
    """
    Standardizes a list of topic words into OCTIS format.
    """
    return {"topics": topic_words}


# Candidate words per topic before numeric words are dropped. Matches the
# `top_n_words` default that src.models gives BERTopic.
TOP_N_CANDIDATE_WORDS = 50


def get_top_words_from_beta(
    beta: np.ndarray, vocab: list[str], topk: int = 10
) -> list[list[str]]:
    """Extracts each topic's top words from a beta matrix (K x V).

    Words are ranked by probability and filtered with `select_topic_words`,
    exactly as BERTopic topic words are, so numeric words never count.
    """
    topic_words = []
    for topic_beta in beta:
        # Assuming beta is either probabilities or log-probabilities
        top_indices = np.argsort(topic_beta)[::-1][:TOP_N_CANDIDATE_WORDS]
        topic_words.append(select_topic_words([vocab[i] for i in top_indices], topk))
    return topic_words


def representation_vectorizer(remove_stop_words: bool = True) -> CountVectorizer:
    """Builds the vectorizer behind every topic-word vocabulary.

    BERTopic variants use it for c-TF-IDF, and the STM bag-of-words is built
    with its analyzer, so both model families draw topic words from the same
    tokens.
    """
    return CountVectorizer(stop_words="english" if remove_stop_words else None)


def representation_tokens(texts: list[str], analyzer=None) -> list[list[str]]:
    """Tokenizes documents the way topic words are extracted.

    Applies BERTopic's c-TF-IDF preprocessing and then the analyzer, which
    defaults to `representation_vectorizer()`'s. Documents left without
    tokens come back as empty lists, aligned with `texts`.
    """
    if analyzer is None:
        analyzer = representation_vectorizer().build_analyzer()
    return [analyzer(bertopic_preprocess(text)) for text in texts]


def bertopic_preprocess(text: str) -> str:
    """Applies BERTopic's c-TF-IDF preprocessing for English to one document.

    Mirrors `BERTopic._preprocess_text`: tabs and newlines become spaces and
    every other non-alphanumeric character is deleted, so "economy.no" becomes
    "economyno". Topic words come from this text, so coherence must tokenize
    it the same way.
    """
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    return text if text != "" else "emptydoc"


def select_topic_words(words: list[Any], topk: int = 10) -> list[str]:
    """Returns a topic's first `topk` non-empty, non-numeric words, unpadded."""
    cleaned = [str(word).strip() for word in words]
    return [word for word in cleaned if word != "" and not word.isdigit()][:topk]


def bertopic_output_to_octis(m: BERTopic, topk: int = 10) -> dict[str, list[list[str]]]:
    """
    Reshapes BERTopic output so that it can be readily passed to OCTIS
    for evaluation.
    """
    topic_words: list[list[str]] = []
    for t_id in m.get_topics().keys():
        if t_id == -1:  # Ignores noise topic
            continue
        topic_info = m.get_topic(t_id)  # type: ignore
        if isinstance(topic_info, list):
            topic_words.append(select_topic_words([w for w, _ in topic_info], topk))
    return {"topics": topic_words}


def tritopic_output_to_octis(m: Any, topk: int = 10) -> dict[str, list[list[str]]]:
    """
    Reshapes TriTopic output so that it can be readily passed to OCTIS
    for evaluation.
    """
    topic_words: list[list[str]] = []
    if hasattr(m, "topics_"):
        for t in m.topics_:
            if getattr(t, "topic_id", None) == -1:
                continue
            words = select_topic_words(getattr(t, "keywords", []), topk)
            if words:
                topic_words.append(words)
    return {"topics": topic_words}


def _scorable_topics(
    topics: list[list[str]], dictionary: Dictionary, topk: int
) -> list[list[str]]:
    """Keeps each topic's distinct in-vocabulary words; drops topics with < 2."""
    scorable = []
    for topic in topics:
        known = [w for w in dict.fromkeys(topic[:topk]) if w in dictionary.token2id]
        if len(known) >= 2:
            scorable.append(known)
    return scorable


def topic_diagnostics(
    model_output: dict,
    texts: list[list[str]] | None = None,
    topk: int = 10,
    dictionary: Dictionary | None = None,
) -> dict[str, int]:
    """Counts the topics that coherence scores only partially or not at all.

    Returns:
        n_topics_short: topics with fewer than `topk` words.
        n_topics_unscored: topics with fewer than two distinct in-vocabulary
            words, which are excluded from the coherence averages.
        n_keywords_oov: distinct topic words absent from the evaluation corpus.
    """
    if dictionary is None:
        dictionary = Dictionary(texts)
    topics = model_output.get("topics") or []
    keywords = {w for topic in topics for w in topic[:topk]}
    return {
        "n_topics_short": sum(len(topic) < topk for topic in topics),
        "n_topics_unscored": len(topics)
        - len(_scorable_topics(topics, dictionary, topk)),
        "n_keywords_oov": len(keywords - set(dictionary.token2id)),
    }


def compute_coherence(
    model_output: dict,
    texts: list[list[str]],
    measure: str = "c_npmi",
    topk: int = 10,
    dictionary: Dictionary | None = None,
) -> float:
    """Mean per-topic coherence over the topics that have word pairs.

    Equivalent to OCTIS's Coherence for topics with `topk` in-vocabulary
    words. Shorter topics are scored on their own distinct words; topics with
    fewer than two are excluded. Returns NaN when no topic can be scored.

    Args:
        model_output: Dictionary with a "topics" list of word lists.
        texts: Tokenized evaluation corpus.
        measure: Gensim coherence measure ("c_npmi", "c_v", "u_mass", ...).
        topk: Number of top words per topic to consider.
        dictionary: Optional prebuilt gensim Dictionary of `texts`, for reuse
            across measures.
    """
    logger = logging.getLogger("pipeline")

    if not model_output.get("topics"):
        logger.warning(f"No topics found for evaluation of {measure}. Returning 0.0")
        return 0.0

    if dictionary is None:
        dictionary = Dictionary(texts)
    topics = model_output["topics"]
    scorable = _scorable_topics(topics, dictionary, topk)
    if len(scorable) < len(topics):
        logger.warning(
            f"Excluded {len(topics) - len(scorable)} of {len(topics)} topics from "
            f"{measure}: fewer than two distinct in-vocabulary words."
        )
    if not scorable:
        return float("nan")

    per_topic = CoherenceModel(
        topics=scorable,
        texts=texts,
        dictionary=dictionary,
        coherence=measure,
        processes=1,
        topn=topk,
    ).get_coherence_per_topic()
    return float(np.mean(per_topic))


def compute_diversity(diversity_type: str, model_output: dict, topk: int = 10) -> float:
    """Topic Diversity or Inverted RBO over each topic's top words.

    Same formulas as OCTIS, without its requirement that the first topic have
    `topk` words. Topic Diversity keeps the K * topk denominator, so short
    topics lower it.
    """
    topics = [topic[:topk] for topic in model_output.get("topics") or []]
    if diversity_type == "topic_diversity":
        if not topics:
            return 0.0
        unique_words = set(itertools.chain.from_iterable(topics))
        return len(unique_words) / (topk * len(topics))
    if diversity_type == "irbo":
        overlaps = []
        for list1, list2 in itertools.combinations(topics, 2):
            word2index = get_word2index(list1, list2)
            overlaps.append(
                rbo(
                    [word2index[w] for w in list1],
                    [word2index[w] for w in list2],
                    p=RBO_WEIGHT,
                )[2]
            )
        return float(1 - np.mean(overlaps)) if overlaps else float("nan")
    raise ValueError(f"Invalid diversity type: {diversity_type}")
