import logging
import time
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from gensim.corpora.dictionary import Dictionary
from tritopic import TriTopic

import src.evaluation as evaluation


def train_and_evaluate(
    topic_model: Any,
    model_id: str,
    text: list[str],
    embeddings: np.ndarray,
    config: dict,
    scaled_metadata: Optional[Union[pl.DataFrame, pd.DataFrame, np.ndarray]] = None,
    after_fit=None,
) -> tuple[dict, Any]:
    """
    Fits a pre-instantiated topic model (BERTopic or TriTopic) and calculates
    evaluation metrics.

    Args:
        topic_model: The instantiated BERTopic or TriTopic object.
        model_id: String identifier for logging/results.
        text: List of document strings.
        embeddings: Pre-computed document embeddings.
        config: The global experiment configuration (for metric settings).
        scaled_metadata: Optional pre-processed metadata (Polars DataFrame,
            Pandas DataFrame, or NumPy array) for multi-view / tritopic models.

    Returns:
        A tuple containing the metrics dictionary and the fitted model.
    """
    logger = logging.getLogger("pipeline")

    start_time = time.time()
    is_tritopic = isinstance(topic_model, TriTopic)

    if is_tritopic:
        # TriTopic expects metadata as a pandas DataFrame with columns
        metadata_df = None
        if scaled_metadata is not None:
            if isinstance(scaled_metadata, pl.DataFrame):
                if scaled_metadata.width > 0 and scaled_metadata.height > 0:
                    metadata_df = scaled_metadata.to_pandas()
            elif hasattr(scaled_metadata, "to_pandas"):
                metadata_df = scaled_metadata.to_pandas()
            elif isinstance(scaled_metadata, pd.DataFrame):
                if not scaled_metadata.empty:
                    metadata_df = scaled_metadata
            elif isinstance(scaled_metadata, np.ndarray):
                if scaled_metadata.size > 0:
                    if scaled_metadata.ndim == 1:
                        metadata_df = pd.DataFrame(
                            scaled_metadata.reshape(-1, 1), columns=["meta_0"]
                        )
                    else:
                        metadata_df = pd.DataFrame(
                            scaled_metadata,
                            columns=[
                                f"meta_{i}" for i in range(scaled_metadata.shape[1])
                            ],
                        )

        if metadata_df is not None:
            topic_model.fit(documents=text, embeddings=embeddings, metadata=metadata_df)
        else:
            topic_model.fit(documents=text, embeddings=embeddings)

        labels = getattr(topic_model, "labels_", [])
        outlier_count = int((np.array(labels) == -1).sum()) if len(labels) > 0 else 0
        n_topics = len([t for t in topic_model.topics_ if t.topic_id != -1])
    else:
        topics, _ = topic_model.fit_transform(documents=text, embeddings=embeddings)
        outlier_count = topics.count(-1) if -1 in topics else 0
        n_topics = len([t for t in topic_model.get_topics() if t != -1])

    duration = time.time() - start_time
    logger.info(f"[{model_id}] Training finished in {duration:.2f} seconds.")

    if after_fit is not None:
        after_fit(topic_model, is_tritopic=is_tritopic)

    # Build tokenized texts for OCTIS metrics
    if is_tritopic:
        # FastTriTopic inherits this extractor. Use the vectorizer fitted during
        # keyword extraction so punctuation, stopwords and n-grams match.
        extractor = getattr(topic_model, "_keyword_extractor", None)
        vectorizer = getattr(extractor, "_vectorizer", None)
        if callable(getattr(vectorizer, "build_analyzer", None)):
            analyzer = vectorizer.build_analyzer()
            tokenized_texts = [analyzer(t) for t in text]
        else:
            # Some extraction methods (e.g. KeyBERT) have no fitted vectorizer.
            if config["experiment"]["coherence_metrics"]:
                logger.warning(
                    "[%s] TriTopic keyword extractor has no vectorizer analyzer; "
                    "coherence evaluation is falling back to lowercase whitespace "
                    "tokenization, which may not match the topic keywords.",
                    model_id,
                )
            tokenized_texts = [t.lower().split() for t in text]
    elif hasattr(topic_model, "vectorizer_model") and hasattr(
        topic_model.vectorizer_model, "build_analyzer"
    ):
        # BERTopic strips punctuation before fitting c-TF-IDF, so its topic
        # words (e.g. "economyno") exist only in the preprocessed text.
        analyzer = topic_model.vectorizer_model.build_analyzer()
        tokenized_texts = [analyzer(evaluation.bertopic_preprocess(t)) for t in text]
    else:
        tokenized_texts = [t.lower().split() for t in text]

    tokenized_texts = [t for t in tokenized_texts if len(t) > 0]

    if is_tritopic:
        octis_output = evaluation.tritopic_output_to_octis(topic_model)
    else:
        octis_output = evaluation.bertopic_output_to_octis(topic_model)

    dictionary = Dictionary(tokenized_texts)
    metrics = {
        "model_name": model_id,
        "duration_seconds": duration,
        "n_topics": n_topics,
        "outliers": outlier_count,
        **evaluation.topic_diagnostics(octis_output, dictionary=dictionary),
        "evaluation_protocol": evaluation.EVALUATION_PROTOCOL,
    }

    # Coherence Loop
    for cm in config["experiment"]["coherence_metrics"]:
        metrics[cm] = evaluation.compute_coherence(
            model_output=octis_output,
            texts=tokenized_texts,
            measure=cm,
            dictionary=dictionary,
        )

    # Diversity Loop
    for dm in config["experiment"]["diversity_metrics"]:
        metrics[dm] = evaluation.compute_diversity(dm, model_output=octis_output)

    for metric in (
        config["experiment"]["coherence_metrics"]
        + config["experiment"]["diversity_metrics"]
    ):
        if not np.isfinite(metrics[metric]):
            logger.warning(
                "[%s] Evaluation metric %s returned %s after successful training. "
                "Keeping the score and continuing; this run has incomplete metrics.",
                model_id,
                metric,
                metrics[metric],
            )
    if any(
        not np.isfinite(metrics[metric])
        for metric in config["experiment"]["coherence_metrics"]
    ):
        vocabulary = {word for document in tokenized_texts for word in document}
        topics = octis_output.get("topics", [])
        missing = sorted({word for topic in topics for word in topic} - vocabulary)
        short_topics = [
            index
            for index, topic in enumerate(topics)
            if len(set(topic) & vocabulary) < 2
        ]
        logger.warning(
            "[%s] Coherence diagnostics: %d tokenized documents; "
            "%d keywords absent from the evaluation vocabulary (examples: %s); "
            "%d topics with fewer than two distinct in-vocabulary keywords "
            "(zero-based evaluation topic indices: %s). "
            "Check keyword/tokenizer alignment, including multiword phrases.",
            model_id,
            len(tokenized_texts),
            len(missing),
            missing[:10],
            len(short_topics),
            short_topics[:20],
        )

    return metrics, topic_model
