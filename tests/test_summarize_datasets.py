"""Tests for the dataset summary table."""

import polars as pl
import pytest

from scripts.data_prep.summarize_datasets import (
    build_table,
    count_covariates,
    count_sentences,
    summarize_dataset,
    to_latex,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("   ", 0),
        ("the economy", 1),
        ("Jobs. Health care!", 2),
        ("The rate rose 2.5 percent.", 1),
    ],
)
def test_count_sentences(text, expected):
    """Unpunctuated answers count as one sentence; decimals do not split."""
    assert count_sentences(text) == expected


@pytest.fixture
def chunked_embeddings(tmp_path):
    """Document 0 is split into two overlapping chunks that repeat its text."""
    path = tmp_path / "toy_embeddings.parquet"
    long_text = "One two three. Four five six. Seven eight nine."
    pl.DataFrame(
        {
            "id": [0, 0, 1],
            "text": [long_text, long_text, "the economy"],
            "clean_text": [
                "One two three. Four five six.",
                "Four five six. Seven eight nine.",
                "the economy",
            ],
            "token_count": [8, 9, 2],
        }
    ).write_parquet(path)
    return path


def test_summarize_dataset_counts_each_document_once(chunked_embeddings):
    """Words and sentences ignore chunk overlap; chunk stats use token_count."""
    stats = summarize_dataset(chunked_embeddings, raw_docs=3, covariates=4)

    assert stats["raw_docs"] == 3
    assert stats["documents"] == 2
    assert stats["dropped_docs"] == 1
    assert stats["covariates"] == 4
    assert stats["words"] == 11
    assert stats["sentences"] == 4
    assert stats["words_per_doc"] == pytest.approx(5.5)
    assert stats["chunks"] == 3
    assert stats["tokens_per_chunk"] == pytest.approx(19 / 3)
    assert stats["max_tokens_per_chunk"] == 9


def test_sample_without_raw_source_drops_nothing(chunked_embeddings):
    """A sample of preprocessed documents reports no dropped documents."""
    stats = summarize_dataset(chunked_embeddings, raw_docs=None, covariates=0)

    assert stats["raw_docs"] == stats["documents"] == 2
    assert stats["dropped_docs"] == 0


def test_count_covariates_maps_sample_to_base_config(tmp_path):
    """yelp_s10000 reads the covariates of the yelp config."""
    (tmp_path / "yelp.yaml").write_text(
        "covariates:\n  numerical: [stars, date]\n  categorical: [state]\n",
        encoding="utf-8",
    )

    assert count_covariates("yelp_s10000", config_dir=tmp_path) == 3


def test_to_latex_labels_trump_sample_and_groups_sections(chunked_embeddings):
    """The Trump sample is labeled "Trump" and sections get their own rows."""
    stats = summarize_dataset(chunked_embeddings, raw_docs=None, covariates=7)
    latex = to_latex(build_table({"trump_s25000": stats}), ["Note."])

    assert r"\label{tab:dataset_summary}" in latex and r"\caption{" in latex
    # Full width: five dataset columns do not fit one ACL column.
    assert latex.startswith(r"\begin{table*}") and r"\end{table*}" in latex
    assert r"\small" in latex
    assert r"\fontsize" not in latex
    assert r" & Trump \\" in latex
    assert "trump" not in latex
    assert r"\textit{Documents}" in latex
    assert r"\textit{Chunks (model input)}" in latex
    assert r"Words & 11 \\" in latex
    assert "Note." in latex
