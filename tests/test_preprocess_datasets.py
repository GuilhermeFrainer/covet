# -*- coding: utf-8 -*-
"""Tests for the dataset preprocessing script."""

import nltk
import polars as pl
import pytest
import yaml
from polars.testing import assert_frame_equal
from transformers import AutoTokenizer

# These imports should now work from src.processing
from src.processing import (
    add_log_transformation,
    chunk_text_with_overlap,
    format_as_yaml,
    process_dataset,
    remove_urls,
    split_long_sentence,
    stem_and_remove_stopwords,
)


@pytest.fixture
def sample_trump_df() -> pl.DataFrame:
    """Provides a sample DataFrame mimicking the Trump dataset structure."""
    return pl.DataFrame(
        {
            "text": [
                "A great day at the office! http://example.com #Trump. "
                "What a great day!",
                "Some valid text here",
                "Making America Great Again! And again.",
            ],
            "retweets": [100, 10, 1000],
            "favorites": [500, 20, 2000],
            "isRetweet": ["f", "f", "t"],
            "isDeleted": ["f", "t", "f"],
            "isFlagged": ["t", "f", "f"],
            "date": [
                "2020-01-01 12:00:00",
                "2020-01-02 12:00:00",
                "2020-01-03 12:00:00",
            ],
            "device": ["Twitter for iPhone", "Android", "Twitter for iPhone"],
        }
    )


@pytest.fixture
def sample_yelp_df() -> pl.DataFrame:
    """Provides a sample DataFrame mimicking the Yelp dataset structure."""
    return pl.DataFrame(
        {
            "text": [
                "This place is absolutely amazing and I love it. The food "
                "is great, the service is even better. I will come back "
                "for sure next week.",
                "I would not recommend this restaurant. The service was slow.",
            ],
            "user_review_count": [50, 5],
            "business_review_count": [1000, 200],
            "state": ["CA", "NV"],
        }
    )


def test_remove_urls():
    """Tests the remove_urls function."""
    df = pl.DataFrame(
        {
            "text": [
                "This is a text with a url http://example.com",
                "Another text with https://another.com/path",
                "No url here",
            ]
        }
    )
    result = df.with_columns(clean_text=remove_urls(pl.col("text")))
    expected = pl.Series(
        "clean_text",
        ["This is a text with a url ", "Another text with ", "No url here"],
    )
    assert_frame_equal(result["clean_text"].to_frame(), expected.to_frame())


def test_add_log_transformation():
    """Tests the add_log_transformation function."""
    df = pl.DataFrame({"numbers": [1, 9, 99]})
    result = add_log_transformation(df, "numbers")
    expected = df.with_columns((pl.col("numbers") + 1).log().alias("log_numbers"))
    assert_frame_equal(result, expected)


def test_format_as_yaml():
    """Tests the format_as_yaml function."""
    df = pl.DataFrame(
        {"device": ["Twitter for iPhone", "Android"], "log_retweets": [5.4, 2.1]}
    )
    result = format_as_yaml(df, ["device", "log_retweets"])
    expected_yaml_0 = yaml.dump(
        {"device": "Twitter for iPhone", "log_retweets": 5.4},
        sort_keys=False,
        default_flow_style=False,
    )
    expected_yaml_1 = yaml.dump(
        {"device": "Android", "log_retweets": 2.1},
        sort_keys=False,
        default_flow_style=False,
    )
    expected = pl.Series("", [expected_yaml_0, expected_yaml_1])
    assert all(result == expected)


def test_chunk_text_with_overlap(tmp_path):
    """Tests the chunk_text_with_overlap function."""
    long_text = ". ".join([f"This is sentence number {i}" for i in range(20)])

    df = pl.DataFrame({"id": [1], "text": [long_text], "metadata": ["some_value"]})

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Mocking the tokenizer to have predictable token counts for testing
    original_encode = tokenizer.encode

    def mock_encode(text, add_special_tokens=False):
        return list(range(len(text.split())))

    tokenizer.encode = mock_encode

    result = chunk_text_with_overlap(
        df, "text", tokenizer, max_tokens=30, overlap_sentences=1
    )

    tokenizer.encode = original_encode  # Restore original method

    assert result.height > 1
    assert "token_count" in result.columns
    assert all(result["id"] == 1)
    assert all(result["metadata"] == "some_value")
    assert all(result["token_count"] <= 30)


@pytest.fixture(scope="module")
def minilm_tokenizer():
    """The embedder's tokenizer, unmocked."""
    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def test_split_long_sentence_fits_and_keeps_words(minilm_tokenizer):
    """Pieces fit the limit, end on word boundaries and keep every word."""
    sentence = " ".join(f"inflation{i} expectations" for i in range(200))

    pieces = split_long_sentence(sentence, minilm_tokenizer, max_tokens=50)

    assert len(pieces) > 1
    assert all(count_tokens(minilm_tokenizer, p) <= 50 for p in pieces)
    assert " ".join(pieces).split() == sentence.split()


def test_split_long_sentence_cuts_inside_an_oversized_word(minilm_tokenizer):
    """A single word longer than the limit is still split."""
    word = "".join(f"x{i}" for i in range(300))

    pieces = split_long_sentence(word, minilm_tokenizer, max_tokens=50)

    assert all(count_tokens(minilm_tokenizer, p) <= 50 for p in pieces)
    assert "".join(pieces) == word


def test_chunks_never_exceed_the_limit(minilm_tokenizer):
    """A run-on sentence longer than the limit no longer yields an oversized chunk."""
    run_on = " ".join(f"policy{i} rate" for i in range(400))
    text = f"Short opening. {run_on}. Short closing sentence."
    df = pl.DataFrame({"id": [7], "text": [text]})

    result = chunk_text_with_overlap(
        df, "text", minilm_tokenizer, max_tokens=254, overlap_sentences=2
    )

    assert result.height > 1
    assert all(result["id"] == 7)
    assert all(count_tokens(minilm_tokenizer, t) <= 254 for t in result["text"])
    assert result["token_count"].max() <= 254
    assert "closing" in result["text"][-1]


def test_process_dataset_trump(sample_trump_df, tmp_path):
    """Tests the overall processing for the 'trump' dataset."""

    input_path = tmp_path / "dummy_trump.csv"
    output_path = tmp_path / "trump_processed.parquet"
    sample_trump_df.write_csv(input_path)

    # The function is expected to read the csv and process it
    processed_df = process_dataset(
        dataset_name="trump",
        input_path=str(input_path),
        output_path=str(output_path),
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=20,
        include_metadata=True,
    )

    assert "id" in processed_df.columns
    assert "log_retweets" in processed_df.columns
    assert "log_favorites" in processed_df.columns

    assert "clean_text" in processed_df.columns
    assert "clean_text_lower" in processed_df.columns
    assert "clean_text_lower_punctless" in processed_df.columns

    assert "clean_text_with_metadata" in processed_df.columns

    # Check that YAML frontmatter is present
    assert processed_df["clean_text_with_metadata"][0].startswith("---")

    # Check that chunking was not done (same number of rows)
    assert processed_df.height == sample_trump_df.height


def test_process_dataset_yelp(sample_yelp_df, tmp_path):
    """Tests the overall processing for the 'yelp' dataset."""

    input_path = tmp_path / "dummy_yelp.csv"
    output_path = tmp_path / "yelp_processed.parquet"
    sample_yelp_df.write_csv(input_path)

    processed_df = process_dataset(
        dataset_name="yelp",
        input_path=str(input_path),
        output_path=str(output_path),
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=20,
        include_metadata=True,
    )

    assert "id" in processed_df.columns

    assert "log_user_review_count" in processed_df.columns
    assert "log_business_review_count" in processed_df.columns

    assert "clean_text_with_metadata" in processed_df.columns
    assert processed_df.height > sample_yelp_df.height


def test_process_dataset_deduplication(tmp_path):
    """Tests that deduplication correctly removes repeated clean_text rows."""

    df = pl.DataFrame(
        {
            "text": ["Same text", "Same text", "Different text", "Same text"],
            "date": [
                "2020-01-01 10:00:00",
                "2020-01-01 09:00:00",  # Earlier date, should be kept
                "2020-01-01 11:00:00",
                "2020-01-01 12:00:00",
            ],
            "val": [1, 2, 3, 4],
        }
    )

    input_path = tmp_path / "dummy_dups.parquet"
    output_path = tmp_path / "dups_processed.parquet"
    df.write_parquet(input_path)

    # Process with deduplication
    processed_df = process_dataset(
        dataset_name="test_dataset",  # Using neutral name to avoid trump-specific
        # schema handling
        input_path=str(input_path),
        output_path=str(output_path),
        deduplicate=True,
    )

    # "Same text" and "Same text!" should both clean to "Same text"
    # "Different text" is unique.
    # Total unique should be 2.
    assert processed_df.height == 2

    # Check that it kept the one with the earliest date (val=2)
    # Note: process_dataset sorts by date then takes unique(keep='first')
    assert 2 in processed_df["val"].to_list()
    assert 3 in processed_df["val"].to_list()
    assert 1 not in processed_df["val"].to_list()
    assert 4 not in processed_df["val"].to_list()


def test_process_dataset_drops_empty(tmp_path):
    """Tests that process_dataset drops rows that result in empty clean_text."""
    df = pl.DataFrame(
        {
            "text": [
                "Valid text",
                "http://only-url.com",  # Becomes empty
                "   ",  # Whitespace only
                "12345",  # Numbers only, becomes empty
                "Another valid one",
            ]
        }
    )

    input_path = tmp_path / "dummy_empty.parquet"
    output_path = tmp_path / "empty_processed.parquet"
    df.write_parquet(input_path)

    processed_df = process_dataset(
        dataset_name="test_dataset",
        input_path=str(input_path),
        output_path=str(output_path),
    )

    # Only "Valid text" and "Another valid one" should remain
    assert processed_df.height == 2
    assert "Valid text" in processed_df["clean_text"].to_list()
    assert "Another valid one" in processed_df["clean_text"].to_list()


def test_no_chunk_only_repeats_the_overlap(minilm_tokenizer):
    """A sentence that cannot fit beside the overlap starts a fresh chunk.

    Otherwise the chunker emitted chunks holding nothing but the previous
    chunk's last sentences.
    """
    run_on = " ".join(f"policy{i} rate" for i in range(120))
    text = f"Short one. Short two. {run_on}. Short three. Short four."
    df = pl.DataFrame({"id": [3], "text": [text]})

    result = chunk_text_with_overlap(
        df, "text", minilm_tokenizer, max_tokens=50, overlap_sentences=2
    )

    chunks = result["text"].to_list()
    assert len(chunks) > 2
    for previous, chunk in zip(chunks, chunks[1:]):
        assert not previous.endswith(chunk), chunk
    assert chunks[0].startswith("Short one.") and chunks[-1].endswith("Short four.")
    assert all(count_tokens(minilm_tokenizer, t) <= 50 for t in chunks)


def test_process_dataset_derives_text_columns_per_chunk(sample_yelp_df, tmp_path):
    """Stemmed and lowercased text describe each chunk, not its whole document."""
    input_path = tmp_path / "dummy_yelp.csv"
    sample_yelp_df.write_csv(input_path)

    processed_df = process_dataset(
        dataset_name="yelp",
        input_path=str(input_path),
        output_path=str(tmp_path / "yelp_processed.parquet"),
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=20,
    )

    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    stop_words = set(nltk.corpus.stopwords.words("english"))
    for row in processed_df.iter_rows(named=True):
        assert row["clean_text_stemmed"] == stem_and_remove_stopwords(
            row["clean_text"], stemmer, stop_words
        )
        assert row["clean_text_lower"] == row["clean_text"].lower()
    first = processed_df.filter(pl.col("id") == processed_df["id"][0])
    assert first.height > 1
    assert first["clean_text_stemmed"].n_unique() == first.height
