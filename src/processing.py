# -*- coding: utf-8 -*-
"""Core data-processing functions for Trump and Yelp datasets."""

import logging
import re
import tempfile
from pathlib import Path
from typing import Union

import nltk
import polars as pl
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Download nltk data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Constants
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# SentenceTransformer truncates all-MiniLM-L6-v2 inputs at max_seq_length=256
# tokens, [CLS] and [SEP] included, although the tokenizer reports
# model_max_length=512. Chunks must fit the embedder, not the tokenizer.
EMBEDDING_MAX_SEQ_LENGTH = 256
MAX_CHUNK_TOKENS = EMBEDDING_MAX_SEQ_LENGTH - 2
BATCH_SIZE = 10000  # Process 10,000 rows at a time
ARTIFACTS_TO_REMOVE = {
    "trump": ["covfefe"],
    "yelp": [],  # No specific artifacts for yelp
    "fed": [],
    "anes": [],
    "gadarian": [],
}
NUMERICAL_COLS = {
    "trump": ["retweets", "favorites"],
    "yelp": ["user_review_count", "business_review_count"],
    "fed": [],  # Macro indicators are already rates or can be negative;
    # skipping log transform
    "anes": [],
    "gadarian": [],
}
CATEGORICAL_COLS = {
    "trump": ["device"],
    "yelp": ["state"],
    "fed": ["type", "president", "party", "fed_chair"],
    "anes": ["party_id"],
    "gadarian": ["partisanship"],
}
BOOLEAN_COLS = {
    "trump": ["is_retweet", "is_deleted", "is_flagged"],
    "yelp": [],
    "fed": [],
    "anes": [],
    "gadarian": ["treatment"],
}
DATETIME_COLS = {
    "trump": ["date"],
    "yelp": [],
    "fed": ["date", "release_date"],
    "anes": [],
    "gadarian": [],
}
METADATA_COLS = {
    "trump": ["device", "log_retweets", "log_favorites"],
    "yelp": ["state", "log_user_review_count", "log_business_review_count"],
    "fed": [
        "type",
        "president",
        "party",
        "fed_chair",
        "gdp_monthly",
        "gdp_monthly_lag",
        "gdp_yearly",
        "gdp_yearly_lag",
        "cpi_monthly",
        "cpi_monthly_lag",
        "cpi_yearly",
        "cpi_yearly_lag",
        "funds_rate",
        "funds_rate_lag",
        "unemployment",
        "unemployment_lag",
    ],
    "anes": ["party_id", "years_of_education", "age"],
    "gadarian": ["treatment", "partisanship"],
}

Frame = Union[pl.DataFrame, pl.LazyFrame]


def apply_trump_schema_and_types(df: Frame) -> Frame:
    """Applies Trump-specific schema and type conversions."""
    df = df.with_columns(
        # Convert boolean-like columns ('t'/'f') to actual booleans
        (pl.col("isRetweet").str.to_lowercase() == "t").alias("is_retweet"),
        (pl.col("isDeleted").str.to_lowercase() == "t").alias("is_deleted"),
        (pl.col("isFlagged").str.to_lowercase() == "t").alias("is_flagged"),
        # Convert date column using the specific format
        pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        # Cast device to categorical
        pl.col("device").cast(pl.Categorical),
    ).drop("isRetweet", "isDeleted", "isFlagged")

    # Rename 'content' to 'text' for consistency across datasets
    schema = df.collect_schema()
    if "content" in schema:
        df = df.rename({"content": "text"})

    # Ensure all other columns are snake_case
    df = df.rename({col: col.lower().replace(" ", "_") for col in schema.names()})
    return df


def apply_anes_schema_and_types(df: Frame) -> Frame:
    """Applies ANES-specific schema and type conversions."""
    # Consolidate 'Refused', 'Unknown', and 'Other' into 'Unknown'
    df = df.with_columns(
        pl.col("party_id")
        .replace({"Refused": "Unknown", "Other": "Unknown"})
        .cast(pl.Categorical)
    )
    return df


def stem_and_remove_stopwords(
    text: str,
    stemmer: nltk.stem.snowball.SnowballStemmer,
    stop_words: set[str],
) -> str:
    """Lowercases, removes punctuation, removes stopwords, and stems tokens."""
    if not text:
        return ""
    # Lowercase & strip punctuation [^\w\s]
    clean = re.sub(r"[^\w\s]", "", text.lower())
    words = nltk.word_tokenize(clean)
    stemmed_words = [
        stemmer.stem(w) for w in words if w not in stop_words and w.strip()
    ]
    return " ".join(stemmed_words)


def stem_text(text: str, stemmer: nltk.stem.snowball.SnowballStemmer) -> str:
    """Stems a string of text using the provided NLTK stemmer."""
    if not text:
        return ""
    words = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(w) for w in words])


def remove_urls(text_expr: pl.Expr) -> pl.Expr:
    """Removes URLs and common URL residues from a Polars expression."""
    # Matches http/https, common domain residues like t.co, and cases where
    # punct was removed (httpstco)
    return text_expr.str.replace_all(r"https?://\S+|www\.\S+|httpstco\S+|t\.co/\S+", "")


def remove_numbers(text_expr: pl.Expr) -> pl.Expr:
    """Removes digits from a Polars expression."""
    return text_expr.str.replace_all(r"\d+", "")


def remove_artifacts(text_expr: pl.Expr, artifacts: list[str]) -> pl.Expr:
    """Removes specific artifacts from a Polars expression."""
    if not artifacts:
        return text_expr
    return text_expr.str.replace_all("|".join(artifacts), "")


def add_log_transformation(df: Frame, column: str) -> Frame:
    """Adds a log-transformed column to a Polars DataFrame or LazyFrame."""
    return df.with_columns((pl.col(column) + 1).log().alias(f"log_{column}"))


def format_as_yaml(df: pl.DataFrame, columns: list[str]) -> pl.Series:
    """Formats specified columns of a DataFrame into a YAML frontmatter string."""

    def to_yaml_string(row_dict):
        return yaml.dump(row_dict, sort_keys=False, default_flow_style=False)

    struct_series = df.select(columns).to_struct(name="metadata")
    return struct_series.map_elements(lambda x: to_yaml_string(x), return_dtype=pl.Utf8)


def count_tokens(tokenizer: PreTrainedTokenizer, text: str) -> int:
    """WordPiece tokens in `text`, without the special tokens."""
    # Chunking tokenizes whole documents only to measure them. The tokenizer
    # warns ("Token indices sequence length is longer than the specified
    # maximum ... will result in indexing errors") for any text longer than
    # its model_max_length of 512, BERT's position-embedding limit, because
    # it cannot tell counting from model input. These IDs never reach the
    # model: every chunk is at most MAX_CHUNK_TOKENS, and the embedder
    # truncates its input to 256 tokens anyway. The warning is noise here,
    # so it is turned off.
    return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))


def split_long_sentence(
    sentence: str, tokenizer: PreTrainedTokenizer, max_tokens: int
) -> list[str]:
    """Splits a sentence longer than max_tokens into pieces that fit.

    Pieces end on word boundaries, so each keeps its own tokenization. A
    single word longer than max_tokens is cut inside the word.
    """
    # verbose=False: the sentence may exceed the tokenizer's 512-token limit;
    # see count_tokens for why that warning does not apply here.
    encoding = tokenizer(
        sentence, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )
    offsets = encoding["offset_mapping"]
    word_ids = encoding.word_ids()
    pieces = []
    start = 0
    while start < len(offsets):
        end = min(start + max_tokens, len(offsets))
        if end < len(offsets):
            boundary = end
            while boundary > start + 1 and word_ids[boundary] == word_ids[boundary - 1]:
                boundary -= 1
            if word_ids[boundary] != word_ids[boundary - 1]:
                end = boundary
        pieces.append(sentence[offsets[start][0] : offsets[end - 1][1]])
        start = end
    return pieces


def derived_text_columns(stem: bool = True) -> list[pl.Expr]:
    """Representations derived from `clean_text`, for a document or a chunk.

    `clean_text_lower` lowercases it, `clean_text_lower_punctless` also drops
    punctuation, and with `stem`, `clean_text_stemmed` (Version 2) is the
    lowercased, stopword-free, Snowball-stemmed text.
    """
    lower = pl.col("clean_text").str.to_lowercase()
    columns = [
        lower.alias("clean_text_lower"),
        lower.str.replace_all(r"[^\w\s]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("clean_text_lower_punctless"),
    ]
    if stem:
        stemmer = nltk.stem.snowball.SnowballStemmer("english")
        stop_words = set(nltk.corpus.stopwords.words("english"))
        columns.append(
            pl.col("clean_text")
            .map_elements(
                lambda x: stem_and_remove_stopwords(x, stemmer, stop_words),
                return_dtype=pl.Utf8,
            )
            .alias("clean_text_stemmed")
        )
    return columns


def chunk_text_with_overlap(
    df: pl.DataFrame,
    text_column: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    overlap_sentences: int = 1,
) -> pl.DataFrame:
    """Chunks text into smaller pieces only if it exceeds a token limit.

    Sentences longer than max_tokens are split first, so no chunk exceeds
    the limit. Consecutive chunks share up to overlap_sentences sentences.
    """
    new_rows = []
    rows_to_chunk = 0

    # Use to_dicts() for efficient row iteration
    for row in df.to_dicts():
        original_text = row.get(text_column, "")
        if not original_text:
            new_rows.append(row)
            continue

        total_tokens = count_tokens(tokenizer, original_text)

        if total_tokens <= max_tokens:
            new_row = row.copy()
            new_row["token_count"] = total_tokens
            new_rows.append(new_row)
            continue

        rows_to_chunk += 1
        sentences = []
        sentence_tokens = []
        for sentence in nltk.sent_tokenize(original_text):
            n_tokens = count_tokens(tokenizer, sentence)
            if n_tokens <= max_tokens:
                sentences.append(sentence)
                sentence_tokens.append(n_tokens)
                continue
            for piece in split_long_sentence(sentence, tokenizer, max_tokens):
                sentences.append(piece)
                sentence_tokens.append(count_tokens(tokenizer, piece))
        if not sentences:
            continue

        current_pos = 0
        previous_end = 0
        while current_pos < len(sentences):
            chunk_sentences = []
            chunk_tokens = 0
            end_pos = current_pos

            while end_pos < len(sentences):
                sent_tokens = sentence_tokens[end_pos]

                if chunk_tokens + sent_tokens > max_tokens and chunk_sentences:
                    break

                chunk_sentences.append(sentences[end_pos])
                chunk_tokens += sent_tokens
                end_pos += 1

            if end_pos <= previous_end:
                # The next sentence does not fit beside the overlap, so this
                # chunk would only repeat the previous one. Drop the overlap.
                current_pos = previous_end
                continue
            previous_end = end_pos

            new_row = row.copy()
            new_row[text_column] = " ".join(chunk_sentences)
            new_row["token_count"] = chunk_tokens
            new_rows.append(new_row)

            if end_pos >= len(sentences):
                break

            step = max(1, len(chunk_sentences) - overlap_sentences)
            current_pos += step

    if rows_to_chunk > 0:
        logging.info(f"Chunked {rows_to_chunk} rows within the batch.")

    if not new_rows:
        return df.with_columns(pl.lit(0, dtype=pl.Int32).alias("token_count"))

    return pl.from_dicts(new_rows, schema={**df.schema, "token_count": pl.Int32})


def process_dataset(
    dataset_name: str,
    input_path: str,
    output_path: str,
    tokenizer_name: str = TOKENIZER_NAME,
    max_tokens: int | None = None,
    include_metadata: bool = False,
    deduplicate: bool = False,
    stem: bool = True,
) -> pl.DataFrame:
    """Main function to process a single dataset using lazy evaluation and batching.

    Returns:
        The processed DataFrame (collected from parquet for verification/testing).
    """
    logging.info(f"Starting preprocessing for dataset: {dataset_name}")

    logging.info(f"Scanning data from {input_path}")
    if input_path.endswith("csv"):
        lf = pl.scan_csv(input_path)
    else:
        lf = pl.scan_parquet(input_path)

    if dataset_name == "trump":
        lf = apply_trump_schema_and_types(lf)
    elif dataset_name == "anes":
        lf = apply_anes_schema_and_types(lf)
    else:
        lf = lf.rename(
            {col: col.lower().replace(" ", "_") for col in lf.collect_schema().names()}
        )

    for col in NUMERICAL_COLS.get(dataset_name, []):
        lf = add_log_transformation(lf, col)

    lf = lf.drop(["index", "id"], strict=False)

    logging.info("Applying lazy text preprocessing...")
    # clean_text: Version 1 (Lightest preprocessing - URLs, numbers, artifacts removed;
    # casing and punctuation preserved; excess whitespace collapsed)
    lf = (
        lf.with_columns(clean_text=remove_numbers(remove_urls(pl.col("text"))))
        .with_columns(
            clean_text=remove_artifacts(
                pl.col("clean_text"), ARTIFACTS_TO_REMOVE.get(dataset_name, [])
            )
        )
        .with_columns(
            clean_text=pl.col("clean_text")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
    )

    # Document-level versions, used only to drop documents empty in any
    # representation; every chunk gets its own versions after chunking.
    if stem:
        logging.info("Adding stemmed and stopword-removed text column (Version 2)...")
    lf = lf.with_columns(derived_text_columns(stem))

    # Filter out empty or whitespace-only rows in clean_text and clean_text_stemmed
    initial_row_count = lf.select(pl.len()).collect().item()

    if stem or "clean_text_stemmed" in lf.collect_schema().names():
        empty_clean = (
            lf.filter(pl.col("clean_text").str.strip_chars() == "")
            .select(pl.len())
            .collect()
            .item()
        )
        empty_stemmed = (
            lf.filter(pl.col("clean_text_stemmed").str.strip_chars() == "")
            .select(pl.len())
            .collect()
            .item()
        )

        asymmetric_empty = (
            lf.filter(
                (pl.col("clean_text").str.strip_chars() == "")
                ^ (pl.col("clean_text_stemmed").str.strip_chars() == "")
            )
            .select(pl.len())
            .collect()
            .item()
        )

        if asymmetric_empty > 0:
            logging.warning(
                f"WARNING: {asymmetric_empty} rows were non-empty in one column "
                f"but empty in another (clean_text empty: {empty_clean}, "
                f"clean_text_stemmed empty: {empty_stemmed}). Dropping these "
                "rows from both columns to maintain strict row alignment across "
                "representations."
            )

        lf = lf.filter(
            (pl.col("clean_text").str.strip_chars() != "")
            & (pl.col("clean_text_stemmed").str.strip_chars() != "")
        )
    else:
        lf = lf.filter(pl.col("clean_text").str.strip_chars() != "")

    after_empty_filter_count = lf.select(pl.len()).collect().item()
    dropped_empty = initial_row_count - after_empty_filter_count
    if dropped_empty > 0:
        logging.info(
            f"Dropped {dropped_empty} rows where clean_text or clean_text_stemmed "
            "was empty or whitespace-only."
        )

    if deduplicate:
        logging.info("Deduplicating based on 'clean_text'...")
        # We need to collect partially to count and deduplicate efficiently
        # if it's not a huge dataset, or we can do it lazily. Polars
        # 'unique' on LazyFrame works but we'll collect once to log.
        initial_count = lf.select(pl.len()).collect().item()

        # Sort by date if it exists to keep the first occurrence chronologically
        if "date" in lf.collect_schema().names():
            lf = lf.sort("date").unique(
                subset=["clean_text"], keep="first", maintain_order=True
            )
        else:
            lf = lf.unique(subset=["clean_text"], keep="first")

        final_count = lf.select(pl.len()).collect().item()
        dropped_count = initial_count - final_count
        logging.info(f"Deduplication complete. Dropped {dropped_count} duplicate rows.")

    logging.info("Starting batch processing for chunking and YAML injection...")

    total_rows = lf.select(pl.len()).collect().item()
    num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if max_tokens is None:
        max_tokens = MAX_CHUNK_TOKENS
    elif max_tokens > MAX_CHUNK_TOKENS:
        logging.warning(
            f"max_tokens={max_tokens} exceeds the {MAX_CHUNK_TOKENS} content tokens "
            "the embedder reads; longer chunks are truncated when embedded."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_dir = Path(tmpdir)

        process_bar = tqdm(
            range(num_batches), desc="Preprocessing and Chunking Batches"
        )
        for i in process_bar:
            batch_df = lf.slice(i * BATCH_SIZE, BATCH_SIZE).collect()

            original_rows = batch_df.height
            batch_df = batch_df.with_columns(
                pl.arange(i * BATCH_SIZE, i * BATCH_SIZE + original_rows).alias("id")
            )

            chunked_df = chunk_text_with_overlap(
                batch_df,
                "clean_text",
                tokenizer,
                max_tokens=max_tokens,
                overlap_sentences=2,
            ).with_columns(derived_text_columns(stem))
            if stem:
                # A chunk of stop words alone has no stemmed text; drop it from
                # every representation to keep the rows aligned.
                empty = chunked_df["clean_text_stemmed"].str.strip_chars() == ""
                if empty.any():
                    logging.info(
                        f"Dropped {empty.sum()} chunks with no text after stemming."
                    )
                    chunked_df = chunked_df.filter(~empty)

            process_bar.set_postfix_str(
                f"Original: {original_rows}, Chunked: {chunked_df.height}"
            )

            if include_metadata:
                yaml_frontmatter = format_as_yaml(
                    chunked_df, METADATA_COLS[dataset_name]
                )

                final_batch_df = chunked_df.with_columns(
                    clean_text_with_metadata=pl.concat_str(
                        [
                            pl.lit("---\n"),
                            yaml_frontmatter,
                            pl.lit("---\n"),
                            pl.col("clean_text"),
                        ]
                    ),
                    clean_text_lower_with_metadata=pl.concat_str(
                        [
                            pl.lit("---\n"),
                            yaml_frontmatter,
                            pl.lit("---\n"),
                            pl.col("clean_text_lower"),
                        ]
                    ),
                    clean_text_lower_punctless_with_metadata=pl.concat_str(
                        [
                            pl.lit("---\n"),
                            yaml_frontmatter,
                            pl.lit("---\n"),
                            pl.col("clean_text_lower_punctless"),
                        ]
                    ),
                )
            else:
                final_batch_df = chunked_df

            final_batch_df.write_parquet(batch_dir / f"batch_{i}.parquet")

        logging.info(f"Stitching batches and saving to {output_path}")
        batch_files = sorted(
            batch_dir.glob("*.parquet"), key=lambda p: int(p.stem.split("_")[-1])
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Adds index back at the end
        pl.scan_parquet(batch_files).with_row_index().sink_parquet(output_path)

    logging.info("Preprocessing finished successfully.")
    return pl.read_parquet(output_path)
