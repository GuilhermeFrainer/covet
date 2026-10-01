import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import polars as pl


@dataclass
class PreparedData:
    """Training inputs and identity captured in the same materialization."""

    text: list[str]
    embeddings: np.ndarray
    metadata: pl.DataFrame
    documents: pl.DataFrame
    provenance: dict
    # Raw (unscaled, unencoded) covariates, row-aligned with text.
    covariates: Optional[pl.DataFrame] = None

    def __iter__(self):
        return iter((self.text, self.embeddings, self.metadata))


def resolve_dataset_path(data_path: str) -> str:
    """Returns the dataset path, falling back to the local Yelp sample file.

    Yelp configs name `yelp_embeddings.parquet`; local checkouts may hold only
    the equivalent `yelp_s10000_embeddings.parquet`.
    """
    from pathlib import Path

    if not Path(data_path).exists() and "yelp_embeddings" in data_path:
        fallback = Path("data/processed/yelp_s10000_embeddings.parquet")
        if fallback.exists():
            logging.getLogger("pipeline").info(
                f"Primary dataset '{data_path}' not found. "
                f"Falling back to '{fallback}'."
            )
            return str(fallback)
    return data_path


def filter_empty_text(lf: pl.LazyFrame, text_col: str) -> pl.LazyFrame:
    """Drops rows whose text is null or only whitespace."""
    return lf.filter(pl.col(text_col).str.strip_chars() != "")


def load_texts(config: dict, random_state: int) -> list[str]:
    """Loads the exact document texts a run trained on, without embeddings.

    Applies the same path resolution, empty-text filter and seeded sampling
    as `load_and_prep_data`.
    """
    experiment_config = config["experiment"]
    text_col = experiment_config.get("text_col", "text")
    lf = filter_empty_text(
        pl.scan_parquet(resolve_dataset_path(experiment_config["dataset_path"])),
        text_col,
    )
    sample_size = experiment_config.get("sample_size")
    if sample_size is not None:
        lf = sample_from_lf(lf, n=sample_size, seed=random_state)
    return lf.select(text_col).collect()[text_col].to_list()


def load_and_prep_data(
    config: dict, random_state: int, *, return_prepared: bool = False
) -> tuple[list[str], np.ndarray, pl.DataFrame] | PreparedData:
    """
    Loads parquet, samples data, and processes metadata.
    """
    logger = logging.getLogger("pipeline")
    experiment_config: dict = config["experiment"]

    data_path = experiment_config["dataset_path"]
    sample_size = experiment_config.get("sample_size")

    covariates_config = experiment_config["covariates"]

    text_col = experiment_config.get("text_col", "text")
    embedding_col = experiment_config.get("embedding_col", "embedding")

    logger.info(f"Target Text Column: '{text_col}'")
    logger.info(f"Target Embedding Column: '{embedding_col}'")

    from pathlib import Path

    data_path = resolve_dataset_path(data_path)

    # Lazy load
    full_lf = pl.scan_parquet(data_path)
    if return_prepared:
        from src.document_assignments import file_checksum

        source_checksum = file_checksum(data_path)
        source_identity_stats = {}
        for column in ("index", "id"):
            if column in full_lf.collect_schema().names():
                unique, nulls, count = (
                    full_lf.select(
                        pl.col(column).n_unique().alias("unique"),
                        pl.col(column).null_count().alias("nulls"),
                        pl.len().alias("count"),
                    )
                    .collect()
                    .row(0)
                )
                source_identity_stats[column] = {
                    "unique_non_null": unique == count and nulls == 0,
                    "distinct_count": unique,
                    "null_count": nulls,
                }
        full_lf = full_lf.with_row_index("source_row_ordinal")

    # Calculate total length before filtering
    total_len = full_lf.select(pl.len()).collect().item()

    # Filter empty rows immediately
    clean_lf = filter_empty_text(full_lf, text_col)

    # Calculate length after filtering
    clean_len = clean_lf.select(pl.len()).collect().item()

    # Explicitly log dropped empty rows
    dropped_empty_rows = total_len - clean_len
    if dropped_empty_rows > 0:
        logger.info(f"Dropped {dropped_empty_rows} rows for being empty strings.")

    # Sample from the CLEAN LazyFrame if requested
    if sample_size is not None:
        logger.info(f"Subsampling dataset to {sample_size} rows.")
        lf = sample_from_lf(clean_lf, n=sample_size, seed=random_state)
    else:
        logger.info(f"Using full dataset. Total rows: {clean_len}")
        lf = clean_lf

    # Identify required columns for selection
    if isinstance(covariates_config, list):
        cov_cols = covariates_config
    else:
        cov_cols = (
            covariates_config.get("numerical", [])
            + covariates_config.get("categorical", [])
            + covariates_config.get("binary", [])
        )

    # Deduplicate required columns
    relevant_cols = list(set([text_col, embedding_col] + cov_cols))
    if return_prepared:
        identity_cols = [
            c
            for c in ("source_row_ordinal", "index", "id")
            if c in full_lf.collect_schema().names()
        ]
        relevant_cols = list(dict.fromkeys(relevant_cols + identity_cols))

    try:
        df = lf.select(relevant_cols).collect()
    except pl.exceptions.ColumnNotFoundError:
        available_cols = full_lf.collect_schema().names()
        logger.error(
            f"Column not found in dataset. Available columns: {available_cols}"
        )
        raise

    logger.info(f"Running experiment on {len(df)} rows.")

    text = df[text_col].to_list()
    embeddings = df[embedding_col].to_numpy()

    processed_metadata = process_metadata(df, covariates_config)

    if return_prepared:
        if file_checksum(data_path) != source_checksum:
            raise ValueError("Source dataset changed while preparing inputs")
        documents = df.select(identity_cols).with_row_index("input_position")
        documents = documents.with_columns(
            pl.col("source_row_ordinal")
            .cast(pl.String)
            .map_elements(
                lambda row: f"{source_checksum}:{row}", return_dtype=pl.String
            )
            .alias("source_document_key")
        )
        numerical = (
            covariates_config
            if isinstance(covariates_config, list)
            else covariates_config.get("numerical", [])
        )
        import hashlib
        import json

        provenance = {
            "dataset_path": str(Path(data_path).resolve()),
            "dataset_sha256": source_checksum,
            "source_key_scheme": "sha256:physical_zero_based_row_ordinal",
            "source_identity_columns": source_identity_stats,
            "source_row_count": total_len,
            "selected_row_count": len(df),
            "text_column": text_col,
            "embedding_column": embedding_col,
            "sample_size": sample_size,
            "sampling_seed": random_state,
            "sampling_replace": False,
            "filter_policy": "selected text column: strip != empty; null excluded",
            "ordered_keys_sha256": hashlib.sha256(
                json.dumps(
                    documents["source_document_key"].to_list(), separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "covariates": covariates_config,
            "raw_covariate_types": {c: str(df.schema[c]) for c in cov_cols},
            "feature_order": processed_metadata.columns,
            "numeric_min_max": {c: [df[c].min(), df[c].max()] for c in numerical},
            "encoding": (
                "src.data.process_metadata: numeric min-max (constant=0, NaN=0), "
                "categorical to_dummies(drop_first=False), binary float; "
                "concatenate in that order"
            ),
            "snapshot_retention": (
                "Retain this exact source parquet once alongside exports; "
                "path/checksum do not reconstruct an overwritten file."
            ),
        }
        return PreparedData(
            text,
            embeddings,
            processed_metadata,
            documents,
            provenance,
            covariates=df.select(cov_cols),
        )

    return text, embeddings, processed_metadata


def process_metadata(
    df: pl.DataFrame, covariates_config: Union[dict, list]
) -> pl.DataFrame:
    """
    Parses the covariates config and applies specific scaling/encoding
    strategies for Numerical, Categorical, and Binary variables, returning
    a Polars DataFrame of processed features.
    """
    logger = logging.getLogger("pipeline")

    # Parse configuration
    # Legacy config
    if isinstance(covariates_config, list):
        logger.warning(
            "Deprecation Warning: 'covariates' is a list. Assuming all are numerical."
        )
        num_cols = covariates_config
        cat_cols = []
        bin_cols = []
    else:
        num_cols = covariates_config.get("numerical", [])
        cat_cols = covariates_config.get("categorical", [])
        bin_cols = covariates_config.get("binary", [])

    processed_features = []

    # Min-max scaling for numerical columns
    if num_cols:
        logger.info(f"Processing Numerical cols: {num_cols}")
        # Check if columns exist
        missing = [c for c in num_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing numerical columns: {missing}")

        num_df = df.select(num_cols)
        # Apply MinMax Scaling safely
        exprs = []
        for c in num_cols:
            c_min = pl.col(c).min()
            c_max = pl.col(c).max()
            # Avoid division by zero if max == min
            exprs.append(
                pl.when(c_max != c_min)
                .then((pl.col(c) - c_min) / (c_max - c_min))
                .otherwise(0.0)
                .alias(c)
            )

        scaled_num_df = num_df.select(exprs).fill_nan(0.0)
        processed_features.append(scaled_num_df)

    # One-hot encoding for categorical values
    if cat_cols:
        logger.info(f"Processing Categorical cols: {cat_cols}")
        missing = [c for c in cat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing categorical columns: {missing}")

        dummies_df = (
            df.select(cat_cols)
            .to_dummies(drop_first=False)
            .select(pl.all().cast(pl.Float64))
        )
        processed_features.append(dummies_df)

    # Binary variables are simply cast to float
    if bin_cols:
        logger.info(f"Processing Binary cols: {bin_cols}")
        missing = [c for c in bin_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing binary columns: {missing}")

        bin_df = df.select(bin_cols).select(pl.all().cast(pl.Float64))
        processed_features.append(bin_df)

    if processed_features:
        final_metadata = pl.concat(processed_features, how="horizontal")
        logger.info(f"Metadata processing complete. Shape: {final_metadata.shape}")
        return final_metadata
    else:
        logger.warning("No covariates found in config. Returning empty DataFrame.")
        return pl.DataFrame()


def sample_from_lf(
    lf: pl.LazyFrame, n: int, seed: Optional[int] = None, replace: bool = False
) -> pl.LazyFrame:
    """
    Samples rows from a LazyFrame.

    This function is designed to be index-agnostic: it creates its own
    temporary contiguous row index for the sampling mathematics, meaning
    it works correctly even if the source data has gaps in its IDs
    or no index column at all.
    """
    rng = np.random.default_rng(seed)

    # Add temporary index for sampling to handle potential gaps in original IDs
    indexed_lf = lf.with_row_index(name="temp_sample_idx")

    # Calculate length of indexed LazyFrame
    lf_len = indexed_lf.select(pl.len()).collect().item()

    if n > lf_len and not replace:
        raise ValueError(
            f"Cannot sample {n} rows without replacement from a dataset of "
            f"{lf_len} rows."
        )

    # Generate sample indices based on the fresh contiguous row index
    sample_idxs = rng.choice(lf_len, size=n, replace=replace)

    # Create a LazyFrame of sampled indices.
    sampled_indices_lf = pl.DataFrame(
        {"temp_sample_idx": sample_idxs}, schema={"temp_sample_idx": pl.UInt32}
    ).lazy()

    # Join back and drop the temporary index
    return sampled_indices_lf.join(indexed_lf, on="temp_sample_idx", how="inner").drop(
        "temp_sample_idx"
    )
