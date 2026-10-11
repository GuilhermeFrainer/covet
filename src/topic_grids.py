"""Requested topic-count grids: common to every dataset, or dataset-specific.

Every dataset was first run with the common grid of 10 to 50 requested topics.
Gadarian and ANES were rerun on grids suited to their size: on Gadarian, the
HDBSCAN models find only 11 to 16 topics, so every count above that repeats
the same partition. Results of both grids are kept, and analyses select one:

- "common": 10, 20, 30, 40 and 50 topics on every dataset.
- "dataset": each dataset's own grid (the common grid where none is set).
"""

from __future__ import annotations

import polars as pl

REQUESTED_TOPICS = (10, 20, 30, 40, 50)
DATASET_REQUESTED_TOPICS = {
    "anes": (8, 11, 14, 17, 20),
    "gadarian": (4, 6, 8, 10, 12),
}
TOPIC_GRIDS = {
    "common": "Common (10–50 on every dataset)",
    "dataset": "Dataset-specific (Gadarian 4–12, ANES 8–20)",
}
DEFAULT_TOPIC_GRID = "common"
# Columns holding a run's requested topic count. Newer results record it in
# `requested_topics`; older ones only in the parameter of their model family
# (`nr_topics` for HDBSCAN models, `n_clusters` for clustering models, `k`
# for STM). `n_topics` is left out: it is the realized count for most models.
REQUESTED_TOPIC_COLUMNS = ("requested_topics", "nr_topics", "n_clusters", "k")


def topic_grid(dataset: str, grid: str = DEFAULT_TOPIC_GRID) -> tuple[int, ...]:
    """Requested topic counts of `dataset` under `grid`.

    Raises:
        ValueError: If `grid` is not a key of `TOPIC_GRIDS`.
    """
    if grid == "common":
        return REQUESTED_TOPICS
    if grid == "dataset":
        return DATASET_REQUESTED_TOPICS.get(dataset, REQUESTED_TOPICS)
    raise ValueError(f"Unknown topic grid: {grid}")


def grid_topic_counts(datasets, grid: str = DEFAULT_TOPIC_GRID) -> list[int]:
    """Sorted topic counts on the grid of any of `datasets`."""
    return sorted({count for name in datasets for count in topic_grid(name, grid)})


def requested_topics_expr(columns) -> pl.Expr:
    """A run's requested topic count, from the first of its columns set."""
    present = [c for c in REQUESTED_TOPIC_COLUMNS if c in columns]
    if not present:
        return pl.lit(None, dtype=pl.Int64)
    return pl.coalesce(
        [
            pl.col(c).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
            for c in present
        ]
    )


def in_topic_grid(
    df: pl.DataFrame, grid: str = DEFAULT_TOPIC_GRID, dataset_column="dataset_label"
) -> pl.DataFrame:
    """Drops the runs of datasets with their own grid that are off `grid`.

    Only datasets in `DATASET_REQUESTED_TOPICS` have two grids to choose
    from, so only their rows are filtered. Rows without a requested topic
    count are kept.
    """
    if df.is_empty() or dataset_column not in df.columns:
        return df
    keep = ~pl.col(dataset_column).is_in(list(DATASET_REQUESTED_TOPICS))
    for dataset in DATASET_REQUESTED_TOPICS:
        keep = keep | (
            (pl.col(dataset_column) == dataset)
            & pl.col("__requested").is_in(list(topic_grid(dataset, grid)))
        )
    return (
        df.with_columns(requested_topics_expr(df.columns).alias("__requested"))
        .filter(pl.col("__requested").is_null() | keep)
        .drop("__requested")
    )
