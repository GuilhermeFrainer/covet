"""Common and dataset-specific requested topic-count grids."""

import polars as pl
import pytest

from src.topic_grids import (
    REQUESTED_TOPICS,
    grid_topic_counts,
    in_topic_grid,
    requested_topics_expr,
    topic_grid,
)


def test_topic_grid_per_dataset():
    assert topic_grid("gadarian") == REQUESTED_TOPICS
    assert topic_grid("gadarian", "dataset") == (4, 6, 8, 10, 12)
    assert topic_grid("anes", "dataset") == (8, 11, 14, 17, 20)
    assert topic_grid("fed", "dataset") == REQUESTED_TOPICS
    with pytest.raises(ValueError):
        topic_grid("fed", "other")


def test_grid_topic_counts_is_the_sorted_union():
    assert grid_topic_counts(["fed", "gadarian"]) == list(REQUESTED_TOPICS)
    union = grid_topic_counts(["fed", "gadarian"], "dataset")
    assert union == [4, 6, 8, 10, 12, 20, 30, 40, 50]


def test_requested_topics_reads_the_first_set_column():
    df = pl.DataFrame(
        {
            "requested_topics": [None, None, None, 7],
            "nr_topics": ["10", None, None, None],
            "n_clusters": [None, 20.0, None, 3.0],
            "k": [None, None, 30, None],
        }
    )
    requested = df.with_columns(requested_topics_expr(df.columns).alias("requested"))
    assert requested["requested"].to_list() == [10, 20, 30, 7]
    unset = df.with_columns(requested_topics_expr(["other"]).alias("requested"))
    assert unset["requested"].to_list() == [None] * 4


def test_in_topic_grid_filters_only_datasets_with_their_own_grid():
    df = pl.DataFrame(
        {
            "dataset_label": ["gadarian"] * 4 + ["fed"] * 2,
            "nr_topics": [4, 10, 20, None, 10, 15],
        }
    )
    common = in_topic_grid(df, "common")
    assert common["nr_topics"].to_list() == [10, 20, None, 10, 15]
    dataset = in_topic_grid(df, "dataset")
    assert dataset["nr_topics"].to_list() == [4, 10, None, 10, 15]
    assert in_topic_grid(df.drop("dataset_label"), "dataset").height == df.height
