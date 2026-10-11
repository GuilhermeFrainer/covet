import pathlib
import sys
import zipfile

import polars as pl
import pytest

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.merge_results import (  # noqa: E402
    archive_files,
    deduplicate_dataframe,
    get_dataset_info,
    group_files,
    merge_files,
    normalize_dataset_name,
    standardize_json_dataframe,
)


def test_normalize_dataset_name():
    assert normalize_dataset_name("anes_stemmed") == "anes"
    assert normalize_dataset_name("trump") == "trump"
    assert normalize_dataset_name("") == ""


def test_normalize_dataset_name_only_aliases_yelp_sample():
    assert normalize_dataset_name("yelp_s10000") == "yelp"
    assert normalize_dataset_name("yelp_s10000_stemmed") == "yelp"
    assert normalize_dataset_name("trump_s25000") == "trump_s25000"
    assert normalize_dataset_name("trump_s25000_stemmed") == "trump_s25000"
    assert normalize_dataset_name("fed_s10000") == "fed_s10000"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("trump_s25000_standard_baseline-20260930-120000-1234.csv", "standard"),
        ("trump_s25000_stemmed_standard_stm-20260930-120000-1234.csv", "stemmed"),
        ("trump_s25000_standard_merged.csv", "standard"),
    ],
)
def test_get_dataset_info_keeps_sample_suffix(tmp_path, filename, expected):
    f = tmp_path / filename
    pl.DataFrame({"dataset_name": ["trump_s25000"], "c_v": [0.5]}).write_csv(f)

    d_name, d_type = get_dataset_info(f)
    assert d_name == "trump_s25000"
    assert d_type == expected


def test_get_dataset_info_yelp_sample_reports_as_yelp(tmp_path):
    f = tmp_path / "yelp_standard_stm-20260930-120000-1234.csv"
    pl.DataFrame({"dataset_name": ["yelp_s10000"], "c_v": [0.5]}).write_csv(f)

    d_name, _ = get_dataset_info(f)
    assert d_name == "yelp"


def test_get_dataset_info_stemmed(tmp_path):
    f_stemmed = tmp_path / "fed_stemmed_standard_baseline-20260805-120000-1234.csv"
    df = pl.DataFrame({"dataset_name": ["fed_stemmed"], "c_v": [0.5]})
    df.write_csv(f_stemmed)

    d_name, d_type = get_dataset_info(f_stemmed)
    assert d_name == "fed"
    assert d_type == "stemmed"


def test_get_dataset_info_no_stopword(tmp_path):
    f_nostop = tmp_path / "fed_no_stopword_baseline-20260805-120000-1234.csv"
    df = pl.DataFrame(
        {
            "dataset_name": ["fed"],
            "stopword_removal": ["keep_rep_stopwords"],
            "c_v": [0.5],
        }
    )
    df.write_csv(f_nostop)

    d_name, d_type = get_dataset_info(f_nostop)
    assert d_name == "fed"
    assert d_type == "no_stopword_removal"


def test_get_dataset_info_standard(tmp_path):
    f_std = tmp_path / "fed_standard_baseline-20260805-120000-1234.csv"
    df = pl.DataFrame(
        {
            "dataset_name": ["fed"],
            "stopword_removal": ["remove_rep_stopwords"],
            "c_v": [0.5],
        }
    )
    df.write_csv(f_std)

    d_name, d_type = get_dataset_info(f_std)
    assert d_name == "fed"
    assert d_type == "standard"


def test_group_files_and_merge(tmp_path):
    # Setup test files
    f_std_1 = tmp_path / "fed_standard_baseline-20260801-100000-1234.csv"
    f_std_2 = tmp_path / "fed_standard_baseline-20260802-100000-1234.csv"  # Latest std
    f_stemmed = tmp_path / "fed_stemmed_baseline-20260801-100000-1234.csv"

    pl.DataFrame({"dataset_name": ["fed"], "val": [1]}).write_csv(f_std_1)
    pl.DataFrame({"dataset_name": ["fed"], "val": [2]}).write_csv(f_std_2)
    pl.DataFrame({"dataset_name": ["fed_stemmed"], "val": [3]}).write_csv(f_stemmed)

    grouped = group_files(tmp_path, ".csv", ignore_suffix="_merged")

    assert ("fed", "standard") in grouped
    assert ("fed", "stemmed") in grouped
    assert len(grouped[("fed", "standard")]) == 1
    assert grouped[("fed", "standard")][0] == f_std_2
    assert len(grouped[("fed", "stemmed")]) == 1
    assert grouped[("fed", "stemmed")][0] == f_stemmed

    # Test merge_files
    out_std = tmp_path / "fed_standard_merged.csv"
    merge_files(
        grouped[("fed", "standard")],
        out_std,
        dry_run=False,
        force=True,
        allow_partial=True,
    )
    assert out_std.exists()

    res_df = pl.read_csv(out_std)
    assert res_df["val"][0] == 2


def test_archive_files(tmp_path):
    f_std_1 = tmp_path / "fed_standard_baseline-20260801-100000-1234.csv"
    f_std_2 = tmp_path / "fed_standard_baseline-20260802-100000-1234.csv"

    pl.DataFrame({"dataset_name": ["fed"], "val": [1]}).write_csv(f_std_1)
    pl.DataFrame({"dataset_name": ["fed"], "val": [2]}).write_csv(f_std_2)

    archive_dir = tmp_path / "archive"
    out_std = tmp_path / "fed_standard_merged.csv"

    archive_files(
        [f_std_1, f_std_2],
        archive_dir,
        "fed",
        "standard",
        out_std,
        dry_run=False,
        keep_originals=False,
    )

    zip_files = list(archive_dir.glob("*.zip"))
    assert len(zip_files) == 1
    zip_path = zip_files[0]
    assert "fed_standard_merged_" in zip_path.name

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert "README.txt" in names
        assert f_std_1.name in names
        assert f_std_2.name in names

        readme_text = zf.read("README.txt").decode("utf-8")
        assert "fed" in readme_text
        assert "standard" in readme_text
        assert out_std.name in readme_text
        assert f_std_1.name in readme_text

    # Originals should be cleaned up
    assert not f_std_1.exists()
    assert not f_std_2.exists()


def test_merge_files_keeps_newer_individual_over_older_merged(tmp_path):
    out_std = tmp_path / "fed_standard_merged.csv"
    pl.DataFrame(
        {
            "dataset_name": ["fed"],
            "model_name": ["baseline"],
            "random_state": [1234],
            "file_timestamp": ["20260801-100000"],
            "c_v": [0.5],
        }
    ).write_csv(out_std)

    f_newer = tmp_path / "fed_standard_baseline-20260805-100000-1234.csv"
    pl.DataFrame(
        {
            "dataset_name": ["fed"],
            "model_name": ["baseline"],
            "random_state": [1234],
            "file_timestamp": ["20260805-100000"],
            "c_v": [0.65],
        }
    ).write_csv(f_newer)

    merge_files([f_newer], out_std, dry_run=False, force=True, allow_partial=True)
    res_df = pl.read_csv(out_std)
    assert len(res_df) == 1
    assert res_df["c_v"][0] == 0.65
    assert res_df["file_timestamp"][0] == "20260805-100000"


def test_merge_files_keeps_newer_merged_over_older_individual(tmp_path):
    out_std = tmp_path / "fed_standard_merged.csv"
    pl.DataFrame(
        {
            "dataset_name": ["fed"],
            "model_name": ["baseline"],
            "random_state": [1234],
            "file_timestamp": ["20260810-100000"],
            "c_v": [0.85],
        }
    ).write_csv(out_std)

    f_older = tmp_path / "fed_standard_baseline-20260802-100000-1234.csv"
    pl.DataFrame(
        {
            "dataset_name": ["fed"],
            "model_name": ["baseline"],
            "random_state": [1234],
            "file_timestamp": ["20260802-100000"],
            "c_v": [0.45],
        }
    ).write_csv(f_older)

    merge_files([f_older], out_std, dry_run=False, force=True, allow_partial=True)
    res_df = pl.read_csv(out_std)
    assert len(res_df) == 1
    assert res_df["c_v"][0] == 0.85
    assert res_df["file_timestamp"][0] == "20260810-100000"


def test_merge_files_json_topic_deduplication(tmp_path):
    out_json = tmp_path / "fed_standard_merged.json"
    # Older run with 2 topics
    j_old = pl.DataFrame(
        {
            "dataset_name": ["fed", "fed"],
            "model_id": ["baseline", "baseline"],
            "topic_id": [0, 1],
            "random_state": [1234, 1234],
            "file_timestamp": ["20260801-100000", "20260801-100000"],
            "count": [10, 20],
        }
    )
    import json

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json.loads(j_old.write_json()), f)

    # Newer run with 2 updated topics and a new model
    f_new = tmp_path / "fed_standard_baseline-20260805-100000-1234.json"
    j_new = pl.DataFrame(
        {
            "dataset_name": ["fed", "fed", "fed"],
            "model_id": ["baseline", "baseline", "model_b"],
            "topic_id": [0, 1, 0],
            "random_state": [1234, 1234, 1234],
            "file_timestamp": [
                "20260805-100000",
                "20260805-100000",
                "20260805-100000",
            ],
            "count": [15, 25, 30],
        }
    )
    with open(f_new, "w", encoding="utf-8") as f:
        json.dump(json.loads(j_new.write_json()), f)

    merge_files([f_new], out_json, dry_run=False, force=True)
    res_df = pl.read_json(out_json)
    assert len(res_df) == 3
    # Baseline count should be 15 and 25 (from newer run)
    baseline_counts = res_df.filter(pl.col("model_id") == "baseline")["count"].to_list()
    assert sorted(baseline_counts) == [15, 25]


def test_group_files_return_superseded(tmp_path):
    f_std_1 = tmp_path / "fed_standard_baseline-20260801-100000-1234.csv"
    f_std_2 = tmp_path / "fed_standard_baseline-20260805-100000-1234.csv"  # Latest std
    f_std_3 = (
        tmp_path / "fed_standard_baseline-20260803-100000-1234.csv"
    )  # Intermediate
    f_stemmed = tmp_path / "fed_stemmed_baseline-20260801-100000-1234.csv"

    pl.DataFrame({"dataset_name": ["fed"], "val": [1]}).write_csv(f_std_1)
    pl.DataFrame({"dataset_name": ["fed"], "val": [2]}).write_csv(f_std_2)
    pl.DataFrame({"dataset_name": ["fed"], "val": [3]}).write_csv(f_std_3)
    pl.DataFrame({"dataset_name": ["fed_stemmed"], "val": [4]}).write_csv(f_stemmed)

    grouped, superseded = group_files(
        tmp_path, ".csv", ignore_suffix="_merged", return_superseded=True
    )

    assert ("fed", "standard") in grouped
    assert len(grouped[("fed", "standard")]) == 1
    assert grouped[("fed", "standard")][0] == f_std_2

    assert ("fed", "standard") in superseded
    superseded_names = [f.name for f in superseded[("fed", "standard")]]
    assert f_std_1.name in superseded_names
    assert f_std_3.name in superseded_names
    assert len(superseded[("fed", "standard")]) == 2

    assert ("fed", "stemmed") not in superseded


def test_archive_files_including_superseded(tmp_path):
    f_std_older = tmp_path / "fed_standard_baseline-20260801-100000-1234.csv"
    f_std_newer = tmp_path / "fed_standard_baseline-20260805-100000-1234.csv"

    pl.DataFrame({"dataset_name": ["fed"], "val": [1]}).write_csv(f_std_older)
    pl.DataFrame({"dataset_name": ["fed"], "val": [2]}).write_csv(f_std_newer)

    archive_dir = tmp_path / "archive"
    out_std = tmp_path / "fed_standard_merged.csv"

    all_raw = [f_std_newer, f_std_older]
    archive_files(
        all_raw,
        archive_dir,
        "fed",
        "standard",
        out_std,
        dry_run=False,
        keep_originals=False,
    )

    zip_files = list(archive_dir.glob("*.zip"))
    assert len(zip_files) == 1
    with zipfile.ZipFile(zip_files[0], "r") as zf:
        names = zf.namelist()
        assert f_std_older.name in names
        assert f_std_newer.name in names

    assert not f_std_older.exists()
    assert not f_std_newer.exists()


def test_merge_results_logging_and_file_creation(tmp_path):

    import src.logger_config as logger_config

    log_dir = tmp_path / "logs"
    logger = logger_config.setup_logging("merge_results", log_dir)

    f_test = tmp_path / "fed_standard_baseline-20260805-100000-1234.csv"
    pl.DataFrame({"dataset_name": ["fed"], "val": [10]}).write_csv(f_test)
    out_csv = tmp_path / "fed_standard_merged.csv"

    success = merge_files(
        [f_test],
        out_csv,
        dry_run=False,
        force=True,
        allow_partial=True,
        logger=logger,
    )
    assert success
    assert out_csv.exists()

    # Verify log file creation
    log_files = list(log_dir.glob("merge_results-*.log"))
    assert len(log_files) >= 1
    log_content = log_files[0].read_text(encoding="utf-8")
    assert "Starting experiment: merge_results" in log_content
    assert "Merging fed_standard_merged.csv" in log_content
    assert "Successfully saved merged results" in log_content


def test_merge_results_logging_warnings(tmp_path):
    import logging

    records = []

    class TestHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    custom_logger = logging.getLogger("test_merge_logger")
    custom_logger.setLevel(logging.INFO)
    custom_logger.addHandler(TestHandler())

    # Trigger warning via get_dataset_info on corrupt json file
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text("{invalid_json", encoding="utf-8")

    get_dataset_info(bad_file, logger=custom_logger)
    assert any("Could not read" in r.getMessage() for r in records)

    # Archive empty files check
    archive_dir = tmp_path / "archive"
    archive_files(
        [],
        archive_dir,
        "fed",
        "standard",
        tmp_path / "out.csv",
        dry_run=False,
        logger=custom_logger,
    )
    assert not archive_dir.exists()


def test_standardize_json_dataframe():
    # Test comma-separated string representation and list[int] representative docs
    df_raw = pl.DataFrame(
        {
            "representation": ["word1, word2, word3", '["word4", "word5"]'],
            "representative_docs": [[101, 102], [103]],
        }
    )
    std_df = standardize_json_dataframe(df_raw)
    assert std_df["representation"].dtype == pl.List(pl.String)
    assert std_df["representative_docs"].dtype == pl.List(pl.String)
    assert std_df["representation"].to_list() == [
        ["word1", "word2", "word3"],
        ["word4", "word5"],
    ]
    assert std_df["representative_docs"].to_list() == [["101", "102"], ["103"]]


def test_merge_files_json_mixed_schema_types(tmp_path):
    import json

    out_json = tmp_path / "yelp_standard_merged.json"

    # File 1: Tritopic style (representation: String, representative_docs: List(Int64))
    f_tritopic = tmp_path / "yelp_standard_tritopic-20260819-224904-1234.json"
    data1 = [
        {
            "model_id": "tritopic_1_seed1234",
            "topic_id": 0,
            "count": 100,
            "name": "Topic 0",
            "representation": "food, good, great",
            "representative_docs": [9118, 79],
            "dataset_name": "yelp",
            "random_state": 1234,
            "file_timestamp": "20260819-224904",
        }
    ]
    with open(f_tritopic, "w", encoding="utf-8") as f:
        json.dump(data1, f)

    # File 2: BERTopic style (representation: List(Str), docs: List(Str))
    f_bertopic = tmp_path / "yelp_standard_baseline-20260820-100000-1234.json"
    data2 = [
        {
            "model_id": "baseline_1_seed1234",
            "topic_id": 0,
            "count": 150,
            "name": "Topic 0",
            "representation": ["service", "staff", "friendly"],
            "representative_docs": ["Great service here!", "Staff was very kind"],
            "dataset_name": "yelp",
            "random_state": 1234,
            "file_timestamp": "20260820-100000",
        }
    ]
    with open(f_bertopic, "w", encoding="utf-8") as f:
        json.dump(data2, f)

    success = merge_files([f_tritopic, f_bertopic], out_json, dry_run=False, force=True)
    assert success
    assert out_json.exists()

    merged_df = pl.read_json(out_json)
    assert len(merged_df) == 2
    assert merged_df["representation"].dtype == pl.List(pl.String)
    assert merged_df["representative_docs"].dtype == pl.List(pl.String)


def _grid_runs(counts, timestamp, c_v):
    """Baseline runs for one seed, named by their index in a topic-count grid."""
    return pl.DataFrame(
        {
            "dataset_name": ["gadarian"] * len(counts),
            "model_name": [f"baseline_{i}_seed1234" for i in range(1, len(counts) + 1)],
            "random_state": [1234] * len(counts),
            "nr_topics": list(counts),
            "file_timestamp": [timestamp] * len(counts),
            "c_v": [c_v] * len(counts),
        }
    )


def test_merge_files_keeps_runs_of_two_topic_grids(tmp_path):
    """Run indices differ between grids; runs are matched by topic count."""
    out_csv = tmp_path / "gadarian_standard_merged.csv"
    _grid_runs((10, 20, 30, 40, 50), "20261001-100000", 0.5).write_csv(out_csv)
    new = tmp_path / "gadarian_standard_baseline-20261011-100000-1234.csv"
    _grid_runs((4, 6, 8, 10, 12), "20261011-100000", 0.9).write_csv(new)

    merge_files([new], out_csv, dry_run=False, force=True, allow_partial=True)

    res = pl.read_csv(out_csv).sort("nr_topics")
    assert res["nr_topics"].to_list() == [4, 6, 8, 10, 12, 20, 30, 40, 50]
    # The rerun of 10 topics replaces the older run; 20 to 50 are kept.
    assert res["c_v"].to_list() == [0.9] * 5 + [0.5] * 4
    assert "__run" not in res.columns


def test_deduplicate_dataframe_json_keeps_topics_of_two_grids():
    old = _grid_runs((10, 20), "20261001-100000", 0.5)
    new = _grid_runs((4, 10), "20261011-100000", 0.9)
    topics = pl.concat([old, new]).rename({"model_name": "model_id"})
    topics = pl.concat([topics.with_columns(topic_id=pl.lit(t)) for t in (0, 1)])

    res = deduplicate_dataframe(topics, is_json=True)

    kept = res.group_by("nr_topics").agg(pl.col("c_v").unique()).sort("nr_topics")
    assert kept["nr_topics"].to_list() == [4, 10, 20]
    assert kept["c_v"].to_list() == [[0.9], [0.9], [0.5]]
    assert res.height == 6
