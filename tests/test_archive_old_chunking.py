"""Tests for scripts/analysis/archive_old_chunking.py."""

import json
import logging
import sys
import zipfile

import polars as pl
import pytest

import scripts.analysis.archive_old_chunking as aoc


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project root with Fed results and topics from both chunkings."""
    (tmp_path / "results").mkdir()
    (tmp_path / "output").mkdir()
    pl.DataFrame(
        {
            "experiment_id": [
                "fed_standard_baseline",
                "fed_standard_baseline",
                "fed_standard_stm",
            ],
            "n_observations": ["5446", "12814", "12790"],
            "c_v": ["0.512", "0.498", "0.401"],
        }
    ).write_csv(tmp_path / "results" / "fed_standard_merged.csv")
    pl.DataFrame(
        {"experiment_id": ["fed_standard_stm"], "n_observations": ["5446"]}
    ).write_csv(tmp_path / "results" / "fed_standard_stm-20261007-152224-36201624.csv")
    pl.DataFrame(
        {"experiment_id": ["anes_standard_baseline"], "n_observations": ["5446"]}
    ).write_csv(tmp_path / "results" / "anes_standard_merged.csv")
    topics = [
        {"model_id": "baseline_1", "topic_id": 0, "n_observations": 5446},
        {"model_id": "baseline_1", "topic_id": 0, "n_observations": 12814},
    ]
    (tmp_path / "output" / "fed_standard_merged.json").write_text(
        json.dumps(topics), encoding="utf-8"
    )
    monkeypatch.setattr(aoc, "PROJECT_ROOT", tmp_path)
    # setup_logging stops the shared "pipeline" logger from propagating, which
    # would hide its records from caplog in later tests.
    monkeypatch.setattr(
        aoc.logger_config,
        "setup_logging",
        lambda *_: logging.getLogger("archive_old_chunking_test"),
    )
    return tmp_path


def run(monkeypatch, *args):
    monkeypatch.setattr(
        sys, "argv", ["archive_old_chunking.py", "--datasets", "fed", *args]
    )
    return aoc.main()


def test_preview_changes_nothing(project, monkeypatch):
    before = {p: p.read_bytes() for p in project.rglob("*") if p.is_file()}
    assert run(monkeypatch) == 0
    after = {p: p.read_bytes() for p in project.rglob("*") if p.is_file()}
    assert after == before


def test_apply_removes_only_old_rows_and_archives_originals(project, monkeypatch):
    merged_csv = project / "results" / "fed_standard_merged.csv"
    original_csv = merged_csv.read_bytes()
    raw_csv = project / "results" / "fed_standard_stm-20261007-152224-36201624.csv"

    assert run(monkeypatch, "--apply") == 0

    kept = pl.read_csv(merged_csv, infer_schema_length=0)
    assert kept["n_observations"].to_list() == ["12814", "12790"]
    assert kept["c_v"].to_list() == ["0.498", "0.401"]
    assert not raw_csv.exists()
    topics = json.loads(
        (project / "output" / "fed_standard_merged.json").read_text(encoding="utf-8")
    )
    assert [t["n_observations"] for t in topics] == [12814]
    assert pl.read_csv(project / "results" / "anes_standard_merged.csv")[
        "n_observations"
    ].to_list() == [5446]

    (zip_path,) = (project / "results" / "archive").glob("fed_old_chunking_*.zip")
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.read("fed_standard_merged.csv") == original_csv
        assert raw_csv.name in zf.namelist()
    assert list((project / "output" / "archive").glob("fed_old_chunking_*.zip"))
