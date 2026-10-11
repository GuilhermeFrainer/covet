# Experiment Results Merge & Lifecycle Guide

This document details the architecture, behavior, and lifecycle management of experimental results merging in the CA-BERTopic project, including the investigation and resolution of superseded raw result files.

---

## 1. Overview of the Merge Pipeline

Experiment executions generate individual run artifacts:
- **CSV Results** in `results/`: Formatted as `<experiment_id>-<timestamp>-<random_state>.csv` (e.g. `yelp_standard_aligned_umap_mv_k_means_keep_rep_stopwords-20260815-180652-36201624.csv`).
- **JSON Topic Outputs** in `output/`: Formatted as `<experiment_id>-<timestamp>-<random_state>.json`.

The script [`scripts/analysis/merge_results.py`](../scripts/analysis/merge_results.py) consolidates individual run files into dataset-wide merged files:
- `results/<dataset>_<type>_merged.csv`
- `output/<dataset>_<type>_merged.json`

where `<type>` is one of `standard`, `stemmed`, or `no_stopword_removal`.

---

## 2. Investigation: Superseded Raw Result Files

### Observed Phenomenon
After executing `merge_results.py`, certain unmerged files (e.g., older runs with timestamp `20260811-*`) remained in `results/` and `output/`, even though newer runs for the same models (with timestamp `20260815-*` or `20260816-*`) had been merged and archived into `results/archive/` and `output/archive/`.

### Root Cause Analysis
1. **Grouping Logic:** [`group_files`](../scripts/analysis/merge_results.py) grouped files by `(dataset_name, dataset_type, exp_id, random_state)` and filtered the list to keep only the newest run per experiment key in its returned list.
2. **Archival & Deletion Scope:** [`archive_files`](../scripts/analysis/merge_results.py) and file cleanup only operated on the list of files returned by `group_files`.
3. **Orphaned Runs:** Because older superseded runs were excluded during initial grouping, they were neither archived into the timestamped ZIP archive nor unlinked (deleted) from disk. They remained as untracked, seemingly unmerged files that required a second run of `merge_results.py` to be cleaned up.

---

## 3. Resolution & Single-Pass Archiving Fix

### Implementation Details
[`scripts/analysis/merge_results.py`](../scripts/analysis/merge_results.py) was refactored to explicitly track superseded files alongside latest runs:

1. **Superseded Tracking in `group_files`:**
   - When multiple files match the same `(dataset_name, dataset_type, exp_id, random_state)`, the newest run is retained in `grouped` (for active merging), while older runs are collected in `superseded_runs`.
   - When called with `return_superseded=True`, `group_files` returns `(grouped_latest, superseded_runs)`.

2. **Unified Archival & Cleanup:**
   - Both active latest files and any superseded older files for each dataset/type are pooled into `all_raw_files = files + superseded`.
   - `archive_files` packages all raw contributing and superseded files into the timestamped ZIP archive in a single pass.
   - All raw files are cleanly removed from `results/` and `output/`, preventing any orphaned files from remaining on disk.

3. **Data Integrity & Deduplication:**
   - The merge step [`merge_files`](../scripts/analysis/merge_results.py) and [`deduplicate_dataframe`](../scripts/analysis/merge_results.py) ensure that the latest timestamped metrics are retained in the merged CSV/JSON.
   - Rows are matched by model, requested topic count and seed, not by the run index in their name (`baseline_2_seed…`), which changes with the topic-count grid. Runs of the common and dataset-specific grids (see [src/topic_grids.py](../src/topic_grids.py)) are therefore kept side by side, and a rerun of the same topic count replaces the older run.
   - The completeness check counts only the topic counts a model's config requests now, so after a grid change a dataset's merge is blocked until the new grid is complete, unless `--allow-partial` is passed.

---

## 4. Verification

- **Unit Tests:** Added unit test coverage in [`tests/test_merge_results.py`](../tests/test_merge_results.py):
  - `test_group_files_return_superseded`: Validates detection and segregation of intermediate/older superseded files.
  - `test_archive_files_including_superseded`: Confirms that both latest and superseded files are archived into the ZIP and unlinked from disk in a single pass.
- **Test Suite:** The full project test suite passes cleanly.
