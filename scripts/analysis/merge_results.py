import argparse
import datetime
import json
import logging
import os
import pathlib
import re
import sys
import zipfile
from typing import Any, Dict, List, Tuple

import polars as pl
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.logger_config as logger_config
from src.results_analysis import DATASET_ALIASES
from src.topic_grids import requested_topics_expr
from src.verification import RUN_INDEX_SUFFIX, verify_dataset_completeness

LOG_DIR = PROJECT_ROOT / "logs"


def normalize_dataset_name(name: str) -> str:
    """Strips the _stemmed suffix and applies dataset reporting aliases.

    Sample suffixes such as _s25000 are kept, so sampled datasets merge
    separately from their full corpus. Only aliases in DATASET_ALIASES
    (e.g. yelp_s10000 -> yelp) are reported under another name.
    """
    if not name:
        return name
    name = re.sub(r"_stemmed$", "", name)
    return DATASET_ALIASES.get(name, name)


def dataset_prefix(identifier: str) -> str:
    """Extracts the dataset name from the start of an experiment id or stem.

    Example: 'trump_s25000_standard_baseline' -> 'trump_s25000',
    'fed_stemmed_standard_stm' -> 'fed'.
    """
    return re.match(r"[^_]*(?:_s\d+(?=_|$))?", identifier).group(0)


def normalize_timestamp(ts: Any) -> str:
    """Normalizes a timestamp into a comparable 14-character string YYYYMMDDHHMMSS."""
    if ts is None:
        return ""
    digits = re.sub(r"\D", "", str(ts))
    return digits[:14].ljust(14, "0")


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """Parses a filename into experiment_id, timestamp, and random_state.

    Format: experiment_id-YYYYMMDD-HHMMSS-random_state.ext
    """
    # Remove extension
    stem = pathlib.Path(filename).stem
    parts = stem.split("-")

    if len(parts) < 3:
        return None, None, None

    random_state = parts[-1]
    # Timestamp has a dash in it: YYYYMMDD-HHMMSS
    timestamp = f"{parts[-3]}-{parts[-2]}"
    experiment_id = "-".join(parts[:-3])

    return experiment_id, timestamp, random_state


def get_dataset_info(
    file_path: pathlib.Path, logger: logging.Logger | None = None
) -> Tuple[str | None, str | None]:
    """Reads the file and/or inspects filename to extract (dataset_name, dataset_type).

    dataset_type can be 'stemmed', 'no_stopword_removal', or 'standard'.
    Uses fast filename-based inference first, falling back to reading file contents
    only when filename is ambiguous.
    """
    _logger = logger or logging.getLogger("pipeline")
    filename_lower = file_path.name.lower()

    # 1. Fast-path extraction from filename alone
    exp_id, _, _ = parse_filename(file_path.name)
    stem = file_path.stem
    if stem.endswith("_merged"):
        stem = stem[:-7]
    parts = stem.split("_")

    candidate_name = None
    if exp_id:
        candidate_name = dataset_prefix(exp_id)
    elif parts and parts[0]:
        candidate_name = dataset_prefix(stem)

    if candidate_name:
        is_stemmed = "stemmed" in filename_lower
        is_no_stopword = ("no_stopword" in filename_lower) or (
            "keep_rep_stopwords" in filename_lower
        )
        is_standard = "standard" in filename_lower

        if is_stemmed:
            return normalize_dataset_name(candidate_name), "stemmed"
        elif is_no_stopword:
            return normalize_dataset_name(candidate_name), "no_stopword_removal"
        elif is_standard:
            return normalize_dataset_name(candidate_name), "standard"

    # 2. Fallback: inspect file contents if filename is ambiguous
    dataset_name = None
    df = None

    try:
        if file_path.suffix == ".csv":
            df = pl.read_csv(file_path, n_rows=5, infer_schema_length=None)
        elif file_path.suffix == ".json":
            df = pl.read_json(file_path, infer_schema_length=None)
    except Exception as e:
        _logger.warning(f"Could not read {file_path}: {e}")

    if df is not None and "dataset_name" in df.columns and len(df) > 0:
        dataset_name = str(df["dataset_name"][0])

    if not dataset_name:
        if candidate_name:
            dataset_name = candidate_name

    if not dataset_name:
        return None, None

    # 1. Stemmed check
    is_stemmed = "stemmed" in filename_lower
    if df is not None:
        if "dataset_name" in df.columns and len(df) > 0:
            if "stemmed" in str(df["dataset_name"][0]).lower():
                is_stemmed = True
        if "text_column" in df.columns and len(df) > 0:
            if "stemmed" in str(df["text_column"][0]).lower():
                is_stemmed = True

    if is_stemmed:
        dataset_type = "stemmed"
    else:
        # 2. No stopword removal check
        is_no_stopword = ("no_stopword" in filename_lower) or (
            "keep_rep_stopwords" in filename_lower
        )
        if df is not None and "stopword_removal" in df.columns and len(df) > 0:
            status = str(df["stopword_removal"][0]).lower()
            if status in ("keep_rep_stopwords", "none", "no_stopword_removal", "false"):
                is_no_stopword = True

        if is_no_stopword:
            dataset_type = "no_stopword_removal"
        else:
            dataset_type = "standard"

    dataset_name = normalize_dataset_name(dataset_name)
    return dataset_name, dataset_type


def get_dataset_name(
    file_path: pathlib.Path, logger: logging.Logger | None = None
) -> str | None:
    """Reads the file to extract the dataset_name."""
    d_name, _ = get_dataset_info(file_path, logger=logger)
    return d_name


def group_files(
    directory: pathlib.Path,
    extension: str,
    ignore_suffix: str = None,
    return_superseded: bool = False,
    logger: logging.Logger | None = None,
) -> (
    Dict[Tuple[str, str], List[pathlib.Path]]
    | Tuple[
        Dict[Tuple[str, str], List[pathlib.Path]],
        Dict[Tuple[str, str], List[pathlib.Path]],
    ]
):
    """Groups files by (dataset_name, dataset_type) and keeps latest run.

    Args:
        directory: Path to directory containing result files.
        extension: File extension to scan for (e.g. '.csv', '.json').
        ignore_suffix: Filename suffix to ignore (e.g. '_merged').
        return_superseded: If True, returns a tuple of (latest_runs_dict,
            superseded_runs_dict). Otherwise, returns only latest_runs_dict.
        logger: Optional logger instance for warning messages.

    Returns:
        Dict or (Dict, Dict) mapping (dataset_name, dataset_type) to file lists.
    """
    latest_runs: Dict[Tuple[str, str, str, str], Tuple[str, pathlib.Path]] = {}
    superseded_runs: Dict[Tuple[str, str], List[pathlib.Path]] = {}
    base_files: Dict[Tuple[str, str], List[pathlib.Path]] = {}

    all_files = list(directory.glob(f"*{extension}"))
    for file_path in tqdm(
        all_files, desc=f"Scanning {directory.name} ({extension})", leave=False
    ):
        # Avoid including the file we might be writing to
        if ignore_suffix and file_path.stem.endswith(ignore_suffix):
            continue

        exp_id, timestamp, random_state = parse_filename(file_path.name)

        dataset_name, dataset_type = get_dataset_info(file_path, logger=logger)
        if not dataset_name or not dataset_type:
            continue

        group_key = (dataset_name, dataset_type)

        if exp_id:
            key = (dataset_name, dataset_type, exp_id, random_state)
            norm_ts = normalize_timestamp(timestamp)
            if key not in latest_runs:
                latest_runs[key] = (norm_ts, file_path)
            elif norm_ts > latest_runs[key][0]:
                _, old_path = latest_runs[key]
                superseded_runs.setdefault(group_key, []).append(old_path)
                latest_runs[key] = (norm_ts, file_path)
            else:
                superseded_runs.setdefault(group_key, []).append(file_path)
        else:
            base_files.setdefault(group_key, []).append(file_path)

    # Regroup by (dataset_name, dataset_type)
    grouped: Dict[Tuple[str, str], List[pathlib.Path]] = {}
    for (dataset_name, dataset_type, _, _), (_, file_path) in latest_runs.items():
        group_key = (dataset_name, dataset_type)
        if group_key not in grouped:
            grouped[group_key] = []
        grouped[group_key].append(file_path)

    for group_key, files in base_files.items():
        if group_key not in grouped:
            grouped[group_key] = []
        # Add base files, avoiding duplicates
        for bf in files:
            if bf not in grouped[group_key]:
                grouped[group_key].append(bf)

    if return_superseded:
        return grouped, superseded_runs
    return grouped


def run_key_expr(columns, name_column: str) -> pl.Expr:
    """Identifies a run by its model and requested topic count.

    Run names such as `baseline_2_seed36201624` carry the run's index in its
    config grid, which changes when the grid does: run 2 requests 20 topics
    on the common grid but 6 on Gadarian's own grid. Runs with a requested
    topic count are therefore keyed by model and count (`baseline@k=20`), so
    the runs of two grids are kept side by side and a rerun of the same
    count replaces the older run. Runs without one keep their name.

    This assumes the topic count is the only parameter a config varies, as
    in every standard config.
    """
    requested = requested_topics_expr(columns)
    name = pl.col(name_column).cast(pl.Utf8)
    return (
        pl.when(requested.is_not_null())
        .then(pl.format("{}@k={}", name.str.replace(RUN_INDEX_SUFFIX, ""), requested))
        .otherwise(name)
    )


def deduplicate_dataframe(df: pl.DataFrame, is_json: bool = False) -> pl.DataFrame:
    """Conservative deduplication: keeps the newer entry for each trial/model run.

    For CSV: keeps latest row for (dataset_name, run, random_state).
    For JSON: keeps all topics for latest (dataset_name, run, random_state).
    A run is its model and requested topic count (see `run_key_expr`).
    Also removes duplicate rows across all columns.
    """
    if df.is_empty():
        return df

    # Build standardized timestamp expression for comparisons
    ts_col = (
        "file_timestamp"
        if "file_timestamp" in df.columns
        else ("timestamp" if "timestamp" in df.columns else None)
    )
    if ts_col:
        ts_clean = (
            pl.col(ts_col)
            .cast(pl.Utf8)
            .fill_null("")
            .str.replace_all(r"\D", "")
            .str.slice(0, 14)
            .str.pad_end(14, "0")
        )
    else:
        ts_clean = pl.lit("00000000000000")

    df_with_ts = df.with_columns(ts_clean.alias("__norm_ts"))

    # Determine grouping key for unique runs
    run_key_cols = []
    if "dataset_name" in df.columns:
        run_key_cols.append("dataset_name")
    name_column = next(
        (c for c in ["model_name", "model_id", "experiment_id"] if c in df.columns),
        None,
    )
    if name_column:
        df_with_ts = df_with_ts.with_columns(
            run_key_expr(df.columns, name_column).alias("__run")
        )
        run_key_cols.append("__run")
    if "random_state" in df.columns:
        run_key_cols.append("random_state")

    if not run_key_cols:
        return df.unique(maintain_order=True)

    if not is_json:
        # Sort ascending by norm_ts, then pick the last (latest) per run_key_cols
        deduped = (
            df_with_ts.sort("__norm_ts", descending=False, nulls_last=False)
            .unique(subset=run_key_cols, keep="last", maintain_order=True)
            .drop("__norm_ts", "__run", strict=False)
        )
        return deduped
    else:
        # For JSON: multiple topic rows per run. Keep all topics
        # for latest __norm_ts per run_key_cols
        deduped = (
            df_with_ts.filter(
                pl.col("__norm_ts") == pl.col("__norm_ts").max().over(run_key_cols)
            )
            .drop("__norm_ts", "__run", strict=False)
            .unique(maintain_order=True)
        )
        return deduped


def standardize_json_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Standardizes JSON topic dataframe schemas.

    Ensures representation and representative_docs are typed as List(String).
    """
    if df.is_empty():
        return df

    repr_ok = "representation" not in df.columns or df[
        "representation"
    ].dtype == pl.List(pl.String)
    docs_ok = "representative_docs" not in df.columns or df[
        "representative_docs"
    ].dtype == pl.List(pl.String)
    if repr_ok and docs_ok:
        return df

    if "representation" in df.columns:
        dtype = df["representation"].dtype
        if dtype in (pl.String, pl.Utf8):

            def parse_repr(x):
                if x is None:
                    return []
                if isinstance(x, list):
                    return [str(w) for w in x]
                if not isinstance(x, str) or not x.strip():
                    return []
                if x.startswith("[") and x.endswith("]"):
                    try:
                        parsed = json.loads(x)
                        if isinstance(parsed, list):
                            return [str(w) for w in parsed]
                    except Exception:
                        pass
                return [w.strip() for w in x.split(",") if w.strip()]

            df = df.with_columns(
                pl.col("representation").map_elements(
                    parse_repr, return_dtype=pl.List(pl.String)
                )
            )
        elif isinstance(dtype, pl.List):
            df = df.with_columns(pl.col("representation").cast(pl.List(pl.String)))

    if "representative_docs" in df.columns:
        dtype = df["representative_docs"].dtype
        if dtype != pl.List(pl.String):
            if isinstance(dtype, pl.List):
                df = df.with_columns(
                    pl.col("representative_docs").cast(pl.List(pl.String))
                )
            elif dtype in (pl.String, pl.Utf8):

                def parse_docs(x):
                    if x is None:
                        return []
                    if isinstance(x, list):
                        return [str(d) for d in x]
                    if not isinstance(x, str) or not x.strip():
                        return []
                    if x.startswith("[") and x.endswith("]"):
                        try:
                            parsed = json.loads(x)
                            if isinstance(parsed, list):
                                return [str(d) for d in parsed]
                        except Exception:
                            pass
                    return [x]

                df = df.with_columns(
                    pl.col("representative_docs").map_elements(
                        parse_docs, return_dtype=pl.List(pl.String)
                    )
                )
            else:
                df = df.with_columns(
                    pl.col("representative_docs").map_elements(
                        lambda x: [str(x)] if x is not None else [],
                        return_dtype=pl.List(pl.String),
                    )
                )

    return df


def merge_files(
    files: List[pathlib.Path],
    output_path: pathlib.Path,
    dry_run: bool,
    force: bool,
    allow_partial: bool = False,
    dataset_name: str | None = None,
    dataset_type: str | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """Merges existing merged data and new incoming files with deduplication.

    Always keeps the newer entry when duplicates or re-runs are detected.
    Returns True if successful, False otherwise.
    """
    _logger = logger or logging.getLogger("pipeline")
    if not files and not output_path.exists():
        return False

    is_json = output_path.suffix == ".json"
    dfs: List[pl.DataFrame] = []
    existing_rows = 0

    # 1. Load existing merged file if present
    if output_path.exists():
        try:
            if output_path.suffix == ".csv":
                existing_df = pl.read_csv(output_path, infer_schema_length=None)
            elif output_path.suffix == ".json":
                existing_df = pl.read_json(output_path, infer_schema_length=None)
                existing_df = standardize_json_dataframe(existing_df)
            else:
                existing_df = None

            if existing_df is not None and not existing_df.is_empty():
                existing_rows = len(existing_df)
                dfs.append(existing_df)
        except Exception as e:
            _logger.warning(f"Could not read existing merged file {output_path}: {e}")

    # 2. Load incoming individual files
    incoming_rows = 0
    for f in tqdm(files, desc=f"Loading {output_path.name}", leave=False):
        try:
            if f.suffix == ".csv":
                df = pl.read_csv(f, infer_schema_length=None)
            elif f.suffix == ".json":
                df = pl.read_json(f, infer_schema_length=None)
                df = standardize_json_dataframe(df)
            else:
                continue

            if df is not None and not df.is_empty():
                # Ensure file_timestamp is populated if missing
                if "file_timestamp" not in df.columns:
                    _, parsed_ts, _ = parse_filename(f.name)
                    if parsed_ts:
                        df = df.with_columns(pl.lit(parsed_ts).alias("file_timestamp"))
                incoming_rows += len(df)
                dfs.append(df)
        except Exception as e:
            _logger.warning(f"Could not read file {f}: {e}")

    if not dfs:
        return False

    try:
        combined_df = pl.concat(dfs, how="diagonal_relaxed")
        deduped_df = deduplicate_dataframe(combined_df, is_json=is_json)
        final_rows = len(deduped_df)

        # 3. Check for partial models on CSV merges
        if not is_json:
            d_name = dataset_name
            d_type = dataset_type
            if not d_name or not d_type:
                d_name, d_type = get_dataset_info(output_path, logger=_logger)
            if d_name and d_type:
                report = verify_dataset_completeness(d_name, d_type, df=deduped_df)
                if report.has_partial_models:
                    _logger.warning(f"Partial models detected for {output_path.name}:")
                    for pm in report.partial_models:
                        missing_str = ", ".join(str(s) for s in pm.missing_seeds)
                        _logger.warning(
                            f"  - {pm.model_name}: {pm.found_runs}/{pm.expected_runs} "
                            f"runs. Missing seeds: [{missing_str}]"
                        )
                    slurm_cmd = report.slurm_rerun_command()
                    if slurm_cmd:
                        _logger.warning(f"  Slurm re-run command: {slurm_cmd}")
                    if not allow_partial:
                        _logger.warning(
                            f"Aborting merge for {output_path.name} to protect "
                            "raw files. Use --allow-partial to override."
                        )
                        return False

        _logger.info(
            f"Merging {output_path.name}: {existing_rows} existing + "
            f"{incoming_rows} incoming ({len(files)} files) -> "
            f"{final_rows} deduplicated rows (newer entries kept)."
        )

        if dry_run:
            _logger.info(f"  [Dry Run] Would write {final_rows} rows to {output_path}")
            return True

        if output_path.suffix == ".csv":
            deduped_df.write_csv(output_path)
        elif output_path.suffix == ".json":
            parsed_json = deduped_df.to_dicts()
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(parsed_json, out_f, indent=4)

        _logger.info(f"Successfully saved merged results to {output_path}")
        return True
    except Exception as e:
        _logger.error(f"Error merging files into {output_path}: {e}")
        return False


def move_files(
    files: List[pathlib.Path],
    move_to_dir: pathlib.Path,
    dry_run: bool,
    logger: logging.Logger | None = None,
):
    """Moves files to a specified directory."""
    _logger = logger or logging.getLogger("pipeline")
    if not move_to_dir.exists() and not dry_run:
        move_to_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(files, desc=f"Moving to {move_to_dir.name}", leave=False):
        target = move_to_dir / f.name
        _logger.info(f"Moving {f.name} to {target}...")
        if not dry_run:
            try:
                os.replace(f, target)
            except Exception as e:
                _logger.error(f"Error moving {f.name}: {e}")


def archive_files(
    files: List[pathlib.Path],
    archive_dir: pathlib.Path,
    dataset: str,
    dataset_type: str,
    output_path: pathlib.Path,
    dry_run: bool,
    keep_originals: bool = False,
    logger: logging.Logger | None = None,
):
    """Packages original raw files into a timestamped ZIP archive with README.txt."""
    _logger = logger or logging.getLogger("pipeline")
    if not files:
        return

    now = datetime.datetime.now().astimezone()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    zip_name = f"{dataset}_{dataset_type}_merged_{timestamp_str}.zip"
    zip_path = archive_dir / zip_name

    file_list_str = "\n".join(f"  {idx + 1}. {f.name}" for idx, f in enumerate(files))
    readme_content = (
        "================================================================================\n"
        "CA-BERTopic - Merged Experiment Results Archive\n"
        "================================================================================\n\n"
        "Merge Metadata:\n"
        "---------------\n"
        f"Merge Timestamp:        {now.strftime('%Y-%m-%d %H:%M:%S %z')}\n"
        f"Dataset Name:           {dataset}\n"
        f"Dataset Type:           {dataset_type}\n"
        f"Target Merged File:     {output_path.name}\n"
        f"Total Raw Files Merged: {len(files)}\n\n"
        "Archived Files:\n"
        "---------------\n"
        f"This archive contains {len(files)} original raw result files that were\n"
        f"consolidated into '{output_path.name}':\n\n"
        f"{file_list_str}\n\n"
        "Notes:\n"
        "------\n"
        "- Only latest file per (experiment_id, random_state) merged; of its rows,\n"
        "  the latest run per (model, requested topic count, random_state) kept.\n"
        "- Identical duplicate rows deduplicated in final merged file.\n"
    )

    if dry_run:
        cleanup_msg = (
            "originals will be kept" if keep_originals else "originals will be removed"
        )
        _logger.info(
            f"  [Dry Run] Archive ZIP: {zip_path} "
            f"(contains README.txt + {len(files)} files; {cleanup_msg})"
        )
        return

    _logger.info(f"Archiving {len(files)} files into {zip_path}...")
    if not archive_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README.txt", readme_content)
            for f in tqdm(files, desc=f"Archiving {zip_name}", leave=False):
                zf.write(f, arcname=f.name)

        _logger.info(f"Successfully created archive {zip_path}")

        if not keep_originals:
            for f in files:
                try:
                    if f.exists():
                        f.unlink()
                except Exception as e:
                    _logger.error(f"Error removing {f.name}: {e}")
    except Exception as e:
        _logger.error(f"Error archiving files into {zip_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge experiment results and outputs by dataset and dataset type."
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory for CSV results."
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Directory for JSON outputs."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs).",
    )
    parser.add_argument(
        "--suffix", type=str, default="_merged", help="Suffix for merged files."
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        help="Optional custom directory for ZIP archives.",
    )
    parser.add_argument(
        "--move-to",
        type=str,
        help="Optional directory to move files after merging.",
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep original raw files in place after creating the ZIP archive.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Disable ZIP archive creation.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow merging even if partial model runs are detected.",
    )

    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    log_dir = pathlib.Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir

    logger = logger_config.setup_logging("merge_results", log_dir)

    # Process CSV Results
    logger.info("--- Processing CSV Results ---")
    if results_dir.exists():
        grouped_csv, superseded_csv = group_files(
            results_dir,
            ".csv",
            ignore_suffix=args.suffix,
            return_superseded=True,
            logger=logger,
        )
        all_csv_to_move = []
        for (dataset, dataset_type), files in tqdm(
            grouped_csv.items(), desc="Processing CSV datasets"
        ):
            output_path = results_dir / f"{dataset}_{dataset_type}{args.suffix}.csv"
            merged_ok = merge_files(
                files,
                output_path,
                args.dry_run,
                args.force,
                allow_partial=args.allow_partial,
                dataset_name=dataset,
                dataset_type=dataset_type,
                logger=logger,
            )
            superseded = superseded_csv.get((dataset, dataset_type), [])
            all_raw_files = files + superseded
            if merged_ok and not args.no_archive and all_raw_files:
                csv_archive_dir = (
                    pathlib.Path(args.archive_dir)
                    if args.archive_dir
                    else results_dir / "archive"
                )
                archive_files(
                    all_raw_files,
                    csv_archive_dir,
                    dataset,
                    dataset_type,
                    output_path,
                    args.dry_run,
                    keep_originals=args.keep_originals,
                    logger=logger,
                )
            if merged_ok:
                all_csv_to_move.extend(all_raw_files)

        if args.move_to and all_csv_to_move and args.no_archive:
            move_dir = pathlib.Path(args.move_to)
            if not move_dir.is_absolute():
                move_dir = PROJECT_ROOT / move_dir
            move_files(all_csv_to_move, move_dir, args.dry_run, logger=logger)
    else:
        logger.warning(f"Results directory {results_dir} not found.")

    # Process JSON Outputs
    logger.info("--- Processing JSON Outputs ---")
    if output_dir.exists():
        grouped_json, superseded_json = group_files(
            output_dir,
            ".json",
            ignore_suffix=args.suffix,
            return_superseded=True,
            logger=logger,
        )
        all_json_to_move = []
        for (dataset, dataset_type), files in tqdm(
            grouped_json.items(), desc="Processing JSON datasets"
        ):
            output_path = output_dir / f"{dataset}_{dataset_type}{args.suffix}.json"
            merged_ok = merge_files(
                files,
                output_path,
                args.dry_run,
                args.force,
                allow_partial=args.allow_partial,
                dataset_name=dataset,
                dataset_type=dataset_type,
                logger=logger,
            )
            superseded = superseded_json.get((dataset, dataset_type), [])
            all_raw_files = files + superseded
            if merged_ok and not args.no_archive and all_raw_files:
                json_archive_dir = (
                    pathlib.Path(args.archive_dir)
                    if args.archive_dir
                    else output_dir / "archive"
                )
                archive_files(
                    all_raw_files,
                    json_archive_dir,
                    dataset,
                    dataset_type,
                    output_path,
                    args.dry_run,
                    keep_originals=args.keep_originals,
                    logger=logger,
                )
            if merged_ok:
                all_json_to_move.extend(all_raw_files)

        if args.move_to and all_json_to_move and args.no_archive:
            move_dir = pathlib.Path(args.move_to)
            if not move_dir.is_absolute():
                move_dir = PROJECT_ROOT / move_dir
            move_files(all_json_to_move, move_dir, args.dry_run, logger=logger)
    else:
        logger.warning(f"Output directory {output_dir} not found.")


if __name__ == "__main__":
    main()
