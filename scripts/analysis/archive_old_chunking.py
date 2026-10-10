"""Archives Fed and Yelp results trained on the old chunking.

The chunking fixes (29a6e6f, 580e44d) rebuilt Fed and Yelp, so every run on
the old chunks is incomparable with the re-runs. Those runs are recognized by
their document count (`n_observations`), which the old chunking fixes
exactly: 5,446 Fed chunks and 10,205 Yelp chunks. Rows with any other count
are kept, including new STM runs, which can train on slightly fewer
documents than the embeddings file holds.

Every top-level file in `results/` (CSV) and `output/` (JSON) whose name
starts with the dataset is checked: merged files and raw run files alike.
Each file holding old rows is zipped into `<dir>/archive/` as it was, the
zip is verified byte for byte, and only then is the file rewritten without
those rows, or removed if none remain. Per-document assignment folders in
`output/document_assignments/` are left in place.

Usage:
    # Preview (nothing is written)
    uv run python scripts/analysis/archive_old_chunking.py
    # Archive and remove the old rows
    uv run python scripts/analysis/archive_old_chunking.py --apply
"""

import argparse
import datetime
import io
import json
import pathlib
import sys
import zipfile
from collections import Counter

import polars as pl

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.logger_config as logger_config

# Document counts of the old chunking (512-token chunks), from the runs on
# disk before the rebuild of 2026-10-08.
OLD_CHUNK_COUNTS = {"fed": "5446", "yelp": "10205"}


def candidate_files(root: pathlib.Path, dataset: str) -> list[pathlib.Path]:
    """Lists a dataset's result CSVs and topic JSONs, merged and raw.

    Args:
        root: Project root holding `results/` and `output/`.
        dataset: Dataset prefix, e.g. "fed".

    Returns:
        Sorted top-level files whose name starts with `<dataset>_`.
    """
    files = list((root / "results").glob(f"{dataset}_*.csv"))
    files += list((root / "output").glob(f"{dataset}_*.json"))
    return sorted(f for f in files if f.is_file())


def read_rows(path: pathlib.Path) -> tuple[pl.DataFrame | list[dict], list[str]]:
    """Reads a results CSV or topic JSON.

    Args:
        path: File to read.

    Returns:
        The rows (all-string DataFrame for CSV, list of dicts for JSON) and
        each row's `n_observations` as a string ("" when missing).
    """
    if path.suffix == ".csv":
        df = pl.read_csv(path, infer_schema_length=0)
        if "n_observations" not in df.columns:
            return df, [""] * df.height
        return df, [v or "" for v in df["n_observations"].to_list()]
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows, [
        "" if r.get("n_observations") is None else str(r["n_observations"])
        for r in rows
    ]


def write_rows(path: pathlib.Path, rows: pl.DataFrame | list[dict]) -> None:
    """Writes rows back in the format `read_rows` read them from."""
    if path.suffix == ".csv":
        rows.write_csv(path)
    else:
        path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def keep_rows(
    rows: pl.DataFrame | list[dict], keep: list[bool]
) -> pl.DataFrame | list[dict]:
    """Returns the rows whose `keep` flag is set."""
    if isinstance(rows, pl.DataFrame):
        return rows.filter(pl.Series(keep))
    return [row for row, k in zip(rows, keep) if k]


def archive_originals(files: list[pathlib.Path], zip_path: pathlib.Path) -> None:
    """Zips files as they are and checks the archive byte for byte.

    Raises:
        RuntimeError: If any archived file differs from the original.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    readme = io.StringIO()
    readme.write(
        "Fed/Yelp results trained on the old chunking (512-token chunks),\n"
        "archived by scripts/analysis/archive_old_chunking.py on "
        f"{datetime.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}.\n"
        "Each file is stored as it was before its old rows were removed.\n\n"
    )
    readme.writelines(f"  {f.name}\n" for f in files)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme.getvalue())
        for f in files:
            zf.write(f, f.name)
    with zipfile.ZipFile(zip_path) as zf:
        for f in files:
            if zf.read(f.name) != f.read_bytes():
                raise RuntimeError(f"Archive check failed for {f.name} in {zip_path}")


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Archive and remove the old rows (default: preview only).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(OLD_CHUNK_COUNTS),
        choices=sorted(OLD_CHUNK_COUNTS),
        help="Datasets to clean (default: fed yelp).",
    )
    args = parser.parse_args()
    logger = logger_config.setup_logging("archive_old_chunking", PROJECT_ROOT / "logs")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset in args.datasets:
        old = OLD_CHUNK_COUNTS[dataset]
        plan = []
        for path in candidate_files(PROJECT_ROOT, dataset):
            rows, counts = read_rows(path)
            is_old = [c == old for c in counts]
            others = Counter(c or "missing" for c in counts if c != old)
            n_old = sum(is_old)
            logger.info(
                f"{path.relative_to(PROJECT_ROOT)}: {n_old} old of {len(counts)} rows"
                + (f"; kept counts {dict(others)}" if others else "")
            )
            if n_old:
                plan.append((path, rows, is_old))

        old_rows = sum(sum(flags) for _, _, flags in plan)
        logger.info(
            f"{dataset}: {old_rows} old rows in {len(plan)} files "
            f"(n_observations = {old})."
        )
        if not args.apply or not plan:
            continue

        for subdir in ("results", "output"):
            files = [p for p, _, _ in plan if p.parent.name == subdir]
            if files:
                zip_path = (
                    PROJECT_ROOT
                    / subdir
                    / "archive"
                    / f"{dataset}_old_chunking_{stamp}.zip"
                )
                archive_originals(files, zip_path)
                logger.info(
                    f"Archived {len(files)} files to "
                    f"{zip_path.relative_to(PROJECT_ROOT)}"
                )

        for path, rows, is_old in plan:
            kept = keep_rows(rows, [not o for o in is_old])
            if len(kept):
                write_rows(path, kept)
                logger.info(f"Rewrote {path.name} with {len(kept)} rows")
            else:
                path.unlink()
                logger.info(f"Removed {path.name} (no rows left)")

    if not args.apply:
        logger.info(
            "Preview only; rerun with --apply to archive and remove the old rows."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
