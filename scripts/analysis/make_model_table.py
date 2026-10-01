"""Exports the LaTeX table of proposed models and the baselines they ablate.

The table is built from `config/model_catalog.yaml` and contains no results.
By default it lists only primary BERTopic variants and their reference baselines.

Example:
    uv run python scripts/analysis/make_model_table.py --latex out.tex
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.make_table import generate_model_ablation_latex_table
from src.model_catalog import CATALOG_PATH, load_catalog


def parse_args(argv=None):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export the model ablation design table as LaTeX."
    )
    parser.add_argument(
        "--latex",
        type=Path,
        help="Output .tex path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--include-secondary",
        action="store_true",
        help=(
            "Also list secondary models. Some differ from their baseline "
            "in more than one component."
        ),
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Also list external baselines (STM, TriTopic, FastTriTopic).",
    )
    parser.add_argument(
        "--include-weighted-append",
        action="store_true",
        help="Also list the weighted Append UMAP variants (append_umap_w*).",
    )
    parser.add_argument(
        "--include-changes",
        action="store_true",
        help=(
            "Add the 'What changes' column. The table then spans both columns "
            "of a two-column layout (table*)."
        ),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_PATH,
        help="Model catalog YAML (default: config/model_catalog.yaml).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Writes the model ablation table to a file or stdout."""
    args = parse_args(argv)
    latex = generate_model_ablation_latex_table(
        load_catalog(args.catalog),
        include_secondary=args.include_secondary,
        include_external=args.include_external,
        include_weighted_append=args.include_weighted_append,
        include_changes=args.include_changes,
    )
    if args.latex is None:
        print(latex)
        return
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    args.latex.write_text(latex + "\n", encoding="utf-8")
    print(f"Model ablation table written to {args.latex}")


if __name__ == "__main__":
    main()
