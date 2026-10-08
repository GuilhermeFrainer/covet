"""Generate the RQ1 paper tables and figures from the current results.

Writes, under --output-dir (default ~/Downloads/covet_paper_outputs):
    tables/pairwise_ablation.tex           T1: planned comparisons (Δ, W/T/L, Holm p)
    tables/pairwise_ablation_appendix.tex  T1 for the appendix metrics
    tables/pairwise_ablation_stats.tex     rank-biserial r and Holm-adjusted p
    tables/pairwise_ablation.csv           the same numbers, for inspection
    figures/tradeoff.{pdf,png}             Δ coherence vs Δ IRBO per dataset
    figures/dose_response.{pdf,png}        weighted Append: Δ against metadata weight

Outputs are marked preliminary while any result was scored with an older
evaluation protocol (see docs/paper_results_plan.md).

Usage:
    uv run python scripts/analysis/make_paper_outputs.py
    uv run python scripts/analysis/make_paper_outputs.py --output-dir out/ --formats pdf
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import paper_outputs  # noqa: E402
from src.comparisons.analysis import TRUMP_VARIANTS, use_trump_variant  # noqa: E402
from src.model_catalog import load_catalog  # noqa: E402

DEFAULT_OUTPUT = Path.home() / "Downloads" / "covet_paper_outputs"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--coherence", default="c_npmi", help="x-axis metric of the trade-off figure"
    )
    parser.add_argument(
        "--diversity",
        default="irbo",
        help="y-axis metric of the trade-off figure",
    )
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument(
        "--trump",
        choices=list(TRUMP_VARIANTS),
        default="trump_s25000",
        help="Trump corpus counted as the fifth dataset (default: the 25k sample). "
        "Non-default choices write to a subfolder of --output-dir.",
    )
    args = parser.parse_args()

    catalog = load_catalog()
    results = use_trump_variant(paper_outputs.load_results(PROJECT_ROOT), args.trump)
    preliminary = paper_outputs.is_preliminary(results)
    documents = paper_outputs.documents_per_dataset(results)
    output_dir = (
        args.output_dir
        if args.trump == "trump_s25000"
        else args.output_dir / args.trump
    )
    tables_dir, figures_dir = output_dir / "tables", output_dir / "figures"
    note = (
        ""
        if args.trump == "trump"
        else f"Trump results use the {TRUMP_VARIANTS[args.trump]}."
    )

    datasets, summary = paper_outputs.compare(
        results, catalog, paper_outputs.PAIRWISE_VARIANTS
    )
    table = paper_outputs.ablation_table(
        datasets,
        summary,
        catalog,
        variants=paper_outputs.PAIRWISE_VARIANTS,
        metrics=paper_outputs.TABLE_METRICS + paper_outputs.APPENDIX_METRICS,
        documents=documents,
    )
    written = [
        paper_outputs.write_text(
            tables_dir / "pairwise_ablation.tex",
            paper_outputs.ablation_table_latex(
                table, catalog, preliminary=preliminary, note=note
            ),
        ),
        paper_outputs.write_text(
            tables_dir / "pairwise_ablation_appendix.tex",
            paper_outputs.ablation_table_latex(
                table,
                catalog,
                metrics=paper_outputs.APPENDIX_METRICS,
                preliminary=preliminary,
                label="tab:pairwise_ablation_appendix",
                note=note,
            ),
        ),
        paper_outputs.write_text(
            tables_dir / "pairwise_ablation_stats.tex",
            paper_outputs.ablation_stats_latex(
                table, catalog, preliminary=preliminary, note=note
            ),
        ),
    ]
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(tables_dir / "pairwise_ablation.csv")
    written.append(tables_dir / "pairwise_ablation.csv")

    points = paper_outputs.tradeoff_points(
        datasets, catalog, args.coherence, args.diversity
    )
    written += paper_outputs.plot_tradeoff(
        points,
        args.coherence,
        args.diversity,
        figures_dir / "tradeoff",
        catalog=catalog,
        preliminary=preliminary,
        formats=args.formats,
    )

    weighted = [model for model, _ in paper_outputs.WEIGHTED_APPEND]
    weighted_datasets, _ = paper_outputs.compare(results, catalog, weighted)
    curve = paper_outputs.dose_response(weighted_datasets, documents)
    written += paper_outputs.plot_dose_response(
        curve,
        figures_dir / "dose_response",
        preliminary=preliminary,
        formats=args.formats,
    )

    if preliminary:
        print(f"NOTE: {paper_outputs.PRELIMINARY_NOTE}")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
