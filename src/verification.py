"""Experiment results completeness verification and partial run detection."""

import pathlib
from dataclasses import dataclass, field
from typing import Any, List, Optional

import polars as pl

import src.utils as utils
from src.document_assignments import requested_topic_setting
from src.optimizer import generate_hyperparameter_combinations
from src.topic_grids import requested_topics_expr

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
# Suffix the optimizer appends to a model id: `_<run index>[_seed<seed>]`.
RUN_INDEX_SUFFIX = r"_\d+(?:_seed\d+)?$"


@dataclass
class ModelCompleteness:
    """Detailed completion status for a specific model configuration."""

    model_name: str
    exp_stem: str
    expected_runs: int
    found_runs: int
    expected_seeds: List[int]
    found_seeds: List[int]
    missing_seeds: List[int]
    status: str  # "complete", "partial", or "unrun"


@dataclass
class CompletenessReport:
    """Structured report verifying experiment completeness before merging."""

    dataset_name: str
    dataset_type: str
    complete_models: List[ModelCompleteness] = field(default_factory=list)
    partial_models: List[ModelCompleteness] = field(default_factory=list)
    unrun_models: List[ModelCompleteness] = field(default_factory=list)
    null_metric_runs: List[str] = field(default_factory=list)

    @property
    def has_partial_models(self) -> bool:
        """Returns True if any model was started but is missing seeds/runs."""
        return len(self.partial_models) > 0

    @property
    def is_valid_for_merge(self) -> bool:
        """Returns True if all executed models are complete and have valid metrics."""
        return not self.has_partial_models and len(self.null_metric_runs) == 0

    def summary(self) -> str:
        """Generates a human-readable summary of the completeness report."""
        lines = [
            "=" * 72,
            f" Completeness Report: {self.dataset_name} ({self.dataset_type})",
            "=" * 72,
            f" Complete Models ({len(self.complete_models)}): "
            + ", ".join(m.model_name for m in self.complete_models)
            if self.complete_models
            else " Complete Models: None",
            f" Unrun Models    ({len(self.unrun_models)}): "
            + ", ".join(m.model_name for m in self.unrun_models)
            if self.unrun_models
            else " Unrun Models: None (All models were run)",
        ]

        if self.has_partial_models:
            lines.extend(
                [
                    "-" * 72,
                    f" PARTIAL MODELS DETECTED ({len(self.partial_models)}):",
                ]
            )
            for m in self.partial_models:
                missing_str = ", ".join(str(s) for s in m.missing_seeds)
                lines.append(
                    f"   - {m.model_name} ({m.exp_stem}): "
                    f"found {m.found_runs}/{m.expected_runs} runs. "
                    f"Missing seeds: [{missing_str}]"
                )

        if self.null_metric_runs:
            lines.extend(
                [
                    "-" * 72,
                    f" RUNS WITH NULL/NAN METRICS ({len(self.null_metric_runs)}):",
                ]
            )
            for r in self.null_metric_runs[:10]:
                lines.append(f"   - {r}")
            if len(self.null_metric_runs) > 10:
                lines.append(f"   ... and {len(self.null_metric_runs) - 10} more")

        lines.append("-" * 72)
        if self.is_valid_for_merge:
            lines.append(" Status: READY FOR MERGE (All executed models are complete)")
        else:
            lines.append(
                " Status: BLOCKED FOR MERGE (Partial models found; use --allow-partial)"
            )
        lines.append("=" * 72)
        return "\n".join(lines)

    def slurm_rerun_command(self) -> Optional[str]:
        """Generates the Slurm command to re-run only the partial models."""
        if not self.has_partial_models:
            return None

        model_names = [m.model_name for m in self.partial_models]
        models_arg = ",".join(model_names)
        stemmed_flag = " --stemmed" if self.dataset_type == "stemmed" else ""
        stopwords_flag = (
            " --keep-rep-stopwords"
            if self.dataset_type == "no_stopword_removal"
            else ""
        )

        return (
            f"./scripts/pipelines/slurm/queue_exp.sh "
            f"-d {self.dataset_name} -m {models_arg}{stemmed_flag}{stopwords_flag}"
        )


def verify_dataset_completeness(
    dataset_name: str,
    dataset_type: str,
    df: Optional[pl.DataFrame] = None,
    experiments_dir: Optional[pathlib.Path] = None,
) -> CompletenessReport:
    """Verifies that all executed models for a dataset and type are complete.

    Args:
        dataset_name: Base dataset name (e.g. 'yelp', 'fed', 'trump', 'anes').
        dataset_type: Preprocessing type ('standard', 'stemmed', or
          'no_stopword_removal').
        df: Optional combined results DataFrame. If None or empty, all models
          are marked unrun.
        experiments_dir: Optional custom path to experiments/ directory.

    Returns:
        A CompletenessReport detailing complete, partial, and unrun models.
    """
    exp_base = experiments_dir if experiments_dir else EXPERIMENTS_DIR
    folder_name = (
        f"{dataset_name}_stemmed" if dataset_type == "stemmed" else dataset_name
    )
    folder_path = exp_base / folder_name

    report = CompletenessReport(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
    )

    if not folder_path.exists():
        return report

    yamls = sorted(folder_path.glob("*.yaml"))

    for y in yamls:
        # Ignore STM or non-model configs
        if y.stem.endswith("_stm"):
            continue

        try:
            cfg = utils.load_config(str(y.relative_to(exp_base)), exp_base)
        except Exception:
            continue

        model_cfg = cfg.get("model", {})
        if not model_cfg:
            continue

        m_id = model_cfg.get("id", y.stem)
        seeds_raw: Any = cfg.get("experiment", {}).get("random_state", [])
        if isinstance(seeds_raw, (int, str)):
            seeds: List[int] = [int(seeds_raw)]
        elif isinstance(seeds_raw, list):
            seeds = [int(s) for s in seeds_raw if str(s).isdigit()]
        else:
            seeds = []

        combos = generate_hyperparameter_combinations(model_cfg)
        expected_runs = len(combos) * max(len(seeds), 1)
        # Topic counts the config requests now; results may also hold runs of
        # an earlier grid (see src/topic_grids.py), which do not count.
        topic_counts = {
            int(count)
            for combo, _ in combos
            if (count := requested_topic_setting(combo)) is not None
        }

        found_runs = 0
        found_seeds: List[int] = []

        if df is not None and not df.is_empty():
            conditions = []
            if "model_name" in df.columns:
                conditions.append(
                    pl.col("model_name").cast(pl.Utf8).str.replace(RUN_INDEX_SUFFIX, "")
                    == m_id
                )
            if "experiment_id" in df.columns:
                conditions.append(pl.col("experiment_id") == y.stem)

            if conditions:
                combined_cond = conditions[0]
                for cond in conditions[1:]:
                    combined_cond = combined_cond | cond
                sub = df.filter(combined_cond)
                if topic_counts:
                    requested = requested_topics_expr(sub.columns)
                    sub = sub.filter(
                        requested.is_null() | requested.is_in(sorted(topic_counts))
                    )
            else:
                sub = pl.DataFrame()

            found_runs = len(sub)
            if "random_state" in sub.columns:
                found_seeds = [
                    int(s)
                    for s in sub["random_state"].drop_nulls().unique().to_list()
                    if str(s).isdigit()
                ]

        missing_seeds = [s for s in seeds if s not in found_seeds]

        if found_runs == 0:
            status = "unrun"
            report.unrun_models.append(
                ModelCompleteness(
                    model_name=m_id,
                    exp_stem=y.stem,
                    expected_runs=expected_runs,
                    found_runs=0,
                    expected_seeds=seeds,
                    found_seeds=[],
                    missing_seeds=seeds,
                    status=status,
                )
            )
        elif found_runs < expected_runs or (seeds and len(missing_seeds) > 0):
            status = "partial"
            report.partial_models.append(
                ModelCompleteness(
                    model_name=m_id,
                    exp_stem=y.stem,
                    expected_runs=expected_runs,
                    found_runs=found_runs,
                    expected_seeds=seeds,
                    found_seeds=found_seeds,
                    missing_seeds=missing_seeds,
                    status=status,
                )
            )
        else:
            status = "complete"
            report.complete_models.append(
                ModelCompleteness(
                    model_name=m_id,
                    exp_stem=y.stem,
                    expected_runs=expected_runs,
                    found_runs=found_runs,
                    expected_seeds=seeds,
                    found_seeds=found_seeds,
                    missing_seeds=[],
                    status=status,
                )
            )

    # Check for NaN / null metrics
    if df is not None and not df.is_empty():
        critical_metrics = [
            c
            for c in ["u_mass", "c_v", "c_npmi", "irbo", "topic_diversity"]
            if c in df.columns
        ]
        if critical_metrics:
            null_conditions = []
            for c in critical_metrics:
                col_expr = pl.col(c)
                if df[c].dtype.is_numeric():
                    cond = col_expr.is_null() | col_expr.is_nan()
                else:
                    cond = col_expr.is_null() | col_expr.cast(
                        pl.Utf8
                    ).str.to_lowercase().is_in(["nan", "null", "none", ""])
                null_conditions.append(cond)

            combined_null_cond = null_conditions[0]
            for cond in null_conditions[1:]:
                combined_null_cond = combined_null_cond | cond

            null_df = df.filter(combined_null_cond)
            if not null_df.is_empty():
                for row in null_df.iter_rows(named=True):
                    m_name = row.get("model_name", "unknown_model")
                    seed = row.get("random_state", "")
                    for metric in critical_metrics:
                        val = row.get(metric)
                        if val is None or str(val).lower() in ("nan", "null", "none"):
                            report.null_metric_runs.append(
                                f"Model: {m_name} (Seed: {seed}) has null '{metric}'"
                            )
                            break

    return report
