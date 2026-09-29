"""Topic–metadata alignment: adjusted mutual information per raw covariate.

Alignment is a descriptive outcome, not a quality score to maximize: a model
whose topics simply reproduce the metadata groups scores 1.0.
"""

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_mutual_info_score

N_BINS = 5
METHOD = (
    "adjusted mutual information (arithmetic normalization) between hard topic "
    f"labels and each raw covariate; numerics with more than {N_BINS} distinct "
    f"values are split into {N_BINS} quantile bins over the run's documents, "
    "other covariates are used as categories; nulls form their own category; "
    "noise (-1) counts as its own topic"
)


def discretize(values: pl.Series, n_bins: int = N_BINS) -> np.ndarray:
    """Turns one raw covariate into category labels for mutual information."""
    if values.dtype.is_temporal():
        values = values.to_physical()
    if values.dtype.is_numeric() and values.n_unique() > n_bins:
        values = values.cast(pl.Float64).qcut(
            n_bins, labels=[f"q{i}" for i in range(n_bins)], allow_duplicates=True
        )
    return values.cast(pl.String).fill_null("NA").to_numpy()


def metadata_alignment(topics, covariates: pl.DataFrame) -> dict:
    """Computes AMI between topic labels and each covariate column.

    Args:
        topics: Final topic label per document, aligned with covariates rows.
        covariates: Raw covariate values, one column per covariate.

    Returns:
        A dict with ``meta_ami_mean`` (mean over covariates) and
        ``meta_ami_by_covariate`` (covariate name -> AMI).
    """
    topics = np.asarray(topics)
    if len(topics) != covariates.height:
        raise ValueError(
            f"{len(topics)} topic labels for {covariates.height} covariate rows"
        )
    if covariates.width == 0:
        return {"meta_ami_mean": float("nan"), "meta_ami_by_covariate": {}}
    by_covariate = {
        name: float(adjusted_mutual_info_score(discretize(covariates[name]), topics))
        for name in covariates.columns
    }
    return {
        "meta_ami_mean": float(np.mean(list(by_covariate.values()))),
        "meta_ami_by_covariate": by_covariate,
    }
