from typing import Any, Optional, Union

import numpy as np
import polars as pl
from umap import UMAP


def _mean_pairwise_sq_dist(X: np.ndarray) -> float:
    """Exact mean squared Euclidean distance over all pairs of distinct rows.

    Uses sum_{i,j} ||x_i - x_j||^2 = 2n * sum_i ||x_i||^2 - 2 * ||sum_i x_i||^2,
    which avoids materializing the n x n distance matrix.
    """
    n = X.shape[0]
    if n < 2:
        raise ValueError("At least two samples are required to calibrate views.")
    total = 2 * n * np.sum(X**2) - 2 * np.sum(np.sum(X, axis=0) ** 2)
    return float(max(total, 0.0) / (n * (n - 1)))


class AppendUMAP(UMAP):
    """
    A custom UMAP class that extends the standard UMAP functionality to concatenate
    user-specified dimensions (metadata) to the input embeddings BEFORE running
    dimensionality reduction.

    Two concatenation modes are supported:

    * Naive (``metadata_weight=None``): raw embeddings and metadata are stacked
      unchanged. The relative influence of each block is then an accident of
      their norms and encodings.
    * Weighted (``metadata_weight=w``): text embeddings are L2-normalized, each
      block is divided by the square root of its mean pairwise squared distance
      (fitted on the training data), and the blocks are scaled by
      ``sqrt(1 - w)`` and ``sqrt(w)``. The squared Euclidean distance in the
      joined space is then exactly
      ``(1 - w) * d_text^2 / s_text + w * d_meta^2 / s_meta``, so ``w`` is the
      metadata share of the average squared distance. Metadata levels are kept
      (no per-row normalization). Requires ``metric="euclidean"``; on unit
      vectors this ranks text neighbors exactly as cosine does.

    Args:
        metadata (Union[np.ndarray, pl.DataFrame, Any], optional):
            An array or DataFrame containing metadata dimensions to be concatenated
            to the input document embeddings.
        metadata_weight (float, optional):
            Metadata share ``w`` in [0, 1] for weighted concatenation. ``None``
            keeps naive concatenation. ``0`` ignores metadata entirely.
        **kwargs:
            Any additional keyword arguments to be passed to the
            underlying umap.UMAP constructor, such as n_components,
            n_neighbors, min_dist, etc.
    """

    def __init__(
        self,
        metadata: Optional[Union[np.ndarray, pl.DataFrame, Any]] = None,
        metadata_weight: Optional[float] = None,
        **kwargs,
    ):
        # Call the constructor of the parent UMAP class, passing all other kwargs.
        super().__init__(**kwargs)

        if metadata_weight is not None:
            if isinstance(metadata_weight, bool) or not 0.0 <= metadata_weight <= 1.0:
                raise ValueError(
                    f"metadata_weight must be in [0, 1], got {metadata_weight!r}"
                )
            if self.metric != "euclidean":
                raise ValueError(
                    "Weighted concatenation calibrates squared Euclidean distances; "
                    f"use metric='euclidean' (got {self.metric!r})."
                )
        self.metadata_weight = metadata_weight
        self.text_scale_: Optional[float] = None
        self.metadata_scale_: Optional[float] = None

        # Store the metadata for later use in concatenation.
        if metadata is not None:
            if isinstance(metadata, pl.DataFrame):
                self.metadata = metadata.to_numpy()
            elif hasattr(metadata, "to_numpy"):
                self.metadata = metadata.to_numpy()
            elif hasattr(metadata, "values"):
                self.metadata = metadata.values
            else:
                arr = np.asarray(metadata)
                self.metadata = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        else:
            self.metadata = None

    def _concatenate_metadata(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Concatenates the metadata to the input document embeddings.

        In weighted mode, ``fit=True`` (re)computes the per-block scales from
        the training data; otherwise the fitted scales are reused.
        """
        if self.metadata is None:
            return X
        if X.shape[0] != self.metadata.shape[0]:
            raise ValueError(
                f"Shape mismatch: X has {X.shape[0]} samples, "
                f"but metadata has {self.metadata.shape[0]} samples."
            )
        if self.metadata_weight is None:
            return np.hstack((X, self.metadata))

        X_arr = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
        text = X_arr / np.where(norms == 0, 1.0, norms)
        meta = np.asarray(self.metadata, dtype=np.float64)

        if fit:
            self.text_scale_ = _mean_pairwise_sq_dist(text)
            self.metadata_scale_ = _mean_pairwise_sq_dist(meta)
            if self.text_scale_ == 0 or self.metadata_scale_ == 0:
                raise ValueError(
                    "Cannot calibrate a view with zero spread (all rows identical)."
                )
        elif self.text_scale_ is None:
            raise ValueError("AppendUMAP has not been fitted yet.")

        w = self.metadata_weight
        return np.hstack(
            (
                np.sqrt((1 - w) / self.text_scale_) * text,
                np.sqrt(w / self.metadata_scale_) * meta,
            )
        )

    def fit(self, X, y=None, *args, **kwargs):
        """
        Fits the UMAP model on the concatenated embeddings and metadata.
        """
        X_combined = self._concatenate_metadata(X, fit=True)
        super().fit(X_combined, y, *args, **kwargs)
        return self

    def transform(self, X, *args, **kwargs) -> np.ndarray:
        """
        Transforms the data into the embedding space after concatenating metadata.
        """
        X_combined = self._concatenate_metadata(X)
        return super().transform(X_combined, *args, **kwargs)

    def fit_transform(self, X, y=None, *args, **kwargs) -> np.ndarray:
        """
        Fits the data and transforms it into the embedding space on the
        concatenated input.
        """
        # UMAP.fit_transform calls self.fit, which already concatenates;
        # concatenating here too would append the metadata twice.
        self.fit(X, y, *args, **kwargs)
        return self.embedding_

    @staticmethod
    def shape_dims(df: pl.DataFrame) -> np.ndarray:
        """
        Extracts and stacks specified columns into a 2D NumPy feature matrix.

        This method converts each selected column into a vertical vector and
        horizontally stacks them, creating a format suitable for dimensionality
        reduction algorithms (like UMAP).

        Parameters
        ----------
        df : pl.DataFrame
            The source Polars DataFrame containing the data.
            Dataframe must contain only the desired metadata.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_rows, n_cols) containing the stacked features.
        """
        import polars.selectors as cs

        # Skips categorical variables
        df_clean = df.select(~cs.by_dtype(pl.String))

        if len(df.columns) != len(df_clean.columns):
            print(
                f"Warning: Dropped {len(df.columns) - len(df_clean.columns)} "
                "non-numeric columns."
            )

        # Casts DateTimes to Float64
        df_clean = df_clean.with_columns(
            cs.by_dtype(pl.Datetime, pl.Date).cast(pl.Int64).cast(pl.Float64)
        )

        cols = df_clean.columns
        reoriented_dims = tuple(df_clean[c].to_numpy().reshape(-1, 1) for c in cols)
        return np.hstack(reoriented_dims)
