from typing import Any, Optional, Union

import numpy as np
import polars as pl
from umap import UMAP


class AppendUMAP(UMAP):
    """
    A custom UMAP class that extends the standard UMAP functionality to concatenate
    user-specified dimensions (metadata) to the input embeddings BEFORE running
    dimensionality reduction.

    Args:
        metadata (Union[np.ndarray, pl.DataFrame, Any], optional):
            An array or DataFrame containing metadata dimensions to be concatenated
            to the input document embeddings.
        **kwargs:
            Any additional keyword arguments to be passed to the
            underlying umap.UMAP constructor, such as n_components,
            n_neighbors, min_dist, etc.
    """

    def __init__(
        self,
        metadata: Optional[Union[np.ndarray, pl.DataFrame, Any]] = None,
        **kwargs,
    ):
        # Call the constructor of the parent UMAP class, passing all other kwargs.
        super().__init__(**kwargs)

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

    def _concatenate_metadata(self, X: np.ndarray) -> np.ndarray:
        """
        Concatenates the metadata to the input document embeddings.
        """
        if self.metadata is not None:
            if X.shape[0] != self.metadata.shape[0]:
                raise ValueError(
                    f"Shape mismatch: X has {X.shape[0]} samples, "
                    f"but metadata has {self.metadata.shape[0]} samples."
                )
            return np.hstack((X, self.metadata))
        return X

    def fit(self, X, y=None, *args, **kwargs):
        """
        Fits the UMAP model on the concatenated embeddings and metadata.
        """
        X_combined = self._concatenate_metadata(X)
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
