# Implementation Plan: `fast-tritopic`

## 1. Executive Summary

This document provides a comprehensive, production-ready specification for implementing **`fast-tritopic`**, an optimized extension of the [`TriTopic`](https://github.com/SmartVisions-AI/tritopic) multi-modal graph topic modeling library created by **Roman Egger** ([SmartVisions-AI](https://github.com/SmartVisions-AI/tritopic)).

The primary goal of this package is to resolve a critical $O(N^2 \cdot k)$ bottleneck during metadata graph construction while guaranteeing 100% mathematical and functional parity with the base `TriTopic` implementation.

---

## 2. Problem Statement & Root Cause

### 2.1 The Issue
During training on datasets with numerical metadata (especially at scales $\ge 50,000$ documents), `TriTopic` execution triggers thousands of continuous SciPy warnings and severe CPU slowdown:

```text
scipy/sparse/_index.py:168: SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil and dok are more efficient.
  self._set_intXint(row, col, x.flat[0])
```

### 2.2 Root Cause Analysis
In `tritopic.core.graph_builder.GraphBuilder.build_metadata_graph`:
- An empty Compressed Sparse Row matrix `adjacency = sp_csr((n_samples, n_samples), dtype=float)` is instantiated.
- When processing numerical metadata columns, a k-Nearest-Neighbors query is executed.
- The resulting neighbors are filtered with a threshold (`similarity > 0.8`) and iteratively assigned into the CSR matrix using scalar indexing:
  ```python
  for local_i in range(len(valid_idx)):
      for j_pos in range(1, idxs.shape[1]):
          sim = 1.0 - dists[local_i, j_pos]
          if sim > 0.8:
              gi = valid_idx[local_i]
              gj = valid_idx[idxs[local_i, j_pos]]
              adjacency[gi, gj] = adjacency[gi, gj] + sim  # Point mutation
              adjacency[gj, gi] = adjacency[gj, gi] + sim  # Point mutation
  ```
- Because CSR format stores elements in contiguous flat arrays (`data`, `indices`, `indptr`), every single point assignment requires reallocating arrays, shifting all subsequent entries down by 1 in memory, and updating row pointer offsets.
- For $N = 55,000$ documents and $k = 15$ neighbors, up to **1,650,000 array reallocations** occur per numerical column, degrading complexity to $O(N^2 \cdot k)$.

---

## 3. Proposed Architecture & Solution

### 3.1 Design Principles
1. **Subclass Extension:** Subclass `GraphBuilder` and `TriTopic` without modifying upstream `tritopic` internals.
2. **Vectorized Coordinate Construction:** Replace scalar loops with NumPy boolean indexing and coordinate triplet arrays `(data, (rows, cols))`.
3. **Drop-in Compatibility:** `FastTriTopic` must be a drop-in replacement for `TriTopic`, preserving all methods, signatures, and attributes (`fit`, `fit_transform`, `transform`, `transform_proba`, `reduce_topics`, etc.).
4. **Mathematical Parity:** Output graphs and model states must be bit-for-bit identical to baseline `TriTopic` when seeded with the same `random_state`.

---

## 4. Repository Structure

```text
fast-tritopic/
├── .github/
│   └── workflows/
│       └── ci.yml
├── benchmarks/
│   └── benchmark_parity_and_speed.py
├── src/
│   └── fast_tritopic/
│       ├── __init__.py
│       ├── core.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   ├── test_parity.py
│   └── test_warnings.py
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 5. Step-by-Step Implementation Roadmap

### Phase 1: Environment & Project Configuration

#### `pyproject.toml`
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fast-tritopic"
version = "0.1.0"
description = "High-performance vectorized extensions for TriTopic topic modeling"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [
    { name = "Guilherme Frainer" }
]
dependencies = [
    "tritopic>=2.3.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.5.0",
]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

### Phase 2: Core Implementation

#### `src/fast_tritopic/core.py`
Implement `FastGraphBuilder` and `FastTriTopic`:

```python
"""
Core implementations for fast-tritopic.
"""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix as sp_csr
from sklearn.neighbors import NearestNeighbors
from tritopic import TriTopic, TriTopicConfig
from tritopic.core.graph_builder import GraphBuilder


class FastGraphBuilder(GraphBuilder):
    """
    Optimized GraphBuilder that vectorizes metadata graph construction,
    eliminating CSR sparsity structure mutation bottlenecks.
    """

    def build_metadata_graph(
        self,
        metadata: pd.DataFrame,
    ) -> sp_csr:
        """
        Build metadata similarity graph using fully vectorized sparse operations.
        """
        n_samples = len(metadata)
        adjacency = sp_csr((n_samples, n_samples), dtype=float)

        for col in metadata.columns:
            if (
                pd.api.types.is_string_dtype(metadata[col])
                or metadata[col].dtype.name == "category"
            ):
                # Categorical: sparse one-hot -> M @ M.T gives co-membership
                codes = metadata[col].astype("category").cat.codes.values
                valid = codes >= 0
                if not valid.any():
                    continue
                valid_idx = np.where(valid)[0]
                valid_codes = codes[valid]
                n_cats = valid_codes.max() + 1
                M = sp_csr(
                    (np.ones(len(valid_idx)), (valid_idx, valid_codes)),
                    shape=(n_samples, n_cats),
                )
                co_member = M.dot(M.T)
                co_member.setdiag(0)
                adjacency = adjacency + co_member
            else:
                # Numerical: vectorized kNN on normalized values
                values = metadata[col].values.astype(float)
                valid_mask = ~np.isnan(values)
                if valid_mask.sum() < 2:
                    continue
                valid_idx = np.where(valid_mask)[0]
                v = values[valid_idx]
                v_range = v.max() - v.min()
                if v_range < 1e-10:
                    continue
                v_norm = ((v - v.min()) / v_range).reshape(-1, 1)

                k_meta = min(self.n_neighbors, len(valid_idx) - 1)
                if k_meta < 1:
                    continue
                nn_meta = NearestNeighbors(
                    n_neighbors=k_meta + 1,
                    metric="euclidean",
                    algorithm="auto",
                )
                nn_meta.fit(v_norm)
                dists, idxs = nn_meta.kneighbors(v_norm)

                # Similarity = 1 - distance, filter threshold > 0.8
                sims = 1.0 - dists[:, 1:]
                mask = sims > 0.8
                if mask.any():
                    source_local = np.repeat(
                        np.arange(len(valid_idx)), idxs.shape[1] - 1
                    ).reshape(len(valid_idx), -1)

                    src_idx = valid_idx[source_local[mask]]
                    tgt_idx = valid_idx[idxs[:, 1:][mask]]
                    weight = sims[mask]

                    # Construct symmetric coordinate triplets
                    rows = np.concatenate([src_idx, tgt_idx])
                    cols = np.concatenate([tgt_idx, src_idx])
                    data = np.concatenate([weight, weight])

                    col_adj = sp_csr((data, (rows, cols)), shape=(n_samples, n_samples))
                    adjacency = adjacency + col_adj

        # Normalize
        max_val = adjacency.max()
        if max_val > 0:
            adjacency = adjacency / max_val

        return adjacency.tocsr()


class FastTriTopic(TriTopic):
    """
    TriTopic subclass utilizing FastGraphBuilder for high-performance metadata processing.
    """

    def __init__(
        self,
        config: TriTopicConfig | None = None,
        embedding_model: str | None = None,
        n_neighbors: int | None = None,
        n_topics: int | Literal["auto"] = "auto",
        use_iterative_refinement: bool | None = None,
        language: str | None = None,
        verbose: bool | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            config=config,
            embedding_model=embedding_model,
            n_neighbors=n_neighbors,
            n_topics=n_topics,
            use_iterative_refinement=use_iterative_refinement,
            language=language,
            verbose=verbose,
            random_state=random_state,
        )
        # Override internal graph builder with FastGraphBuilder
        self._graph_builder = FastGraphBuilder(
            n_neighbors=self.config.n_neighbors,
            metric=self.config.metric,
            graph_type=self.config.graph_type,
            snn_weight=self.config.snn_weight,
            language=self.config.language,
        )
```

#### `src/fast_tritopic/__init__.py`
```python
"""
fast-tritopic: Vectorized TriTopic implementation.
"""

from fast_tritopic.core import FastGraphBuilder, FastTriTopic

__all__ = ["FastGraphBuilder", "FastTriTopic"]
__version__ = "0.1.0"
```

---

### Phase 3: Comprehensive Verification & Test Suite

#### `tests/test_parity.py`
Verify that `FastGraphBuilder` and `FastTriTopic` yield results identical to baseline `GraphBuilder` and `TriTopic`.

```python
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from tritopic import TriTopic, TriTopicConfig
from tritopic.core.graph_builder import GraphBuilder

from fast_tritopic import FastGraphBuilder, FastTriTopic


@pytest.fixture
def sample_metadata():
    np.random.seed(42)
    n = 300
    return pd.DataFrame(
        {
            "category": np.random.choice(["X", "Y", "Z"], size=n),
            "feature_1": np.random.uniform(0, 100, size=n),
            "feature_2": np.random.normal(0, 1, size=n),
        }
    )


def test_metadata_graph_matrix_equality(sample_metadata):
    """Ensure FastGraphBuilder output matrix is identical to GraphBuilder."""
    gb_orig = GraphBuilder(n_neighbors=10)
    gb_fast = FastGraphBuilder(n_neighbors=10)

    # Disable warning capture here as orig will warn
    with pytest.warns(None):
        adj_orig = gb_orig.build_metadata_graph(sample_metadata).toarray()

    adj_fast = gb_fast.build_metadata_graph(sample_metadata).toarray()

    np.testing.assert_allclose(
        adj_fast,
        adj_orig,
        rtol=1e-6,
        atol=1e-6,
        err_msg="FastGraphBuilder output diverged from baseline GraphBuilder!",
    )


def test_full_model_fit_parity():
    """Ensure FastTriTopic produces identical topic assignments to TriTopic."""
    n_docs = 200
    docs = [
        f"This is document {i} discussing machine learning and data science."
        for i in range(n_docs)
    ]
    embeddings = np.random.RandomState(42).randn(n_docs, 64)
    metadata = pd.DataFrame(
        {
            "cat": ["A", "B", "C", "D"] * (n_docs // 4),
            "num": np.linspace(0, 1, n_docs),
        }
    )

    cfg = TriTopicConfig(
        use_metadata_view=True,
        random_state=42,
        n_consensus_runs=3,
        max_iterations=2,
        verbose=False,
    )

    m_orig = TriTopic(config=cfg, n_topics=4)
    m_orig.fit(documents=docs, embeddings=embeddings, metadata=metadata)

    m_fast = FastTriTopic(config=cfg, n_topics=4)
    m_fast.fit(documents=docs, embeddings=embeddings, metadata=metadata)

    np.testing.assert_array_equal(
        m_orig.labels_,
        m_fast.labels_,
        err_msg="Topic assignments between TriTopic and FastTriTopic do not match!",
    )
```

#### `tests/test_warnings.py`
Verify that `FastGraphBuilder` does not emit `SparseEfficiencyWarning`.

```python
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import SparseEfficiencyWarning
from fast_tritopic import FastGraphBuilder


def test_no_sparse_efficiency_warnings():
    df = pd.DataFrame(
        {
            "num": np.random.RandomState(42).rand(1000),
        }
    )
    builder = FastGraphBuilder(n_neighbors=15)

    with warnings.catch_warnings():
        warnings.simplefilter("error", SparseEfficiencyWarning)
        builder.build_metadata_graph(df)  # Must not raise
```

---

### Phase 4: Benchmark Script

#### `benchmarks/benchmark_parity_and_speed.py`
```python
import logging
import time
import numpy as np
import pandas as pd
from tritopic.core.graph_builder import GraphBuilder
from fast_tritopic import FastGraphBuilder

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def run_benchmark(n_samples: int = 5000) -> None:
    logger.info("=" * 60)
    logger.info(f"Starting Graph Construction Benchmark (N = {n_samples:,} samples)")
    logger.info("=" * 60)

    # Generate synthetic categorical and numerical metadata
    metadata = pd.DataFrame(
        {
            "cat": np.random.choice(["A", "B", "C", "D"], size=n_samples),
            "num1": np.random.rand(n_samples),
            "num2": np.random.rand(n_samples),
        }
    )

    # Benchmark FastGraphBuilder
    gb_fast = FastGraphBuilder(n_neighbors=15)
    t0 = time.perf_counter()
    adj_fast = gb_fast.build_metadata_graph(metadata)
    t_fast = time.perf_counter() - t0
    logger.info(f"[FastGraphBuilder] Completed in {t_fast:.4f}s")

    # Benchmark original GraphBuilder for comparable sample sizes
    if n_samples <= 5000:
        gb_orig = GraphBuilder(n_neighbors=15)
        t0 = time.perf_counter()
        adj_orig = gb_orig.build_metadata_graph(metadata)
        t_orig = time.perf_counter() - t0
        speedup = t_orig / t_fast if t_fast > 0 else float("inf")
        logger.info(
            f"[Original GraphBuilder] Completed in {t_orig:.4f}s (Speedup: {speedup:.1f}x)"
        )

        diff = np.abs((adj_orig - adj_fast).toarray()).max()
        logger.info(f"[Parity Check] Max absolute matrix difference: {diff:.6e}")
        assert diff < 1e-6, "Parity check failed: output matrices differ!"
    else:
        logger.info(
            "[Original GraphBuilder] Skipped for N > 5,000 to avoid CPU stalls."
        )


if __name__ == "__main__":
    for n in [500, 2000, 10000]:
        run_benchmark(n)
```

---

## 6. Acceptance Criteria

An agent working on this task in the new repository should verify the following before completing:
1. [ ] `uv build` builds clean sdist and wheel packages without warnings.
2. [ ] `pytest` passes 100% of parity and warning tests.
3. [ ] `ruff check .` and `ruff format .` pass with zero linter errors.
4. [ ] `FastGraphBuilder` emits zero `SparseEfficiencyWarning`.
5. [ ] Benchmark verifies $\ge 100\times$ speedup on numerical metadata graph creation.
