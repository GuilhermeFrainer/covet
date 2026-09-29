# BERTopic Default Parameters, Dimensionality, and Clustering Decisions

**Date:** 2026-09-16  
**Status:** Implemented & Verified in Active Configurations  
**Related Documents:**
- [docs/archive/EXPERIMENT_INTEGRITY_NEXT_STEPS.md](archive/EXPERIMENT_INTEGRITY_NEXT_STEPS.md)
- [docs/REPOSITORY_ISSUES.md](REPOSITORY_ISSUES.md)
- [docs/archive/experiment_integrity_repair_plan.md](archive/experiment_integrity_repair_plan.md)
- [docs/pairwise_repository_findings.md](pairwise_repository_findings.md)

---

## 1. Context and Problem Statement

BERTopic relies on a modular architecture typically pairing a dimensionality reducer (stock: UMAP) with a density-based clustering algorithm (stock: HDBSCAN). When creating custom topic models or integrating document metadata (such as in CA-BERTopic), submodels are constructed independently and passed into `BERTopic(umap_model=..., hdbscan_model=...)`.

An audit of the repository revealed that constructing these estimators directly via bare constructors (`UMAP()`, `HDBSCAN()`) caused them to adopt bare library defaults rather than BERTopic's internally tuned defaults.

### Summary of Parameter Discrepancies

| Submodel | Parameter | BERTopic Internal Default | Bare Library Default | Previous Repository State | Corrected Active State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UMAP** | `n_components` | **5** | **2** | Omitted (defaulted to 2) | **Explicitly 5** |
| **UMAP** | `low_memory` | **False** | **True** | Omitted (defaulted to True) | **Explicitly False** |
| **UMAP** | `n_neighbors` | **15** | 15 | 15 (explicit/default) | **Explicitly 15** |
| **UMAP** | `min_dist` | **0.0** | 0.1 | 0.0 (explicit) | **Explicitly 0.0** |
| **UMAP** | `metric` | **cosine** | euclidean | cosine (explicit) | **Explicitly cosine** |
| **HDBSCAN** | `prediction_data` | **True** | **False** | Omitted (defaulted to False) | **Explicitly True** |
| **HDBSCAN** | `min_cluster_size` | **10** | **5** | Omitted (5) / 15 / 30 | **5 / 10 / 30 (by corpus)** |
| **HDBSCAN** | `cluster_selection_method` | **eom** | eom | eom | **Explicitly eom** |
| **PCA** | `n_components` | *N/A* | None (all features) | 5 (explicit) | **Explicitly 5** |

### Consequences of Previous State
1. **Uncontrolled Dimensional Confound:** PCA models explicitly used 5 components, while UMAP-family models (`umap`, `append_umap`, `aligned_umap`) operated in 2 dimensions, confounding dimensionality reduction architecture comparisons with coordinate width.
2. **Divergence from Canonical BERTopic:** The baseline model was operating in 2D rather than the standard 5D reference used in the literature.
3. **Out-of-sample Prediction:** HDBSCAN lacked `prediction_data=True`, disabling standard BERTopic inductive inference capabilities.

---

## 2. Clustering Decisions: HDBSCAN `min_cluster_size` Policy

In BERTopic, `min_cluster_size` controls the minimum number of documents required to form a topic cluster. Setting this parameter uniformly across drastically different corpora either fragments large datasets into uninterpretable micro-clusters or forces smaller survey datasets to collapse into pure outlier noise.

To maintain a consistent, comparable number of expected topics (evaluating grid `[10, 20, 30, 40, 50]`) across all five benchmark corpora, the following dataset-specific `min_cluster_size` policy has been adopted:

| Dataset | Domain & Content | Corpus Scale | `min_cluster_size` | Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **Trump** | Social media posts (tweets) | Large (~44,000 documents) | **30** | Large sample size; a threshold of 30 prevents severe fragmentation into hundreds of micro-clusters while preserving coherent macro topics. |
| **Yelp** | Business & restaurant reviews | Medium-Large (~10,000 documents) | **10** | Matches BERTopic's canonical default; balances topic granularity and coverage on 10k aligned reviews. (Updated from previous value of 15). |
| **FED** | Federal Reserve communications | Medium (~5,400 documents) | **10** | Matches BERTopic's canonical default; ensures topic separation across macroeconomic statements and speeches. |
| **ANES** | Election survey open responses | Small (~1,000–2,000 documents) | **5** | Smaller survey sample; a threshold of 10 would classify excessive proportions of short responses as outlier noise. Size 5 captures distinct voter opinion clusters. |
| **Gadarian** | Public health open responses | Small (~500–1,000 documents) | **5** | Small experimental survey dataset; size 5 allows extracting the target 10–50 topics without swallowing valid low-frequency clusters. |

---

## 3. Dimensionality Reduction Policy

Across all **194 active standard UMAP-family configurations** (plain `umap`, `append_umap`, `aligned_umap` across both standard unstemmed and stemmed regimes):
- `n_components: 5`
- `n_neighbors: 15`
- `min_dist: 0.0`
- `metric: "cosine"`
- `low_memory: false`

Across all **50 active standard PCA configurations**:
- `n_components: 5` (guaranteeing structural dimensionality parity with UMAP).
- Enforced with an automated fallback in `src/models.py` (`get_algorithm`) so omitted `n_components` cannot trigger sklearn's default of retaining all dimensions.

---

## 4. Verification and Enforcement

Compliance with this policy is enforced via:
1. **Automated Parity Test Suite ([tests/test_bertopic_defaults_parity.py](../tests/test_bertopic_defaults_parity.py)):**
   - Directly checks that reference `BERTopic()` instantiates the documented defaults.
   - Iterates through all 274 active YAML files in `experiments/` to verify parameter compliance.
   - Executes synthetic fits for UMAP, AppendUMAP, AlignedUMAP, and PCA, asserting that fitted output dimension is strictly 5.
2. **Configuration Immutability Suite ([tests/test_config_mutation_regression.py](../tests/test_config_mutation_regression.py)):**
   - Guarantees that parameter dictionaries are not mutated across seeds or multi-run batches.
