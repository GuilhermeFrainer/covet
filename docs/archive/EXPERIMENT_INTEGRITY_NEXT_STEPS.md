# Experiment Integrity Repair: Next Steps and Execution Guide

> Historical planning document; implementation details may be superseded.
> Start experiment batches with `scripts/pipelines/slurm/queue_exp.sh`;
> use `scripts/experiments/run_optimizer.py` for individual Python configurations.

**Date:** 2026-09-16  
**Status:** In Progress (Stages 1 and 2 completed; Stage 3 in progress)  
**Parent Plan:** [docs/experiment_integrity_repair_plan.md](experiment_integrity_repair_plan.md)  
**Issue Tracking:** [REPOSITORY_ISSUES.md](../REPOSITORY_ISSUES.md)

---

## 1. Executive Summary & Current State

This document provides clear, actionable instructions for any agent or engineer continuing the experiment integrity repair. The repair addresses critical configuration confounds (2D vs 5D UMAP), factory dictionary mutation (loss of normalization on multi-seed runs), lack of run provenance, and metric truncation in storage.

### Status of Stages:
- **Stage 1 (Historical Audit): COMPLETED**
  - Audited 37 top-level CSV files (3,103 rows) and 55 archive ZIP files.
  - Confirmed that all CSVs share an identical 19-column schema lacking dimensionality, reducer parameters, or normalization fields.
  - No fitted BERTopic or reducer model artifacts exist in the repository (only R-based STM `.rds` files and autoencoder weights).
  - Formal conclusion: **"The stored artifacts do not establish this run's dimensionality."**
- **Stage 2 (Advisor Snapshot): COMPLETED**
  - Manually completed and preserved at `results/archive/pre_correction_2026-09-16.7z`.
  - Presentation materials and baseline tables for the advisor are securely frozen.
- **Stage 3 (Configuration Mutation Fix): COMPLETED**
  - Fixed in-place `.pop()` and parameter assignment mutation in `get_algorithm()` and model factory functions in [src/models.py](../../src/models.py).
  - Ensured multi-seed optimizer runs in [src/optimizer.py](../../src/optimizer.py) pass deep copies to prevent state leakage across seeds.
  - Added regression test suite in [tests/test_config_mutation_regression.py](../../tests/test_config_mutation_regression.py).
- **Stage 4 (BERTopic Defaults & 5D Dimensionality): COMPLETED**
  - Updated all 194 active standard UMAP-family YAML configurations to explicitly set `n_components: 5`, `n_neighbors: 15`, `min_dist: 0.0`, `metric: "cosine"`, `low_memory: false`.
  - Updated all 30 active HDBSCAN configurations to `prediction_data: true` and applied the dataset-specific `min_cluster_size` policy (5 for ANES/Gadarian, 10 for FED/Yelp, 30 for Trump).
  - Maintained 50 active PCA configurations at `n_components: 5` and added explicit fallback in `get_algorithm()`.
  - Full documentation in [docs/bertopic_default_parameters_and_clustering_decisions.md](../bertopic_default_parameters_and_clustering_decisions.md) and parity test suite in [tests/test_bertopic_defaults_parity.py](../../tests/test_bertopic_defaults_parity.py).
- **Stage 5 (Run Provenance & Metadata Recording): COMPLETED**
  - Implemented [src/run_provenance.py](../../src/run_provenance.py) capturing 20 provenance attributes: `result_schema_version`, `campaign_id` (`"bertopic_defaults_v2"`), `run_status`, `dim_red_output_dim`, `dim_red_n_components`, `dim_red_n_neighbors`, `dim_red_metric`, `dim_red_min_dist`, `dim_red_low_memory`, `cluster_min_cluster_size`, `cluster_min_samples`, `cluster_metric`, `cluster_selection_method`, `cluster_prediction_data`, `normalize_text_view`, `resolved_config_hash`, `run_manifest_path`, `code_revision`, `code_dirty`, `dependency_lock_hash`.
  - Integrated provenance collection and lightweight JSON run manifests into [src/optimizer.py](../../src/optimizer.py) and the now-removed legacy runner.
  - Enforced campaign isolation on resumption and diagonal merging in `Optimizer.save_results()`.
  - Protected table generators in [src/make_table.py](../../src/make_table.py) so provenance columns are excluded from metric calculations.
  - Unit and integration tests in [tests/test_run_provenance.py](../../tests/test_run_provenance.py).
- **Stage 6 (Full-Precision Storage, Unchanged Display Precision): COMPLETED**
  - Removed `float_precision=decimal_digits` truncation from `Optimizer.save_results()`, preserving full `Float64` precision in CSV storage across all model runners.
  - Preserved display formatting at 3 decimals in publication outputs (`float_format="%.3f"` in LaTeX, `decimals=3` in Great Tables).
  - Verified full-precision round-trip serialization and formatting stability in [tests/test_run_provenance.py](../../tests/test_run_provenance.py).
- **Stage 7 (Validation & Controlled Rollout): IN PROGRESS**

---

## 2. Detailed Instructions for Remaining Stages

### Stage 4: Versioned BERTopic Default Profile & Active YAML Alignment (5D)

#### Objective
Ensure all active standard UMAP-family models (plain UMAP, `AppendUMAP`, `AlignedUMAP`) explicitly configure **5 dimensions** to match canonical BERTopic defaults and maintain parity with the 50 active PCA configurations (which are already explicitly 5D).

#### Actionable Steps:
1. **Inspect Active Standard YAML Configurations:**
   - Standard configs reside in `experiments/<dataset>/<dataset>_standard_*.yaml` and `experiments/<dataset>_stemmed/<dataset>_stemmed_standard_*.yaml`.
   - Update `dimensionality_reduction.params` across all standard UMAP-family YAML files to explicitly define:
     ```yaml
     dimensionality_reduction:
       type: umap # or append_umap / aligned_umap
       params:
         n_components: 5
         n_neighbors: 15
         min_dist: 0.0
         metric: cosine
     ```
   - **Do NOT rewrite archived configurations** under `experiments/archive/`.
2. **Audit HDBSCAN & Clustering Defaults:**
   - Align the standard baseline HDBSCAN configuration with BERTopic's reference default (`min_cluster_size=10`, `prediction_data=True`).
   - If dataset-specific cluster sizes (e.g. Trump=30, Yelp=15) are retained as intentional variants, clearly label them as dataset-tuned variants rather than default BERTopic.
3. **Assert PCA Parity:**
   - Verify that all active PCA configurations specify `n_components: 5`.
   - Prevent constructor fallback in `get_algorithm()` where an omitted `n_components` for PCA might retain all features.

---

### Stage 5: Run Provenance & Effective Settings Tracking

#### Objective
Guarantee that every future experiment run persists its effective runtime configuration, estimator parameters, observed output dimensions, and code revision.

#### Actionable Steps:
1. **Create Provenance Collector Module (`src/run_provenance.py`):**
   - Inspect fitted estimators to extract:
     - `dim_red_n_components`: configured target dimensions.
     - `dim_red_output_dim`: actual observed fitted output shape (e.g. `embedding_.shape[1]`).
     - `dim_red_metric`, `dim_red_min_dist`, `dim_red_n_neighbors`.
     - `cluster_min_cluster_size`, `cluster_min_samples`.
     - `normalize_text_view`: boolean flag indicating whether text L2 normalization was applied.
     - `campaign_id`: e.g., `"bertopic_defaults_v2"`.
     - `code_revision`: git commit SHA and dirty state.
2. **Update Result CSV Output & Run Manifest:**
   - In [src/training.py](../../src/training.py) and [src/optimizer.py](../../src/optimizer.py), include provenance fields in the metric dictionary saved to CSV.
   - Write a lightweight JSON run manifest alongside qualitative topic outputs.
3. **Protect Table Generation ([src/make_table.py](../../src/make_table.py)):**
   - Ensure table generation utilities (`generate_gt_table`, `export_latex_table`) filter out provenance columns so they are not mistaken for floating-point evaluation metrics.

---

### Stage 6: Full-Precision Storage, Unchanged Display Precision

#### Objective
Preserve full floating-point accuracy in CSV storage while retaining standard 3-decimal formatting in publication tables.

#### Actionable Steps:
1. **Remove CSV Float Truncation:**
   - In [src/optimizer.py](../../src/optimizer.py), remove `float_precision=3` from `Optimizer.save_results()` when calling `write_csv()`.
   - Check all result writers in [src/training.py](../../src/training.py) and [src/optimizer.py](../../src/optimizer.py) to ensure unrounded `Float64` metrics are written.
2. **Preserve Presentation Precision:**
   - Leave `float_format="%.3f"` in LaTeX exporters and `decimals=3` in Great Tables formatters unchanged in [src/make_table.py](../../src/make_table.py).

---

### Stage 7: Validation and Controlled Rollout

#### Objective
Run complete verification across test suites, code standards, and establish the corrected rerun matrix.

#### Actionable Steps:
1. **Regression & Parity Tests:**
   - Verify repeated model construction across multiple seeds leaves parameters unaltered.
   - Run synthetic fits for UMAP, `AppendUMAP`, `AlignedUMAP`, and PCA verifying that observed fitted output width is 5.
2. **Full Test Suite & Code Quality:**
   - Run full pytest:
     ```bash
     uv run pytest
     ```
   - Run Ruff linter and formatter:
     ```bash
     uvx ruff check . --fix
     uvx ruff format .
     ```
3. **Controlled Rerun Execution:**
   - Formulate a clean rerun matrix by dataset, seed, and topic count for the `bertopic_defaults_v2` campaign.
   - Avoid blanket retraining of unaffected models (e.g. 5D PCA runs with valid historical records if provenance is established).

---

## 3. Important Architectural Constraints
- **Preserve Preprocessing Parity:** All text cleaning and stemming must occur in Python ([src/processing.py](../../src/processing.py)).
- **Strict Row Alignment:** Retain documents only if non-empty in both `clean_text` and `clean_text_stemmed`.
- **Representation Stopwords Policy:** Filter stop words at the c-TF-IDF representation layer (`CountVectorizer(stop_words="english")`) rather than embedding input text.
- **Campaign Isolation:** Never merge corrected `bertopic_defaults_v2` runs into legacy pre-correction CSVs.
