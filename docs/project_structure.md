# Project Script Structure

This document outlines the organization of scripts and execution pipelines in the CA-BERTopic project.

## Directory Overview

To maintain a clean and scalable workspace, all scripts, R files, test scripts, and execution pipelines are consolidated under the [scripts/](../scripts) directory. 

The top-level `batch/` directory has been removed, and its contents have been reorganized into subdirectories under `scripts/`.

```
├── scripts/
│   ├── data_prep/              # Dataset building, preprocessing, and feature prep
│   │   ├── align_yelp_sample.py
│   │   ├── build_datasets.py
│   │   ├── generate_embeddings.py
│   │   ├── generate_stemmed_configs.py
│   │   ├── preprocess_datasets.py
│   │   ├── sample_yelp_interim.py
│   │   └── summarize_datasets.py
│   │
│   ├── experiments/            # Core model training/execution entry points
│   │   ├── queue_exp.py
│   │   ├── slurm_job.sh
│   │   ├── run_optimizer.py
│   │   └── run_stm.py
│   │
│   ├── analysis/               # Results evaluation and verification utilities
│   │   ├── backfill_metadata_alignment.py
│   │   ├── calculate_noise_coverage.py
│   │   ├── calculate_total_time.py
│   │   ├── check_missing_results.py
│   │   ├── compare_stopword_impact.py
│   │   ├── count_models.py
│   │   ├── demsar_all_vs_all_analysis.py
│   │   ├── demsar_delta_analysis.py
│   │   ├── find_best_models.py
│   │   ├── inspect_document_assignments.py
│   │   └── merge_results.py
│   │
│   ├── r_scripts/              # R-specific execution scripts
│   │   ├── build_bow.R
│   │   ├── stm_smoke_test.R
│   │   └── train_stm.R
│   │
│   ├── pipelines/              # Orchestrators and batch runner scripts
│   │   ├── local_windows/      # Local Windows PowerShell (.ps1) runners [TRACKED]
│   │   ├── local_unix/         # Local Linux/macOS shell scripts (.sh) [IGNORED]
│   │   └── slurm/              # Cluster/Slurm cluster scripts (.sh) [TRACKED]
│   │
│   └── dashboard.py            # Streamlit visual results dashboard (entrypoint)
```

---

## Detailed Directory Breakdown

### 1. Data Preparation ([scripts/data_prep/](../scripts/data_prep))
This directory contains scripts for data ingestion, cleaning, feature engineering, and embedding generation.
- **[build_datasets.py](../scripts/data_prep/build_datasets.py)**: Unifies multiple raw data sources or converts format (e.g., Yelp NDJSON to Parquet).
- **[preprocess_datasets.py](../scripts/data_prep/preprocess_datasets.py)**: Performs text preprocessing, cleaning, and optional stemming.
- **[generate_embeddings.py](../scripts/data_prep/generate_embeddings.py)**: Runs SentenceTransformers to compute document embeddings.
- **[align_yelp_sample.py](../scripts/data_prep/align_yelp_sample.py)**: Subsamples the Yelp dataset to 10k documents while keeping exact alignment between BERTopic (chunked) and STM (un-chunked) document IDs.
- **[summarize_datasets.py](../scripts/data_prep/summarize_datasets.py)**: Utility to output statistics (token counts, document numbers) about the processed datasets.
- **[sample_yelp_interim.py](../scripts/data_prep/sample_yelp_interim.py)**: Samples 10,000 documents from the interim Yelp parquet data before preprocessing.
- **[generate_stemmed_configs.py](../scripts/data_prep/generate_stemmed_configs.py)**: Generates the `experiments/<dataset>_stemmed/` YAML configurations from the standard ones.

### 2. Experiments ([scripts/experiments/](../scripts/experiments))
Contains the batch dispatcher and experiment workers. Start experiment batches with [queue_exp.sh](../scripts/pipelines/slurm/queue_exp.sh).
- **[queue_exp.py](../scripts/experiments/queue_exp.py)**: Selects configurations and submits SLURM jobs through the shell entry point.
- **[slurm_job.sh](../scripts/experiments/slurm_job.sh)**: Executes a queued job.
- **[run_optimizer.py](../scripts/experiments/run_optimizer.py)**: Runs individual Python experiment configurations, including fixed parameters and parameter grids; also used by SLURM workers.
- **[run_stm.py](../scripts/experiments/run_stm.py)**: A Python wrapper to coordinate R-based Structural Topic Model training.

### 3. Analysis & Evaluation ([scripts/analysis/](../scripts/analysis))
Scripts for processing, evaluating, and compiling results.
- **[find_best_models.py](../scripts/analysis/find_best_models.py)**: Scrapes results to identify the best model configurations per metric and generates plots/LaTeX tables.
- **[merge_results.py](../scripts/analysis/merge_results.py)**: Consolidates and deduplicates results from individual experiment runs into merged files, with single-pass archiving and cleanup (see [merge_results_lifecycle.md](merge_results_lifecycle.md)).
- **[calculate_total_time.py](../scripts/analysis/calculate_total_time.py)**: Analyzes time metrics from logs to determine model throughput and compute times.
- **[count_models.py](../scripts/analysis/count_models.py)**: Summarizes completed model files.
- **[calculate_noise_coverage.py](../scripts/analysis/calculate_noise_coverage.py)**: Calculates HDBSCAN noise-cluster coverage and exports it as LaTeX.
- **[check_missing_results.py](../scripts/analysis/check_missing_results.py)**: Verifies experiment completeness and detects partial runs.
- **[compare_stopword_impact.py](../scripts/analysis/compare_stopword_impact.py)**: Compares metrics between runs with and without representation stopword removal.
- **[demsar_all_vs_all_analysis.py](../scripts/analysis/demsar_all_vs_all_analysis.py)**: Demšar (2006) all-vs-all comparisons: Friedman/Iman-Davenport, Nemenyi critical difference, and Holm-adjusted pairwise tests.
- **[demsar_delta_analysis.py](../scripts/analysis/demsar_delta_analysis.py)**: Model-by-metric delta tables with exact Wilcoxon tests and Holm adjustment.
- **[backfill_metadata_alignment.py](../scripts/analysis/backfill_metadata_alignment.py)**: Scores topic–metadata alignment (AMI) for already-exported runs without retraining; writes `results/metadata_alignment_backfill.csv`.
- **[inspect_document_assignments.py](../scripts/analysis/inspect_document_assignments.py)**: Read-only inspection of exported per-document topic assignments and raw metadata (see [document_assignment_exports.md](document_assignment_exports.md)).

### 4. R Scripts ([scripts/r_scripts/](../scripts/r_scripts))
Keeps R language scripts separated from the Python codebase.
- **[build_bow.R](../scripts/r_scripts/build_bow.R)**: Generates bag-of-words (BoW) representations and saves STM-compatible RDS data objects.
- **[train_stm.R](../scripts/r_scripts/train_stm.R)**: Subroutine executing the training of Structural Topic Models in R.
- **[stm_smoke_test.R](../scripts/r_scripts/stm_smoke_test.R)**: Trains a tiny STM on 200 Trump documents to check the R environment. It is the default command of the `Dockerfile.stm` image and is used by `scripts/pipelines/slurm/slurm_hello.sh`.

### 5. Pipelines & Orchestration ([scripts/pipelines/](../scripts/pipelines))
Consolidates sequential execution and batch runners.
- **`local_windows/`**: Local Windows PowerShell `.ps1` runners (representation builds, STM image build, results sync and export). *Note: Tracked by git.*
- **`local_unix/`**: Local shell scripts to fetch results. *Note: Ignored by git to allow local modification.*
- **`slurm/`**: Production SLURM scripts to dispatch jobs to clusters (primary batch entry point: `queue_exp.sh`). *Note: Tracked by git for reproducibility.*

---

## Example Usage Commands

### Dataset Prep
```bash
# Build & Preprocess
uv run scripts/data_prep/build_datasets.py --dataset fed
uv run scripts/data_prep/preprocess_datasets.py --dataset fed

# Generate Embeddings
uv run scripts/data_prep/generate_embeddings.py --dataset fed --columns clean_text
```

### Running Experiments
```bash
# Preview a batch, then submit it
bash scripts/pipelines/slurm/queue_exp.sh -d fed --dry-run
bash scripts/pipelines/slurm/queue_exp.sh -d fed

# Run an individual configuration
uv run python scripts/experiments/run_optimizer.py --exp fed/fed_standard_baseline
```

### Result Analysis
```bash
# Find best models and print table
uv run python scripts/analysis/find_best_models.py --dataset fed
```

### Dashboard UI
```bash
# Run Streamlit dashboard
uv run streamlit run scripts/dashboard.py
```
