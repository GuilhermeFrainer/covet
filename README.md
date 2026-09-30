# CA-BERTopic Project

This project aims to modify BERTopic to incorporate document-level metadata into the topic modeling process using multi-view clustering (e.g., Multi-View K-Means, Multi-View Spectral, Co-regularized Spectral), multi-modal graph modeling (FastTriTopic/TriTopic), and Structural Topic Model (STM) benchmarks.

---

## Getting Started

### Prerequisites

*   **Python:** 3.12+
*   **Package Manager:** `uv`
*   **R (Optional):** R >= 4.0 with `renv` (required only for building Bag-of-Words and training R-based STM models)
*   **Local Dependencies:** `fast-tritopic` (`../fast-tritopic`) and `mv-hdbscan` (`../MV-HDBSCAN`) as local path dependencies, cloned next to this repository

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd CA-BERTopic
    ```

2.  Create a virtual environment and install dependencies using `uv`:
    ```bash
    uv venv
    uv sync
    ```
    > [!NOTE]
    > Linux- and NVIDIA GPU-specific dependencies (e.g., `cudf`, `cuml`, `cugraph`) are marked with `sys_platform == 'linux'` in `pyproject.toml`. On Windows or macOS, `uv sync` resolves CPU dependencies cleanly without modifying `pyproject.toml`.

---

## Supported Datasets & Ingestion Pipeline

The project supports five core datasets:
*   `anes`: American National Election Studies open-ended survey responses with political covariates.
*   `fed`: Federal Reserve communications linked with macroeconomic indicators and political party metadata.
*   `gadarian`: Open-ended responses regarding public health and emotion with demographic covariates.
*   `trump` / `trump_s25000`: Social media posts linked with engagement and timestamp metadata (`trump_s25000` is a fixed 25k-document sample for the heaviest models, reported separately from full Trump; see [docs/trump_downsampling.md](docs/trump_downsampling.md)).
*   `yelp` / `yelp_s10000`: Business reviews joined with business metadata and star ratings (subsampled to 10k aligned documents for parity with STM).

### Standardized Multi-Stage Data Pipeline

To eliminate confounding between models, **all text preprocessing is executed 100% in Python**. R scripts ingest preprocessed Parquet text directly, without performing any R-side stopword filtering or stemming.

1.  **Build Unified Raw/Interim Datasets:**
    ```bash
    uv run scripts/data_prep/build_datasets.py --dataset <dataset_name>
    ```

2.  **Align and Subsample (Yelp Only):**
    ```bash
    uv run scripts/data_prep/align_yelp_sample.py
    ```
    Trump is sampled after step 4 instead, from its finished embeddings, with `uv run python scripts/data_prep/sample_trump.py --n 25000`; then run step 5 with `--dataset trump_s25000`.

3.  **Preprocess Text (Dual Representation):**
    Generates both `clean_text` (unstemmed, casing/syntax preserved for SentenceTransformers) and `clean_text_stemmed` (lowercased, NLTK stopwords removed, Snowball stemmed for classical BoW/STM models), enforcing strict row alignment:
    ```bash
    uv run scripts/data_prep/preprocess_datasets.py --dataset <dataset_name>
    ```

4.  **Generate Dual Embeddings:**
    Computes dense embeddings for both text columns using `all-MiniLM-L6-v2`:
    ```bash
    uv run scripts/data_prep/generate_embeddings.py --dataset <dataset_name> --columns clean_text clean_text_stemmed
    ```

5.  **Build BoW and STM Objects (R):**
    ```bash
    # Unstemmed representation
    Rscript scripts/r_scripts/build_bow.R --dataset <dataset_name> --text_col clean_text

    # Stemmed representation
    Rscript scripts/r_scripts/build_bow.R --dataset <dataset_name> --text_col clean_text_stemmed --output_suffix _stemmed
    ```

6.  **Automated Representation Pipeline (Windows PowerShell):**
    To regenerate all datasets and representations in a single pass:
    ```powershell
    .\scripts\pipelines\local_windows\build_representations.ps1
    ```

---

## Running Experiments

The primary entry point for experiment batches is `scripts/pipelines/slurm/queue_exp.sh`.
It delegates to `scripts/experiments/queue_exp.py`, which selects configurations and
submits SLURM workers. Python experiment workers execute `scripts/experiments/run_optimizer.py`;
STM uses its separate runner. Agents should use this batch workflow for experiment campaigns.

Active production experiments are organized by dataset:
- `experiments/<dataset>/`: Standard unstemmed runs (`<dataset>_standard_*.yaml`).
- `experiments/<dataset>_stemmed/`: Stemmed text runs (`<dataset>_stemmed_standard_*.yaml`).
- `experiments/archive/<dataset>/`: Archived optimization, ablation, and exploratory runs.

### Batch Execution (SLURM)

From the repository root on the cluster:

```bash
# Preview the FED batch without submitting jobs
bash scripts/pipelines/slurm/queue_exp.sh -d fed --dry-run

# Submit the FED batch
bash scripts/pipelines/slurm/queue_exp.sh -d fed

# Submit selected model categories across datasets, split by configuration and seed
bash scripts/pipelines/slurm/queue_exp.sh -d gadarian,anes -m baseline,mv_spectral --split

# Submit stemmed FED experiments
bash scripts/pipelines/slurm/queue_exp.sh -d fed --stemmed
```

Use `--help` for selection and resource options. `--dry-run` previews submission;
it does not train models.

### Individual Experiments

For an individual configuration, use `scripts/experiments/run_optimizer.py` directly.
It supports both fixed configurations and parameter grids:

```bash
uv run python scripts/experiments/run_optimizer.py --exp fed/fed_standard_mv_k_means

# Sampled training run with one seed (executes training)
uv run python scripts/experiments/run_optimizer.py --exp trump/trump_standard_baseline --sample 500 --single-seed
```

You may omit the directory prefix and `.yaml` extension if the configuration name is unique.

### Representation Stop Words Removal (`--remove-rep-stopwords`)

By default, BERTopic's c-TF-IDF representation layer removes English stop words via `CountVectorizer(stop_words="english")` (`--remove-rep-stopwords`). This ensures extracted topic keywords are informative without altering natural sentence structure in the transformer embeddings. To retain stop words in topic representations, pass `--keep-rep-stopwords`.

### Running STM Baseline Experiments

To run Structural Topic Model baselines via R:
```bash
# Standard unstemmed STM
uv run python scripts/experiments/run_stm.py --exp fed/fed_standard_stm

# Stemmed STM
uv run python scripts/experiments/run_stm.py --exp fed_stemmed/fed_standard_stm
```

STM trains on the same preprocessing level as the models it is compared against: configs with `text_col: clean_text_stemmed` load `<dataset>_stemmed_stm_data.rds` and `<dataset>_stemmed_bow.parquet`, and all others load the unstemmed files.

### Running Hyperparameter Optimization

Optimization configurations specify parameter search spaces as lists (e.g., `n_clusters: [30, 50, 80]`) and are located in `experiments/archive/<dataset>/`:
```bash
uv run python scripts/experiments/run_optimizer.py --exp archive/yelp/yelp_opt_mv_spectral
```

---

## Results Management & Analysis

### Result Type Separation

Experiments produce results across three isolated preprocessing regimes:
1.  `standard`: Models operating on unstemmed `clean_text`.
2.  `stemmed`: Models operating on stemmed, stopword-removed `clean_text_stemmed`.
3.  `no_stopword_removal`: Legacy baseline models evaluated before stopword filtering.

Analysis utilities support the `--result-type` parameter (`standard`, `stemmed`, `no_stopword_removal`, or `all`) to prevent metrics from being cross-contaminated.

### Merging Results & Single-Pass Archival

Individual experiment runs generate raw CSVs in `results/` and topic JSON files in `output/`. Consolidate these into unified dataset files using:
```bash
uv run python scripts/analysis/merge_results.py
```
This script performs single-pass consolidation:
- Merges latest runs into `results/<dataset>_<type>_merged.csv` and `output/<dataset>_<type>_merged.json`.
- Pools all contributing and superseded raw run files into timestamped ZIP archives (`results/archive/` and `output/archive/`).
- Deletes unmerged raw files from disk, avoiding orphaned artifacts.

### Scraping Best Models & Generating LaTeX Tables

To extract top-performing models per metric and generate publication-ready LaTeX tables:
```bash
# Display best models for a dataset
uv run python scripts/analysis/find_best_models.py --dataset fed --result-type standard

# Export formatted LaTeX table
uv run python scripts/analysis/find_best_models.py --dataset fed --result-type standard --latex tables/fed_table.tex
```

### Noise Coverage Calculation

To evaluate and format HDBSCAN noise coverage into LaTeX:
```bash
uv run python scripts/analysis/calculate_noise_coverage.py --dataset fed --result-type standard --latex
```

### Automated Results Pipeline (Windows PowerShell)

To scrape, merge, and export all LaTeX tables and figures across all result types to the dissertation output directory:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/pipelines/local_windows/get_results.ps1 -Release
```

---

## Visualizing Results

The dashboard defaults to primary models and supports family, role, and reference
baseline filters. Model priorities and baseline–ablation relationships are stored
in [`config/model_catalog.yaml`](config/model_catalog.yaml); see the
[model catalog guide](docs/model_catalog.md) for the classification and editing rules.

The project includes an interactive Streamlit dashboard to explore and compare experiment metrics:
```bash
uv run streamlit run scripts/dashboard.py
```

The dashboard enables:
*   Filtering by dataset, model type, date, and preprocessing regime (`standard`, `stemmed`, `no_stopword_removal`).
*   Direct comparison across coherence (`c_v`, `u_mass`), diversity (`irbo`), and outlier metrics.
*   Interactive scatter plots and automated highlighting of best models.

---

## Running Tests

The unit and integration test suite covers builders, models, config inheritance, the experiment queue, provenance, and merge pipelines:
```bash
uv run pytest
```

---

## Project Structure

```
├── config/                    # Model catalog and RQ1 comparison-edge registry
├── data/                      # Raw, interim, and processed datasets (.parquet, .rds)
├── docs/                      # Current guides, decisions, and proposals
│   ├── REPOSITORY_ISSUES.md   # Open issue tracker
│   ├── experiments_summary.md # Experimental campaign and model taxonomy
│   ├── model_catalog.md       # Model priorities and baseline–ablation mapping
│   ├── pairwise_*.md          # Pairwise comparison proposal (partially implemented)
│   ├── project_structure.md   # Script-level directory guide
│   ├── ...                    # Preprocessing, stopwords, results separation, merging
│   └── archive/               # Completed plans and resolved incidents (historical only)
├── experiments/               # Experiment configuration files
│   ├── anes/                  # Active standard ANES configs
│   ├── fed/                   # Active standard FED configs
│   ├── gadarian/              # Active standard Gadarian configs
│   ├── trump/                 # Active standard Trump configs
│   ├── yelp/                  # Active standard Yelp configs
│   ├── *_stemmed/             # Active standard stemmed configs
│   ├── datasets/              # Base dataset definitions (inherited via 'extends')
│   └── archive/               # Archived optimization and exploratory configs
├── models/                    # Serialized model artifacts
├── notebooks/                 # Jupyter & Marimo exploratory notebooks
├── output/                    # Qualitative topic representations (.json)
├── results/                   # Metric evaluation results (.csv) and archives (.zip)
├── scripts/                   # Utility scripts and execution pipelines
│   ├── data_prep/             # Ingestion, preprocessing, embeddings, sampling
│   ├── experiments/           # Experiment runners, optimizer, STM coordinator
│   ├── analysis/              # Results merge, best models scraper, noise coverage
│   ├── pipelines/             # Local Windows (.ps1), Linux (.sh), and SLURM runners
│   ├── r_scripts/             # R scripts for Bag-of-Words and STM training
│   └── dashboard.py           # Streamlit results dashboard
├── src/                       # Core Python library
│   ├── builders/              # Dataset-specific ingestion builders
│   ├── comparisons/           # Matched baseline-vs-ablation statistics for the dashboard
│   ├── append_umap.py         # AppendUMAP dimension reduction wrapper
│   ├── data.py                # Dataset loading and splitting
│   ├── decoupled_kmeans.py    # Decoupled Multi-View K-Means implementations
│   ├── decoupled_spectral.py  # Decoupled Multi-View Spectral clustering
│   ├── document_assignments.py # Per-run document-topic assignment exports
│   ├── embeddings.py          # SentenceTransformers embedding generation
│   ├── evaluation.py          # Metric calculations (c_v, u_mass, irbo)
│   ├── experiment_queue.py    # Experiment queue orchestration
│   ├── experiment_tracker.py  # Experiment run tracking and persistence
│   ├── logger_config.py       # Centralized logging configuration
│   ├── make_table.py          # Great Tables & LaTeX table generators
│   ├── model_catalog.py       # Loader/validator for config/model_catalog.yaml
│   ├── models.py              # BERTopic, Multi-View, and TriTopic integrations
│   ├── mvc_wrapper.py         # Multi-View Clustering wrappers
│   ├── optimizer.py           # Hyperparameter optimization engine
│   ├── processing.py          # Standardized text cleaning and stemming
│   ├── results_analysis.py    # Results parsing and model type classification
│   ├── run_provenance.py      # Effective-settings capture and run manifests
│   ├── training.py            # Training routines
│   ├── utils.py               # Config loading and helper utilities
│   ├── verification.py        # Config and dataset verification tools
│   └── visualization.py       # Plotly chart generators
├── tables/                    # Generated LaTeX and Great Tables outputs
├── tests/                     # Comprehensive pytest test suite
└── pyproject.toml             # Project dependencies and configuration
```

---

## Core Technologies

*   **BERTopic:** Modular topic modeling framework.
*   **mvlearn:** Multi-view learning algorithms (Multi-View K-Means, Multi-View Spectral, Co-regularized Spectral).
*   **FastTriTopic / TriTopic:** Multi-modal graph topic modeling integrating text embeddings and document metadata via sparse graph laplacians with vectorized coordinate construction.
*   **Structural Topic Model (STM) / R (`quanteda`, `stm`):** Semi-parametric topic modeling incorporating document covariates.
*   **SentenceTransformers:** Contextual text representation models (default: `all-MiniLM-L6-v2`).
*   **OCTIS & gensim:** Coherence evaluation metrics (`c_v`, `u_mass`).
*   **Polars & PyArrow:** High-performance tabular data manipulation and Parquet storage.
*   **Great Tables & Plotly:** Publication-grade LaTeX/HTML tables and interactive figures.
*   **uv:** Fast Python packaging and project management.
