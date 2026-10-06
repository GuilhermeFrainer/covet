# Text Preprocessing & Dataset Regeneration Guide

This document explains the text preprocessing architecture, the two standardized text representation versions (`clean_text` and `clean_text_stemmed`), why Python-only preprocessing is strictly used across both Python and R, and how to regenerate all datasets in the project.

---

## 1. Experimental Motivation & Reviewer Justification

> **Reviewer 1 Note:** Previous iterations of topic modeling experiments suffered from a confounding factor: BERTopic/CAST models operated on unstemmed clean text, while R-based Structural Topic Models (STM) applied R-side tokenization, stopword removal, and stemming via `quanteda`. This mismatch in text representation meant performance differences between models could stem from preprocessing disparities rather than model architecture.

To eliminate this confounder, **all text preprocessing (cleaning, stopword removal, and stemming) is executed 100% in Python**. R scripts ingest preprocessed text strings directly from Python and perform no R-side stopword filtering or stemming. Both BERTopic/CAST and STM run on both preprocessed versions for complete, un-confounded experimental comparison.

---

## 2. Standardized Preprocessing Versions

In `src/processing.py`, the preprocessing pipeline standardizes raw text into two core variants:

### Version 1: Lightest Preprocessing (`clean_text`)
* **Objective:** Clean raw noise while preserving original casing and punctuation for sentence transformer models (e.g., `all-MiniLM-L6-v2`).
* **Transformations Applied:**
  1. **URL Removal (`remove_urls`)**: Strips `https?://\S+`, `www\.\S+`, `t.co/\S+`, and residue patterns.
  2. **Number Removal (`remove_numbers`)**: Strips digits (`\d+`).
  3. **Dataset Artifact Removal (`remove_artifacts`)**: Strips platform noise (e.g. `covfefe`).
  4. **Whitespace Condensation**: Collapses multiple whitespace characters and strips leading/trailing spaces.
* **Preserved:** Original word casing, punctuation, and all stopwords are preserved intact.

### Version 2: Stemmed & Stopword-Removed (`clean_text_stemmed`)
* **Objective:** Produce a stemmed, noise-free bag-of-words representation for topic modeling.
* **Transformations Applied:**
  1. Built directly on top of Version 1 (`clean_text`).
  2. **Lowercasing & Punctuation Stripping**: Text is lowercased and all non-alphanumeric punctuation (`[^\w\s]`) is removed.
  3. **Stopword Filtering**: Removes NLTK English stopwords (*"the"*, *"and"*, *"in"*, etc.).
  4. **Snowball Stemming**: Stems remaining tokens using NLTK's `SnowballStemmer("english")`.
  5. **String Re-assembly**: Tokens are joined into a single space-separated string.

---

## 3. Strict Row Alignment & Empty Document Filtering

### Preventing Metric and Model Failures
Coherence evaluation frameworks (e.g., OCTIS) require non-empty token lists per document. Documents containing only links, digits, or stopwords become empty (`""`) after preprocessing. 

### Row Alignment & Asymmetric Empty Warning
If a short document becomes empty in `clean_text_stemmed` but was non-empty in `clean_text` (or vice-versa), filtering it out from only one column would corrupt row index alignment across representations.

To solve this:
* A document is retained **only if both `clean_text` and `clean_text_stemmed` are non-empty strings**.
* If asymmetric empty rows are detected, `src/processing.py` logs an explicit `WARNING` detailing the count of dropped rows to verify alignment integrity.

---

## 4. Manual Dataset Regeneration Steps (CLI)

Each dataset can be regenerated manually using the Python and R CLI scripts:

### A. Preprocessing & Embedding Generation (Python)
1. **Build Raw Dataset:**
   ```bash
   uv run scripts/data_prep/build_datasets.py --dataset <dataset_name>
   ```
2. **Preprocess (Creates `clean_text` and `clean_text_stemmed`):**
   ```bash
   uv run scripts/data_prep/preprocess_datasets.py --dataset <dataset_name>
   ```
3. **Generate Dual Embeddings (`clean_text_embedding` and `clean_text_stemmed_embedding`):**
   ```bash
   uv run scripts/data_prep/generate_embeddings.py --dataset <dataset_name> --columns clean_text clean_text_stemmed
   ```

### B. BoW & STM Representation Generation
Python also tokenizes the bag-of-words, with the same analyzer BERTopic uses for c-TF-IDF topic words (`CountVectorizer(stop_words="english")` on punctuation-stripped text). R only splits these tokens on whitespace, so it performs no lowercasing, filtering, stopword removal, or stemming. STM and the neural models therefore share one vocabulary; see [stm.md](stm.md).

1. **Tokens for both text columns:**
   ```bash
   uv run scripts/data_prep/build_bow_tokens.py --dataset <dataset_name>
   ```
   *Output:* `data/interim/<dataset_name>_bow_tokens.parquet`, built from the embeddings file so rows and metadata match the neural models.

2. **Unstemmed BoW & STM (`clean_text`):**
   ```bash
   Rscript scripts/r_scripts/build_bow.R --dataset <dataset_name> --text_col clean_text
   ```
   *Outputs:* `data/processed/<dataset_name>_bow.parquet` and `data/processed/<dataset_name>_stm_data.rds`.

3. **Stemmed BoW & STM (`clean_text_stemmed`):**
   ```bash
   Rscript scripts/r_scripts/build_bow.R --dataset <dataset_name> --text_col clean_text_stemmed
   ```
   *Outputs:* `data/processed/<dataset_name>_stemmed_bow.parquet` and `data/processed/<dataset_name>_stemmed_stm_data.rds`.

Documents left without tokens (only stopwords) are dropped from STM; at most 25 per corpus.

---

## 5. Automated Pipeline Script

We provide a PowerShell pipeline script to build and generate representations for all datasets in one command:

```powershell
# Default: Runs on all datasets using the 10k sampled Yelp dataset (yelp_s10000)
.\scripts\pipelines\local_windows\build_representations.ps1

# Optional: Generate for full 6.9M Yelp dataset if explicitly needed
.\scripts\pipelines\local_windows\build_representations.ps1 -FullYelp
```

By default, the script generates both unstemmed and stemmed/stopword-removed versions for all datasets (`fed`, `anes`, `gadarian`, `trump`, and `yelp_s10000`).
