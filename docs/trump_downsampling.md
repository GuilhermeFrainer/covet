# Trump Downsampling (`trump_s25000`)

This document records how the Trump corpus is downsampled for the heaviest models, why the sample is drawn *after* preprocessing (unlike Yelp), and how sampled results are kept apart from full-corpus results.

---

## 1. Motivation

The full Trump corpus has 55,251 documents. Most models run on it, but the heaviest ones do not finish within cluster limits. A fixed 25,000-document sample (about 45% of the corpus) is added as a separate dataset, `trump_s25000`. The full-corpus configs and results stay untouched; whether the sample replaces Trump in reporting is decided later.

The sample size is a parameter: `sample_trump.py --n <size>` produces `trump_s<size>` with the same pipeline.

---

## 2. Sampling Rules (Shared with Yelp)

- **Drawn once, not per run.** The sample is materialized into its own files. Experiment configs point at those files and set no `sample_size`, so every model and seed trains on identical documents.
- **First fixed seed only.** Sampling uses `36201624`, the first entry of every `random_state` list, through `sample_from_lf` in [src/data.py](../src/data.py), the same helper Yelp uses.
- **Separate artifacts.** The sample gets its own Parquet and RDS files:

| File | Consumer |
| :--- | :--- |
| `data/processed/trump_s25000_embeddings.parquet` | Python models ([run_optimizer.py](../scripts/experiments/run_optimizer.py)) |
| `data/interim/trump_s25000_processed.parquet` | [build_bow.R](../scripts/r_scripts/build_bow.R) input |
| `data/processed/trump_s25000_stm_data.rds`, `trump_s25000_bow.parquet` | Unstemmed STM |
| `data/processed/trump_s25000_stemmed_stm_data.rds`, `trump_s25000_stemmed_bow.parquet` | Stemmed STM |

Rows are sorted by `index` after sampling so the file keeps the corpus's original order.

### Regenerating

```bash
uv run python scripts/data_prep/sample_trump.py --n 25000
Rscript scripts/r_scripts/build_bow.R --dataset trump_s25000 --text_col clean_text
Rscript scripts/r_scripts/build_bow.R --dataset trump_s25000 --text_col clean_text_stemmed --output_suffix _stemmed
```

[build_representations.ps1](../scripts/pipelines/local_windows/build_representations.ps1) runs these steps right after `trump`.

---

## 3. Decision: Sample After Preprocessing and Embedding

Yelp samples the raw interim reviews ([sample_yelp_interim.py](../scripts/data_prep/sample_yelp_interim.py)) and then preprocesses and embeds the 10k sample. Trump instead samples rows of the finished `trump_embeddings.parquet` with [sample_trump.py](../scripts/data_prep/sample_trump.py), then rebuilds only the R bag-of-words objects from the sampled rows.

### Why this is equivalent

Trump preprocessing in [src/processing.py](../src/processing.py) is per-document: cleaning, stopword removal, and stemming depend only on each document's own text. The pipeline enables no corpus-level step (no deduplication, no token chunking), and Trump has one row per document (`index == id`, 55,251 unique values) with no empty `clean_text` or `clean_text_stemmed`. SentenceTransformer embeddings are also per-document. A document therefore receives the same `clean_text`, `clean_text_stemmed`, and embeddings whether it is processed alone or inside the full corpus, so sampling before or after these steps yields the same data.

Covariate scaling and encoding are not baked into the files; they are fitted at load time in `process_metadata`, so they are fitted on the sample either way.

### Why this is preferable for Trump

- **Strict subset of the full corpus.** Each sampled row is identical to its full-corpus row (verified for text, identifiers, dates, and embeddings), so sample and full-corpus runs can be compared document by document if needed.
- **No re-embedding.** The existing embeddings are reused, so the sample cannot drift from a changed model version or batch configuration.
- **Row alignment is inherited.** The full file already satisfies the rule that both text columns are non-empty, so no filter is reapplied to one column alone. The script still verifies this and fails otherwise.

### Why Yelp does it the other way

The full Yelp corpus is about 16 GB and was chunked during earlier preprocessing, so preprocessing and embedding the whole corpus just to sample it is wasteful, and chunk-level sampling breaks the document unit. Neither constraint applies to Trump.

### What is still rebuilt

The bag-of-words and STM objects are rebuilt from the sampled rows. Subsetting the full-corpus RDS would keep vocabulary terms that never occur in the sample, which changes the STM input: the stemmed vocabulary is 24,150 terms for the sample against 39,818 for the full corpus.

### Sample profile

All 25,000 documents are kept by the RDS conversion. Every binary covariate (`is_retweet`, `is_deleted`, `is_flagged`) keeps both levels, so STM factor terms remain estimable. `device` keeps 18 of its 20 levels; the two dropped levels each had a single document in the full corpus.

---

## 4. Keeping Results Separate

Yelp's sample is copied onto `yelp_embeddings.parquet` on the cluster ([slurm_job.sh](../scripts/experiments/slurm_job.sh)), so its runs are labeled `yelp`. That is intentional: the sample *is* the Yelp result. Trump does not follow that pattern, because sampled and full-corpus results must never pool (see [results_separation.md](results_separation.md)).

`trump_s25000` is therefore a distinct dataset name end to end:

- Configs live in `experiments/trump_s25000/` and `experiments/trump_s25000_stemmed/`, extending `experiments/datasets/trump_s25000{,_stemmed}.yaml`. They were generated from the Trump configs; the stemmed set is registered in [generate_stemmed_configs.py](../scripts/data_prep/generate_stemmed_configs.py).
- `dataset_name` in results comes from the `dataset_path` file stem, so runs record `trump_s25000`, and document-assignment exports land under `output/document_assignments/trump_s25000/`.
- `DATASET_ALIASES` in [src/results_analysis.py](../src/results_analysis.py) is the single list of names reported under another name. `yelp_s10000 → yelp` is the only sample alias. [merge_results.py](../scripts/analysis/merge_results.py), [find_best_models.py](../scripts/analysis/find_best_models.py), the dashboard, and the analysis helpers use it instead of stripping any `_s<digits>` suffix, so results merge into `results/trump_s25000_<type>_merged.csv`.

The sample is not in any default dataset list for queueing, result export (`get_results.ps1 -Datasets`), or cross-dataset comparisons. Pass it explicitly, e.g. `queue_exp.sh -d trump_s25000`.

---

## 5. STM Preprocessing Parity

STM originally always trained on stemmed text, since stemming is the conventional preprocessing for bag-of-words models. A reviewer pointed out that this makes STM a different preprocessing condition from the embedding-based models, so each STM run now uses the same preprocessing level as the models it is compared against. [run_stm.py](../scripts/experiments/run_stm.py) selects `<dataset>_stemmed_stm_data.rds` and `<dataset>_stemmed_bow.parquet` when `text_col` is `clean_text_stemmed`, and the unstemmed files otherwise. This applies to `trump_s25000` like every other dataset.

---

## 6. Transferring to the Cluster

No script pushes data to the cluster ([sync_results.ps1](../scripts/pipelines/local_windows/sync_results.ps1) only pulls results). From PowerShell on the local machine:

```powershell
wsl rsync -avP -e "ssh -i ~/.ssh/pcad_ufrgs" /mnt/d/CA-BERTopic/data/processed/trump_s25000_* gdsfrainer@gppd-hpc.inf.ufrgs.br:~/ca_bertopic/data/processed/
```

Python jobs need only the embeddings file; the queue's generic copy step (`{dataset}_embeddings.parquet`) picks it up with `-d trump_s25000`. STM needs the RDS and BoW files.
