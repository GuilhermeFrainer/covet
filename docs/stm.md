# STM Baseline

The Structural Topic Model (STM) is the classical covariate-aware baseline. R trains it; Python evaluates it with the same metrics as every other model. This guide covers how STM is kept comparable to the embedding-based models and how it runs on the cluster, where R is not installed.

---

## 1. Same Documents, Same Tokens

A reviewer asked that STM and the neural models see identically preprocessed text. Two layers are shared:

1. **Text.** Both families read `clean_text` or `clean_text_stemmed` from `data/processed/<dataset>_embeddings.parquet`, produced by the Python pipeline in [preprocessing.md](preprocessing.md). STM's metadata comes from the same rows.
2. **Tokens.** BERTopic variants take topic words from a `CountVectorizer(stop_words="english")` fitted on punctuation-stripped text, and coherence is scored on that same tokenization ([src/training.py](../src/training.py)). STM's bag-of-words is built with the identical function, `representation_tokens` in [src/evaluation.py](../src/evaluation.py), so STM's vocabulary is the vocabulary neural topic words come from.

For the neural models this tokenizer only shapes the topic words and the coherence corpus; their embeddings still see full sentences. For STM the bag-of-words is both input and representation, so for STM it is preprocessing proper. The STM authors also recommend removing stopwords: with them, the most probable words of every topic are function words.

Paper sentence: *STM's vocabulary is constructed with the same tokenizer and stop-word list as the c-TF-IDF representation of the embedding-based models.*

No rare-word trimming is applied (`prepDocuments(lower.thresh=…)` is common in STM tutorials), because the neural vectorizer keeps every word (`min_df=1`).

### Documents without tokens

A document made only of stopwords has no bag-of-words and is dropped from STM. The neural models train on it, but it contributes no tokens to their coherence corpus either ([src/training.py](../src/training.py) drops empty tokenized documents), so both families are scored against the same reference corpus.

| Dataset | Unstemmed dropped | Unstemmed vocabulary | Stemmed dropped | Stemmed vocabulary |
| :--- | ---: | ---: | ---: | ---: |
| anes (2,055) | 4 | 1,400 | 1 | 1,213 |
| fed (5,446) | 0 | 11,102 | 0 | 7,356 |
| gadarian (341) | 3 | 1,304 | 0 | 1,039 |
| trump (55,251) | 22 | 47,553 | 25 | 39,021 |
| trump_s25000 (25,000) | 10 | 29,640 | 12 | 23,718 |
| yelp_s10000 (10,205) | 0 | 28,606 | 0 | 20,555 |

### Building the inputs (local machine)

```bash
uv run python scripts/data_prep/build_bow_tokens.py --dataset <dataset>
Rscript scripts/r_scripts/build_bow.R --dataset <dataset> --text_col clean_text
Rscript scripts/r_scripts/build_bow.R --dataset <dataset> --text_col clean_text_stemmed
```

[build_bow_tokens.py](../scripts/data_prep/build_bow_tokens.py) writes `data/interim/<dataset>_bow_tokens.parquet` with one token column per text column. [build_bow.R](../scripts/r_scripts/build_bow.R) only splits those tokens on whitespace and writes `data/processed/<dataset>[_stemmed]_stm_data.rds` (STM input) and `<dataset>[_stemmed]_bow.parquet` (the same documents and metadata, read by Python). [build_representations.ps1](../scripts/pipelines/local_windows/build_representations.ps1) runs these steps for every dataset. Copy the outputs to `~/ca_bertopic/data/processed/` on the cluster.

---

## 2. Training and Evaluation

[run_stm.py](../scripts/experiments/run_stm.py) runs each (K, seed) of a config:

1. [train_stm.R](../scripts/r_scripts/train_stm.R) fits the model with spectral initialization and the config's `prevalence_formula`, then exports beta, theta, the vocabulary, the trained documents' `index`, and the training time.
2. Python checks that R's documents match the BoW parquet row for row, then scores the topics.

| Step | Rule |
| :--- | :--- |
| Topic words | The 10 most probable words of each topic (beta), skipping purely numeric words as BERTopic topics do |
| Coherence corpus | The BoW documents, i.e. the neural models' tokenization of the same texts |
| Outliers | 0; STM assigns every document |
| Metadata alignment | AMI between each document's most probable topic and each raw covariate (see below) |

Results, qualitative topics, and assignments are written like the neural runs: `results/<exp>_<tag>[_m<idx>]-<timestamp>-<seed>.csv`, `output/<same>.json`, and `output/document_assignments/<dataset>/<run_uid>/`. Rows carry `campaign_id: stm_shared_vocab_v1`, `r_runner`, and `stm_image`.

`--r-runner local` uses the local R installation (with renv); `--r-runner docker` runs R in the STM image. The script exits non-zero if any run fails.

### One seed per configuration

Spectral initialization is deterministic: two seeds produce identical beta and theta (verified on Gadarian, K = 10). STM configs therefore list a single seed, and each K is one run. Report STM as deterministic rather than as a seed average.

### Metadata alignment (AMI) caveat

STM assigns each document a topic *mixture*. AMI needs one label per document, so each document takes its most probable topic (`assignment_semantics: argmax_theta`; its proportion is kept as `topic_strength`). This discards the rest of the mixture, so STM's AMI is not strictly comparable to the hard-clustering models'. AMI is descriptive only; note this in the paper.

### Numerical reproducibility

The image pins R 4.5.0 and the packages in `renv.lock` (`stm` 1.3.8, `data.table` 1.18.4, `quanteda` 4.4). Container runs are bit-identical across repetitions. A local Windows run of the same model gives slightly different topics, most likely from different linear-algebra libraries in the spectral initialization. Report only container runs.

---

## 3. Running on the Cluster

PCAD has no R. STM runs in a container with rootless Podman (`docker` is an alias for `podman`), and evaluation runs on the host with the project's `uv` environment.

### The image

[Dockerfile.stm](../Dockerfile.stm) holds R and the packages `train_stm.R` needs: `stm` (with its dependencies, including `data.table` and `quanteda`) and `optparse`. Packages are prebuilt binaries from a dated Posit Package Manager snapshot, so nothing compiles. Code and data are mounted at run time, so the image needs no build context and never changes when the code does. Compressed, the image is about 370 MB.

Build it locally and upload it:

```powershell
.\scripts\pipelines\local_windows\build_stm_image.ps1 -Version 0.2.0
```

This builds `cast:stm-lite-v0.2.0` for `linux/amd64` and copies `cast_stm-lite-v0.2.0.tar.gz` to `~/docker_images/` on PCAD. When the version changes, update `STM_IMAGE` in [src/experiment_queue.py](../src/experiment_queue.py).

PCAD stores images per user and per node, so [slurm_stm_job.sh](../scripts/experiments/slurm_stm_job.sh) loads the tarball on each node the first time a job lands there, under a lock so concurrent jobs do not load it twice. To check a node by hand: `sbatch scripts/pipelines/slurm/slurm_hello.sh`.

Rootless Podman maps the container's root user to the job's user, so files the container writes belong to you. Do not pass `--user`.

### Queueing

STM is part of the regular queue:

```bash
bash scripts/pipelines/slurm/queue_exp.sh -d fed -m stm --dry-run
bash scripts/pipelines/slurm/queue_exp.sh -d gadarian,anes -m stm --split
bash scripts/pipelines/slurm/queue_exp.sh -d fed -m stm --stemmed
```

STM jobs default to 1 CPU and 16 GB (`--cpus`, `--mem`, and `--time` override them). In split mode, each config's five K values become five jobs; indices beyond a config's run count are not submitted. STM is skipped with `--keep-rep-stopwords`, since its bag-of-words exists only with stopwords removed.

Quick check before a campaign:

```bash
uv run python scripts/experiments/run_stm.py --exp gadarian/gadarian_standard_stm --model 1 --r-runner docker
```
