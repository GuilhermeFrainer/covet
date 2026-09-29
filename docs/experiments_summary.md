# Experiments and Results Summary: CA-BERTopic (CAST)

This document provides a comprehensive summary of the experimental campaign, model architectures, evaluation metrics, and statistical tests in the **CA-BERTopic** (Covariate-Assisted / Multi-View Topic Modeling) framework.

---

## 1. Experimental Setup & Benchmark Scope

### 1.1 Benchmark Datasets
The benchmark spans five empirical datasets covering diverse document lengths, vocabularies, and covariate modalities:

| Dataset | Observations ($N$) | Domain & Text Style | Metadata Covariates Incorporated |
| :--- | :---: | :--- | :--- |
| **`anes`** | 2,055 | Short open-ended survey responses | Political party affiliation, ideological self-placement, demographic variables. |
| **`fed`** | 5,446 | Long-form FOMC statements and minutes | Macroeconomic indicators (GDP monthly/yearly + lags, CPI monthly/yearly + lags, Fed funds rate + lags, unemployment + lags), Fed chair, US president, political party. |
| **`gadarian`** | 341 | Short survey responses on public health and threat | Experimental anxiety framing, participant demographics (age, gender, education, party). |
| **`trump`** | 55,251 | Social media posts (Twitter/X) | Engagement metrics (retweets, favorites), temporal timestamps/calendar dates. |
| **`yelp`** (`yelp_s10000`) | 10,205 | Consumer business reviews (subsampled/aligned) | Star ratings (1–5), review count, business categories, user check-in metrics. |

---

### 1.2 Seeds, Topic Configurations & Experimental Scale
* **Random Seeds ($3$ fixed seeds)**: `36201624`, `62613654`, `57116123`.
* **Topic Counts ($K$ / $nr\_topics$ / $n\_clusters$)**: $5$ evaluation blocks: **`10, 20, 30, 40, 50`** topics.
* **Runs per Complete Configuration**: $3\text{ seeds} \times 5\text{ topic counts} = \mathbf{15\text{ runs}}$ per model.
* **Text Embeddings**: Standardized dense semantic representations via `all-MiniLM-L6-v2` ($d=384$) generated in Python with strict row alignment.
* **Benchmark Size** (as of 2026-09-15): Over **2,945 individual model evaluations** consolidated into merged result files, plus recent decoupled multi-view experimental runs.

---

### 1.3 Preprocessing Conditions (Results Isolation)
Experiments are evaluated across three isolated preprocessing regimes:
1. **`standard` (Production)**: Clean unstemmed text (`clean_text`), casing/syntax preserved for SentenceTransformers, with representation stop words filtered at the c-TF-IDF layer (`CountVectorizer(stop_words="english")`).
2. **`stemmed`**: Lowercased, NLTK stopwords removed, Snowball stemmed text (`clean_text_stemmed`).
3. **`no_stopword_removal`**: Legacy baseline runs conducted prior to representation stopword filtering (stopwords retained).

---

## 2. Model Architectures & Experimental Taxonomies

The evaluated models span eight architectural paradigms:

### 2.1 Unimodal / Text-Only Baselines (No Covariates)
* **`baseline`**: Standard BERTopic (Cosine UMAP $\to$ HDBSCAN $\to$ c-TF-IDF).
* **`k_means`**: BERTopic with UMAP $\to$ K-Means.
* **`umap_spectral`**: BERTopic with UMAP $\to$ Spectral Clustering.
* **`pca_k_means`**: BERTopic with PCA dimensionality reduction $\to$ K-Means.

### 2.2 Early Fusion (Concatenation)
* **`append_umap`** (naive): Horizontally concatenates scaled metadata directly to dense text embeddings before cosine UMAP $\to$ HDBSCAN $\to$ c-TF-IDF. The metadata block is not balanced against the unit-norm text block, so under cosine the text's weight in document similarity is about $1/(1+\lVert m\rVert^2)$. Measured on 2026-09-29, that was 0.13 (FED), 0.23 (Yelp), 0.31 (Trump), 0.36 (ANES) and 0.44 (Gadarian). In a single-seed Gadarian fit, the resulting topics reproduced the dataset's 14 metadata profiles exactly (NMI = 1.00). These runs are kept as the unweighted early-fusion reference, not as a balanced text+metadata model.
* **`append_umap_w000` … `append_umap_w050`** (weighted): Text embeddings are L2-normalized. Each block is divided by the square root of its mean pairwise squared distance (fitted on the training data), and the blocks are scaled by $\sqrt{1-w}$ and $\sqrt{w}$. Euclidean UMAP then sees $d^2 = (1-w)\,d_\text{text}^2/s_\text{text} + w\,d_\text{meta}^2/s_\text{meta}$, so $w$ is the metadata share of the average squared distance, and metadata levels are preserved. Grid $w \in \{0, 0.05, 0.1, 0.2, 0.3, 0.5\}$, dense at low $w$ because text neighbourhoods change fastest there. $w=0$ ignores metadata and serves as a parity check against `baseline`: same text neighbour ranking, with UMAP edge weights computed from Euclidean rather than cosine distances. Report the full curve rather than selecting a best $w$ per dataset.
* **`append_umap_mv_k_means`**: Append UMAP $\to$ Multi-View K-Means.
* **`append_umap_mv_spherical_k_means`**: Append UMAP $\to$ Multi-View Spherical K-Means.
* **`append_umap_mv_spectral` & `append_umap_mv_spectral_info0`**: Append UMAP $\to$ Multi-View Spectral.
* **`append_umap_mv_co_reg_spectral` & `append_umap_mv_co_reg_spectral_info0`**: Append UMAP $\to$ Co-Regularized Multi-View Spectral.

### 2.3 Manifold Alignment
* **`aligned_umap`**: Uses UMAP's `AlignedUMAP` to align relations across semantic text embeddings and metadata spaces $\to$ HDBSCAN.
* **`aligned_umap_mv_k_means`**: Aligned UMAP $\to$ Multi-View K-Means.
* **`aligned_umap_mv_spherical_k_means`**: Aligned UMAP $\to$ Multi-View Spherical K-Means.

### 2.4 Multi-View Clustering (Intermediate / Late Fusion)
Evaluates View 0 (Text Embeddings) and View 1 (Metadata Covariates):
* **`mv_k_means`**: Multi-View K-Means (Euclidean distance on all views).
* **`mv_spherical_k_means`**: Multi-View Spherical K-Means (Cosine distance on all views).
* **`mv_spectral` & `mv_spectral_info0`**: Multi-View Spectral Clustering with co-training graph Laplacians.
* **`mv_co_reg_spectral` & `mv_co_reg_spectral_info0`**: Co-regularized Multi-View Spectral Clustering (Kumar & Daume, enforces consensus across eigenvectors).
* **`mv_hdbscan`**: Multi-View HDBSCAN from the sibling `MV-HDBSCAN` package (per-view scaling, `max` fusion) with baseline-matched UMAP and `nr_topics` settings. Primary ablation of `baseline` in [`config/model_catalog.yaml`](../config/model_catalog.yaml).
* **`feature_stacking_hdbscan`**: HDBSCAN on standardized, horizontally stacked text and metadata views (`MV-HDBSCAN` package) with baseline-matched UMAP and `nr_topics` settings.

### 2.5 PCA + Multi-View Clustering
* **`pca_mv_k_means`**, **`pca_mv_spherical_k_means`**, **`pca_mv_spectral`**, **`pca_mv_co_reg_spectral`**: Linear dimensionality reduction on text embeddings before multi-view clustering.

### 2.6 Decoupled Distance Multi-View Models
Resolves metric mismatch (semantic text is angular/Cosine; metadata is magnitude/Cartesian Euclidean):
* **`mv_k_means_l2_norm`**: L2-normalized text view (monotonically equivalent to Cosine under Euclidean distance) + Euclidean metadata.
* **`mv_spectral_l2_norm`**: L2-normalized text view + Euclidean metadata for spectral Gaussian kernel.
* **`decoupled_mv_k_means`**: Explicit dual-metric Expectation-Maximization (spherical Cosine EM on text with unit hypersphere centroid projection + Euclidean EM on metadata).
* **`decoupled_mv_spectral`**: Modality-specific affinity graphs (Cosine kNN on text, Euclidean RBF on metadata) $\to$ co-training Laplacians.

### 2.7 Multi-Modal Hypergraph Topic Models
* **`tritopic`**: Tri-partite graph model connecting Documents, Words, and Covariates with spectral graph partitioning.
* **`fast_tritopic`**: Vectorized coordinate construction and sparse Laplacian solver for high-performance TriTopic.

### 2.8 Parametric / Econometric Baseline
* **`stm`**: Structural Topic Model (Roberts et al.) trained in R (`stm`, `quanteda`) using document-level prevalence covariates, evaluated via the exact same Python evaluation pipeline for benchmark parity.

---

## 3. Results and Statistical Tests Currently Available

### 3.1 Evaluated Metrics (Stored in `results/*_merged.csv`)
For every individual model run, the pipeline computes:
* **Topic Coherence**:
  * **$C_V$**: Sliding window (size 110) + NPMI + cosine similarity of context vectors. Scale: $[0, 1]$ (higher is better).
  * **$C_{\text{NPMI}}$**: Normalized Pointwise Mutual Information. Scale: $[-1, 1]$ (higher is better).
  * **$U_{\text{Mass}}$**: Intrinsic log-conditional co-occurrence probability over corpus texts. Scale: negative values (closer to $0$ is better).
* **Topic Diversity**:
  * **$\text{IRBO}$ (Inverted Rank-Biased Overlap)**: Mutual exclusivity of topic top words with geometric rank decay (penalizes top-ranked word overlaps). Scale: $[0, 1]$ (higher is better).
  * **$\text{Topic Diversity}$**: Percentage of unique words across top-10 words of all topics ($\frac{|\bigcup_k W_k|}{k \times 10}$). Scale: $[0, 1]$ (higher is better).
* **Topic–Metadata Alignment** (added 2026-09-29, [`src/metadata_alignment.py`](../src/metadata_alignment.py)):
  * **`meta_ami_mean`**: Adjusted mutual information (AMI) between each document's final topic and each **raw** covariate, averaged over the dataset's covariates. Numerics with more than 5 distinct values are split into quintile bins over the run's documents; other covariates are used as categories; noise (topic $-1$) counts as its own topic. AMI rather than NMI because NMI rises by chance with the number of topics. Scale: $\approx 0$ for unrelated partitions, $1$ for identical ones.
  * **Descriptive outcome, not "higher is better"**: topics that simply reproduce the metadata groups score $1$ (e.g., naive Append UMAP on Gadarian). Read it alongside coherence as a trade-off.
  * Per-covariate values are stored in each run's `output/document_assignments/<dataset>/<run_uid>/metrics.json` (`meta_ami_by_covariate`) and manifest (`metadata_alignment`). They're computed only for runs executed through `run_optimizer.py` with assignment export; STM is not covered yet.
  * **Earlier runs**: [`scripts/analysis/backfill_metadata_alignment.py`](../scripts/analysis/backfill_metadata_alignment.py) scores every exported run without retraining. It locates the source dataset by the SHA-256 in the manifest and writes the long-format sidecar `results/metadata_alignment_backfill.csv` (one row per run × covariate, keyed by `run_uid`). Runs without exported assignments can't be backfilled and must be rerun.
* **Clustering & Operational Metrics**:
  * **$\text{Outliers}$**: Count of documents assigned to topic $-1$ (in HDBSCAN models).
  * **$\text{Noise Coverage \%}$ & $\text{Clustered Coverage \%}$**: Document retention rates.
  * **$\text{Duration Seconds}$**: Wall-clock training latency.
  * Realized Topics ($n\_topics$) vs Requested Topics ($nr\_topics$ / $n\_clusters$).

### 3.2 Qualitative Outputs (Stored in `output/*_merged.json`)
For every topic of every model run:
* Top 50 keywords with c-TF-IDF representation scores.
* Descriptive topic label.
* Topic document frequency count.
* Full text of top representative documents.

### 3.3 Statistical Tests Executed

Following **Demšar (2006)** guidelines for comparing machine learning algorithms across multiple evaluation blocks ($N=5$ topic counts: $10, 20, 30, 40, 50$):

1. **Demšar All-vs-All Benchmark Analysis**:
   * **Omnibus Test**: Friedman test and Iman-Davenport $F_F$ test evaluating significance across algorithms per metric.
   * **Critical Difference (CD) & Significance Cliques**: Nemenyi post-hoc test grouping statistically indistinguishable models into cliques ($\alpha = 0.05$).
   * **Pairwise Delta Matrices**: Full $M \times M$ matrix of pairwise score differences ($\Delta = \text{Score}_{\text{row}} - \text{Score}_{\text{col}}$) with **paired exact Wilcoxon signed-rank tests** and **Holm-Bonferroni FWER step-down correction** ($\alpha = 0.05$).
   * *Output tables*: `tables/demsar_all_vs_all_fed_standard.*`, `tables/demsar_all_vs_all_yelp_standard.*`, and cross-dataset `tables/demsar_all_vs_all_fed_yelp_standard.*`.

2. **Demšar Preprocessing Delta Analysis**:
   * Measures statistically significant metric shifts across preprocessing conditions (`stemmed` vs `standard`, `no_stopword_removal` vs `standard`).
   * Paired exact Wilcoxon signed-rank tests with Holm-Bonferroni correction per metric.
   * *Output tables*: `tables/demsar_delta_fed_stemmed.*`, `tables/demsar_delta_yelp_stemmed.*`, `tables/demsar_delta_yelp_no_stopword_removal.*`.

3. **HDBSCAN Noise Coverage Summaries**:
   * Mean $\pm$ standard deviation of unclustered noise rates across models and datasets (`tables/hdbscan_noise_coverage.tex`).

4. **Interactive Dashboard & Automated Plots**:
   * Streamlit dashboard (`scripts/dashboard.py`) with dynamic filtering, Pareto frontiers (Coherence vs Diversity), Cleveland dot plots, parallel coordinates, and radar/star plots (`src/visualization.py`).

---

## 4. Candidate Metrics Available to Compute

Without retraining models, the existing artifacts (topic keywords, document-topic distributions, and document metadata) allow computing:

### 4.1 Covariate Alignment & Topic-Metadata Associations (Core to CA-BERTopic)
* **Topic Purity / Categorical Entropy**: Shannon entropy or Gini impurity of metadata classes (e.g., political party, Fed chair, star rating) within each topic.
* **Statistical Association with Continuous Covariates**: Mutual Information, ANOVA $F$-statistic, or Kruskal-Wallis tests between topic distributions and continuous covariates (e.g., GDP, inflation, ratings).
* **Topic Prevalence Regression ($R^2$ / Effect Estimation)**: Regressing document-topic probabilities on covariates (mirroring STM's `estimateEffect`).

### 4.2 Topic Stability & Reproducibility Across Seeds
* **Hungarian / Bipartite Topic Matching**: Bipartite matching between topic top-words across the 3 seeds (`36201624`, `62613654`, `57116123`) computing:
  * Jaccard similarity across matched topic pairs.
  * Rank-Biased Overlap (RBO) stability across seeds.

### 4.3 Additional OCTIS Metrics
* **$C_{\text{UCI}}$ Coherence**: Pointwise Mutual Information over sliding context windows.
* **Word Embedding Coherence (`WECoherenceCentroid`, `WECoherencePairwise`)**: Semantic cosine distance in dense embedding space.
* **Word Embedding Diversity (`WordEmbeddingsInvertedRBO`)**: Penalizes semantic synonymy across topics rather than only exact token overlap.
* **Kullback-Leibler Divergence & Log-Odds Ratio**: Topic-word distribution divergence.

### 4.4 Downstream / Extrinsic Task Evaluation
* **Covariate Prediction Accuracy**: Training a linear classifier/regressor using document-topic distribution vectors $\theta_d$ to predict metadata targets (e.g., star rating or interest rate policy).

### 4.5 Efficiency & Scalability
* Throughput (documents/sec), peak RAM/VRAM, and scaling curves as a function of corpus size $N$ (e.g., `FastTriTopic` vs `TriTopic`, decoupled EM vs naive append).
