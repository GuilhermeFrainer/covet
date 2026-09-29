# Pairwise comparisons: repository findings

Part of the [pairwise comparison proposal](pairwise_comparisons.md). Findings describe the inspection on **2026-09-16**; coverage and implementation may change later.

**User clarification, 2026-09-16:** the baseline is intended to use BERTopic defaults. The 2-versus-5 UMAP dimensionality difference is therefore a confirmed configuration bug, and the user will fix it. This supersedes the earlier suggestion that retaining a renamed 2-dimensional baseline might satisfy the intended experiment. Track current issues in [REPOSITORY_ISSUES.md](REPOSITORY_ISSUES.md).

## Current experiment representation

Experiments use YAML under [experiments/](../experiments). Dataset definitions are inherited through `extends`. [src/utils.py](../src/utils.py) merges dataset settings into `experiment`; model settings live under `model` or `models`.

The standard configurations specify three seeds:

```text
36201624, 62613654, 57116123
```

The requested topic grid is `10, 20, 30, 40, 50`. The [optimizer](../src/optimizer.py) expands these into 15 runs per model and dataset. A numeric suffix in a model name identifies a **configuration-combination index**, not necessarily the requested topic count. Matching must use the actual parameter or an unambiguous resolved grid mapping.

The primary batch entry point is [queue_exp.sh](../scripts/pipelines/slurm/queue_exp.sh). Python workers and individual configurations use [run_optimizer.py](../scripts/experiments/run_optimizer.py). Models are constructed in [src/models.py](../src/models.py); fitting and evaluation occur in [src/training.py](../src/training.py).

## Current result representation

| Meaning | Current representation |
| --- | --- |
| Dataset | `dataset_name` |
| Experiment/configuration | `experiment_id` |
| Model/run | `model_name` |
| Training seed | `random_state` |
| Requested topics | `n_clusters` or `nr_topics`, depending on architecture |
| Realized non-outlier topics | `n_topics` |
| Document count | `n_observations` |
| Algorithms | `clustering_algo`, `dim_red_algo` |
| Metrics | `u_mass`, `c_v`, `c_npmi`, `irbo`, `topic_diversity` |
| Other outcomes | `duration_seconds`, `outliers` |
| Preprocessing condition | Mainly inferred from filenames and experiment identifiers |
| Run provenance | Timestamps and some varied hyperparameters |

CSV results live in [results/](../results), including merged dataset files and individual runs. Qualitative topic outputs live in [output/](../output). The existing merge utility also archives contributing and superseded artifacts.

The inspected runners do **not** persist complete resolved configurations, configuration hashes, document/sample fingerprints, or explicit run-status records. Failed training attempts are logged and omitted from successful-result CSVs. The new pipeline must distinguish:

- Missing with unknown cause.
- Documented failure.
- Successful run with valid metrics.

Equal row counts or matching current YAML files cannot establish historical sample/configuration parity.

## Reusable analysis code and limitations

[src/results_analysis.py](../src/results_analysis.py) already contains dataset-level aggregation, Wilcoxon tests, Holm adjustment, and all-versus-all comparisons. [src/experiment_tracker.py](../src/experiment_tracker.py) provides experiment discovery, condition classification, and coverage infrastructure. Table/plot conventions exist in [src/make_table.py](../src/make_table.py) and [src/visualization.py](../src/visualization.py).

However, the existing delta analysis:

- Compares the same model across preprocessing conditions.
- Averages available rows before matching, so unequal run grids can be hidden.
- Excludes PCA and K-means by default.
- Can infer topic indices from model names and available fields.
- Does not enforce the proposed configuration/sample parity requirements.

It should not become the new pipeline unchanged. Reuse narrowly suitable helpers, especially Holm and export conventions, while introducing explicit registered node matching.

The current Wilcoxon helper requests SciPy's `method="exact"`, uses Pratt zeros by default, and catches exceptions by returning `p=1`. This can conceal errors and is not a sufficient guarantee of exact treatment of ties/zeros. See the [statistical protocol](pairwise_statistical_protocol.md).

Existing merge deduplication keys do not include complete configuration/sample identity. The new analysis needs its own auditable source-selection policy and must not blindly merge all available rows.

## Model names versus effective configurations

| Finding | Consequence |
| --- | --- |
| The repository baseline explicitly constructs UMAP without `n_components`, resulting in the installed UMAP default of **2**, rather than BERTopic's usual internally configured 5. | Call it the **repository BERTopic baseline**, unless a stock-BERTopic control is added. |
| HDBSCAN parameters vary by dataset: default `min_cluster_size=5` for ANES/FED/Gadarian, explicit 30 for Trump, and 15 for Yelp. | These are dataset-specific repository settings, not one untouched stock-BERTopic configuration. They match within the inspected baseline/AppendUMAP/AlignedUMAP branches. |
| PCA configurations explicitly use 5 dimensions; UMAP configurations generally leave dimensionality unspecified. | Current PCA → UMAP comparisons change both reducer and dimensionality. |
| Single-view K-means uses sklearn's `n_init="auto"`; MV K-means defaults to 5 initializations. | Current PCA and UMAP K-means pairs change optimization settings as well as architecture. |
| Single-view spectral uses RBF `gamma=1`; MV spectral defaults to a bandwidth estimated separately from each view. | The spectral pair changes affinity construction as well as adding multiple views. |
| `info_view=0` selects the text-side spectral embedding **after multi-view fitting**. | It is not a text-only or metadata-free control. |
| `normalize_text_view` is applied inside `MVCWrapper` after dimensionality reduction. | It normalizes reduced PCA/UMAP coordinates, not original sentence embeddings. |
| Spherical MV K-means normalizes both views. | It does not isolate text normalization. |
| AppendUMAP concatenates metadata before UMAP. | It changes the geometry of the reducer's input. |
| AlignedUMAP jointly fits text and metadata embeddings, returning the text-side embedding. | It is a different architecture for incorporating metadata into reduction. |

Relevant implementations: [models.py](../src/models.py), [mvc_wrapper.py](../src/mvc_wrapper.py), [append_umap.py](../src/append_umap.py), [decoupled_kmeans.py](../src/decoupled_kmeans.py), and [decoupled_spectral.py](../src/decoupled_spectral.py).

### Runtime confirmation and intended defaults

Actual model construction, without fitting, confirmed UMAP dimensions of **2** for the repository FED baseline versus **5** for plain `BERTopic()` in the current environment. HDBSCAN `min_cluster_size` was **5** versus **10**, respectively.

BERTopic technically accepts 2 dimensions; 5 is a default, not an API requirement. Nevertheless, the user explicitly requires BERTopic defaults, so the dimensionality mismatch is a confirmed issue for this project. The user owns the fix; no model code/configuration was changed during this review. Other effective baseline defaults also need checking. Preserve the distinction between historical 2-dimensional runs and corrected experiments.

### Requested versus realized topics

HDBSCAN branches vary BERTopic's post-clustering `nr_topics` reduction request. K-means/spectral branches vary `n_clusters`. These are different mechanisms even when both request 10–50 topics.

Retain the mechanism and realized `n_topics` separately. Match on the declared requested condition, not realized counts: realized topic count is an outcome of the intervention.

### Confirmed configuration-mutation issue

Model construction removes `normalize_text_view` from the supplied parameter dictionary using `pop`. The optimizer reuses each expanded configuration across seeds.

An isolated check of the existing constructor logic, without training models or editing files, produced:

```text
Repeated construction normalization flags: True False
Remaining params: {'n_clusters': 10}
```

This is a concrete correctness issue. Its historical impact depends on execution mode: separately dispatched individual runs may be unaffected, while sequential runs reusing the dictionary may be affected. Do not assume every normalized result is wrong, or that every result bearing that name actually used normalization.

The behavior was subsequently reproduced with the actual current constructors, again yielding `True`, then `False`. Affected historical files have not been identified. Ordinary configurations that never requested normalization are not implicated by this finding.

Proposed response: fix configuration mutation after approval, add a repeated-construction regression test, audit execution history, and **mark potentially affected normalized runs as unverified** until verified or rerun.

The earlier term “quarantine” meant temporarily excluding specific uncertain cells from an analysis claiming to measure normalization's effect. It did not mean moving/deleting files, rejecting every experiment, or interrupting MV-HDBSCAN integration. No such action was taken. Fresh-process individual runs may be unaffected; establish execution history before deciding which cells need reruns. Preserve original artifacts and document analytical exclusions.

### Geometry descriptions need precision

Normalizing input points does not make ordinary Euclidean K-means identical to cosine/spherical K-means because centroid updates also matter. Similarly, RBF affinity on normalized inputs is not the same as the decoupled spectral implementation's clipped cosine similarity.

The identity between squared Euclidean and cosine distance for unit vectors does not establish equivalence of the complete clustering algorithms. Descriptions in the comparison graph must name the actual intervention.

## Observed standard-result coverage

The inspection inventoried **top-level standard CSVs**, including merged files and individual runs. Counts below are observed seed × requested-count cells, **before full metric/provenance validation**. They are not certification of completed primary comparisons.

| Comparison | Observed coverage |
| --- | --- |
| PCA K-means ↔ PCA MV K-means | 15 cells on each side for all five datasets |
| UMAP K-means ↔ UMAP MV K-means | 15 cells on each side for all five datasets |
| Baseline ↔ AppendUMAP | 15 cells on each side for all five datasets |
| Baseline ↔ AlignedUMAP | Complete for four datasets; Trump has 6 of 15 AlignedUMAP cells |
| UMAP spectral ↔ MV spectral | Complete for ANES, FED, Gadarian; no ordinary MV spectral rows found for Trump or Yelp in the inspected top-level files |

Normalization and decoupled active configurations currently exist only for Yelp. Their results include **500-document runs mixed with full-size runs**. Full-size standard Yelp rows report **10,205 observations**.

The active [Yelp base configuration](../experiments/datasets/yelp.yaml) points to `data/processed/yelp_embeddings.parquet`. The [alignment script](../scripts/data_prep/align_yelp_sample.py) produces a separate `yelp_s10000_embeddings.parquet`. The canonical sample needs clarification; these identities must not be silently collapsed.

Archives may contain additional historical runs. They can be inspected read-only in a later discovery stage, but recovery must not silently choose incompatible or superseded experiments.

## Parity requirements for every proposed edge

Audit dataset and document sample/order; preprocessing regime; text and embeddings; metadata variables and transformations; dimensionality reduction and effective parameters; clustering settings other than the intended change; requested topics; training and sampling seeds; representation and evaluation procedures; and implementation/dependency versions.

Current metadata processing uses min-max scaling for numeric covariates, one-hot encoding for categoricals, and float conversion for binary fields. These transformations and their fitted input sample belong in the parity specification.

Python-only preprocessing, strict row alignment, representation-layer stopword filtering, and result-regime isolation remain repository requirements from [AGENTS.md](../AGENTS.md).
