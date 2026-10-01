# Model organization and dashboard scope

[`config/model_catalog.yaml`](../config/model_catalog.yaml) is the source of truth
for model priority, family, presentation role, and baseline–ablation correspondence.
It is deliberately outside `experiments/`, whose YAML files are discovered as
runnable experiment configurations. No experiments or results need to be moved.

## Primary correspondence

| Reference baseline | Primary ablations |
| --- | --- |
| `baseline`: UMAP + HDBSCAN | UMAP + MV-HDBSCAN; UMAP + Feature-Stacking HDBSCAN; Append UMAP + HDBSCAN; Aligned UMAP + HDBSCAN |
| `umap_spectral`: UMAP + Spectral | UMAP + MV Spectral, including co-regularized, `info0`, normalization, and decoupled variants |
| `pca_k_means`: PCA + K-Means | PCA + MV K-means variants, including spherical K-means |

STM, TriTopic, and FastTriTopic are primary external baselines. They have no forced
reference-baseline relationship.

Append/Aligned UMAP combined with **any MVC algorithm remains secondary**.
PCA with clustering outside the K-means family and ordinary UMAP with K-means
variants are also secondary. Family membership never changes priority.

The correspondence is organizational. A baseline reference does not assert that
exactly one component changes. The catalog defines no statistical procedure.
The broader proposals under `docs/pairwise_*.md` remain proposals; any future
implementation can reference these model IDs instead of duplicating this policy.

## Editing the catalog

Each key under `models` is the canonical model ID, ordinarily the existing
experiment's `model.id`. Every entry explicitly declares:

- `label`: readable presentation name.
- `priority`: `primary` or `secondary`.
- `role`: `baseline`, `ablation`, or `external_baseline`.
- `family`: `hdbscan`, `spectral`, `k_means`, or `external`.
- `baseline_id`: the reference baseline for ablations; otherwise `null`.
- `reduction` and `clustering`: the architectural components.
- Optional `aliases`: historical model IDs with exactly the same identity.
- Optional `short_label`: plain-text paper name used by the model ablation table.
- Optional `latex_label`: the model's LaTeX symbol, written as raw LaTeX
  (e.g. `'$\text{\systemshort}_1$'`). Existing symbols must match the results
  tables in [`src/make_table.py`](../src/make_table.py); a test enforces this
  for `MODEL_LATEX_MAP`.
- Optional `change`: one plain-text sentence describing what differs from
  `baseline_id` (or describing the model, for baselines). Write plain text; the
  table generator escapes LaTeX characters.

All primary BERTopic variants and baselines carry `short_label`, `latex_label`,
and `change`.

Entries apply across datasets and preprocessing conditions. Seed and numeric run
suffixes and the `stemmed_` prefix are normalized when resolving results; `info0`,
normalization, and spherical variants retain separate identities. STM's topic-count
IDs resolve to `stm`. Unknown models remain visible under All or Unclassified;
they are never automatically promoted to primary.

The loader rejects duplicate YAML keys, conflicting aliases, invalid priorities,
unknown fields, and invalid baseline references. An ablation must reference a
baseline in the same family. Add new variants explicitly to the catalog, even if
their algorithm combination resembles an existing primary variant.

A catalog entry can exist before its runnable configuration. For example,
`mv_hdbscan` records the intended primary model even while its experiment files
are missing. This does not create or schedule any training jobs.

## Model ablation table

[`scripts/analysis/make_model_table.py`](../scripts/analysis/make_model_table.py)
exports a LaTeX table (Label | Model | Ablates on) from the catalog, sized for
one column of a two-column (ACL) layout. It contains no results and requires the
`booktabs`, `tabularx`, and `amsmath` packages and the `\systemshort` macro:

```bash
uv run python scripts/analysis/make_model_table.py --latex tables/model_ablations.tex
```

By default it lists primary BERTopic variants grouped under their reference
baseline, excluding the weighted Append UMAP variants. Optional flags:

- `--include-weighted-append` adds the `append_umap_w*` variants.
- `--include-changes` adds a What changes column from the `change` field and
  makes the table span both columns (`table*`).
- `--include-secondary` adds secondary models; entries without a `change` fall
  back to a generated reduction/clustering difference, which exposes pairings
  that change more than one component.
- `--include-external` adds STM, TriTopic, and FastTriTopic.

`get_results.ps1` writes the default table to `tables/model_ablations.tex` in its
output directory.

## Dashboard behavior

The Experiment Organization controls default to Primary and apply to all tabs:
quantitative results, qualitative topics, coverage, and paper tables. Family,
role, and reference-baseline selections further narrow the scope. Reference
selection includes both the baseline and its ablations, subject to priority and
role selections. “Include external baselines alongside families” adds external
baselines to family/reference selections, still respecting priority and role.

The catalog expander shows configuration and result availability across the
repository, independently of dataset filters. Coverage classifies configurations
even when they have no results, groups families, and places the reference baseline
first. Coverage-generated commands use the selected catalog scope. If the scope
has no results, the availability and coverage views remain accessible.

Existing result filters and paper-tab dataset/condition controls remain local to
their respective views; the new catalog scope is shared. Optional algorithm
exclusions default off so they do not silently remove primary PCA/K-means models.
Paper-table and aggregated-result export filenames include the catalog scope.
Analysis caches include the scoped dataframe so switching scopes refreshes results.

Training configuration discovery, CLI analysis defaults, result merging, and
statistical calculations are unchanged.
