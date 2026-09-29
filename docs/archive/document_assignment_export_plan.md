# Implementation handoff: document assignments for qualitative CAST analysis

> Historical planning document; implementation details may be superseded.
> Start experiment batches with `scripts/pipelines/slurm/queue_exp.sh`;
> use `scripts/experiments/run_optimizer.py` for individual Python configurations.

## Objective

Implement persistent document-level outputs so that we can find and verify examples where textually similar documents share a text-only baseline topic but CAST separates them into substantively meaningful topics associated with different metadata contexts.

The user will run the experiments after implementation and return with the artifacts. Implement and test the export and prepare narrow rerun configurations and commands. Do not automatically launch the full experimental campaign. Preserve existing results and saved qualitative examples.

The immediate priority is FED with CAST2 (`mv_spectral`), conventional BERTopic (`baseline`), and text-only spectral BERTopic (`umap_spectral`). This plan does not authorize changing their modeling behavior, preprocessing, defaults, or scientific comparison settings as part of export work.

## Existing evidence to preserve

All paths below are relative to the repository root.

- `scratch/poster_candidates/stronger_candidate.md`: recommended candidate, complete topic descriptions and six representative passages.
- `scratch/poster_candidates/fed_accommodation_vs_restraint.json`: structured evidence for that candidate.
- `scratch/poster_candidates/README.md`, `fed_selected_evidence.json`, `corresponding_topics.md`, and `fed_poster_example.svg`: earlier, weaker example that the user explicitly wants retained.
- `scratch/find_poster_example.py`: search using pairs exposed in both baseline and CAST representative lists.
- `scratch/search_contextual_topics.py`: broader search over CAST representatives, without requiring baseline co-membership.

The stronger candidate uses FED parquet source indices **3367** and **5129**:

| Property | A | B |
|---|---|---|
| Meeting date | 2016-07-27 | 2024-05-01 |
| Document type | Minute | Minute |
| Funds-rate metadata | 0.39% | 5.33% |
| CPI year-over-year metadata | 0.86836% | 3.24428% |
| Historical CAST topic | 12 | 38 |
| Historical topic size | 127 | 70 |

Historical model: `mv_spectral_5_seed36201624`, requested K=50, seed 36201624. Full-document embedding cosine similarity is approximately **0.864331**. The texts use similar language about maintaining interest rates, but their surrounding topics describe accommodation/gradual normalization versus restrictive inflation control. Neither baseline co-membership nor metadata causality has been established for this pair.

Treat historical topic IDs as references, not expected outputs of a rerun. Recover new assignments by source document key. Topic numbering and even membership may change with environment/code changes.

## Current implementation and missing information

Inspect applicable repository instructions before editing. Relevant code as reviewed for this handoff:

- `src/data.py::load_and_prep_data`: filters empty texts and optionally samples before selecting text, embeddings, and covariates. Returns three objects and currently discards source identifiers.
- `src/training.py::train_and_evaluate`: BERTopic calls `fit_transform`, then evaluates and returns metrics plus the fitted model. TriTopic uses a different interface (`labels_` versus topic objects in `topics_`).
- `src/utils.py::extract_qualitative_data`: saves topic-level counts, keywords, and representative texts, not full membership.
- `src/run_provenance.py`: already records estimator/configuration/code/dependency provenance. Extend this machinery where appropriate.
- `src/optimizer.py`: another caller of training and qualitative extraction; audit compatibility when changing shared interfaces.
- `scripts/experiments/run_stm.py`: obtains theta from `theta.parquet`, then extracts topic-level summaries. STM assignments require separate semantics.
- `scripts/analysis/merge_results.py`: merges and archives top-level output files. New artifacts must survive this lifecycle.

The minimal missing information is **a source-aligned final topic assignment for every training document in each run**, including noise. Existing source parquets already provide text, embeddings, and raw metadata; do not duplicate these large arrays per run.

## Phase 1: reliable source identity and input provenance

1. Preserve identity through the exact same filtering and sampling operations used to build training inputs. Do not reload/sample independently to reconstruct identity afterward.
2. Carry an input document table alongside text/embeddings/metadata. Prefer a named prepared-data object or a backward-compatible optional return/helper; audit all callers and mocks if changing the existing three-value interface.
3. Record both:
   - `input_position`: contiguous zero-based position actually passed to the model;
   - `source_document_key`: a stable, unique key within an identified immutable dataset snapshot.
4. Preserve existing source `index` and `id` columns when available. FED `id` alone is not unique across chunks. Verify whether `index` is unique; do not assume all datasets share that property.
5. If a source has no unique key, attach a physical source row ordinal **before filtering/sampling**, scoped by the source dataset checksum. Text hashes alone cannot distinguish duplicate documents.
6. Record the resolved dataset path (including the Yelp fallback), file checksum, selected text and embedding columns, full source row count, selected row count, sampling settings and effective sampling seed, and ordered selected-document key checksum. The current runner samples using `primary_random_state`, not necessarily each model seed.
7. Record raw covariate names/types and effective feature encoding/scaling order. Save an input-level preprocessing manifest (or reference a reproducible versioned procedure) including numeric scaling parameters and categorical feature names. Do not treat scaled values as raw metadata when producing qualitative reports.
8. Preserve access to the actual source snapshot: checksum plus path detects changes but cannot reconstruct an overwritten dataset. Use an existing immutable snapshot if available; otherwise document how the user should retain the parquet alongside the exports. Avoid duplicating it for each model run.

## Phase 2: per-run artifacts and schema

Use a dedicated subtree, for example:

```text
output/document_assignments/<dataset>/<run_uid>/assignments.parquet
output/document_assignments/<dataset>/<run_uid>/manifest.json
output/document_assignments/<dataset>/<run_uid>/representative_documents.parquet
```

Names are proposed, not an obligation to add three files if existing manifest/qualitative structures can safely hold the same information. Keep these files outside the existing top-level qualitative JSON discovery pattern.

Required assignment columns:

| Column | Meaning |
|---|---|
| `run_uid` | Unique execution identity, shared with the metrics, qualitative output, and manifest |
| `input_position` | Actual training row position |
| `source_document_key` | Unambiguous join to the identified source snapshot |
| `topic_id` | Final exported topic assignment; preserve `-1` noise |

Preserve source `index`/`id` as additional useful columns when present. Store a dataset/input snapshot identifier either in each row or unambiguously in the referenced manifest.

Manifest requirements:

- Schema version and unique `run_uid`; human-readable model ID, experiment ID, dataset, seed, timestamp, and preprocessing regime.
- Existing resolved model configuration, effective estimator settings, code revision/dirty state, dependency provenance, and links to the existing metrics/topic artifacts.
- Input provenance from Phase 1, including the selected ordered document set.
- Requested topic/cluster setting **and actual non-noise topic count**, document count, noise count.
- Assignment semantics (`hard_cluster`, etc.), artifact paths, row counts, checksums, and export status.
- Metadata condition (`observed` initially; distinguish any later ablations explicitly).

A model name such as `mv_spectral_5_seed36201624` is not globally unique across reruns. Use a collision-resistant execution identity; keep the existing human-readable identifier. Do not deduplicate new assignment artifacts solely by model name/seed. Prefer immutable per-execution files and no silent overwrite.

Optional assignment strength:

- Export only if it is already available and has well-defined semantics.
- Use nullable fields plus a strength-kind description. Do not fabricate confidence for K-means or spectral models.
- Do not blindly take the maximum of an arbitrary probability matrix: validate row/topic-column mapping after reduction and distinguish membership strength from topic probability.
- Do not enable expensive probability computation solely for the required export.

## Phase 3: extract final labels and integrate persistence

1. Implement a small shared extraction/export utility rather than duplicate logic across runner branches.
2. For BERTopic-backed models, extract final `topics_` after fitting and any topic reduction/remapping. Do not use the underlying clusterer's raw `labels_` as final BERTopic topic IDs. Verify agreement with the topic table being exported.
3. For TriTopic/FastTriTopic, inspect their fitted APIs and map document labels to exported topic IDs explicitly. `topics_` can contain topic objects rather than document labels. Do not apply the BERTopic extraction path blindly.
4. Validate before writing:
   - exactly one assignment per input document;
   - label count equals input count;
   - source keys are unique under the selected-input policy;
   - input positions preserve order;
   - all assigned topic IDs exist in the exported topic table, including noise where applicable;
   - hard-assignment topic counts agree with topic-table counts;
   - no accidental scalar, ragged, null, or misaligned label arrays.
5. Write artifacts atomically per successful fit, without waiting for the entire seed/configuration sweep to finish. Persist run status incrementally so a later failure does not orphan already exported assignments.
6. Prefer retaining valid fit assignments even if metric evaluation subsequently fails. If refactoring training/evaluation to permit this, track `fit_status`, `evaluation_status`, and `export_status` explicitly; never label a partially failed run a complete success.
7. Ensure optimizer executions retain assignment exports for baseline and other models. An export failure must be visible in run status/logging, not silently ignored.
8. For the new runs, enable assignment export by default; a documented opt-out is acceptable. Record when deliberately disabled. Existing historical outputs remain readable and are explicitly missing these artifacts; do not guess/backfill assignments from representatives.
9. Audit optimizer compatibility. Reuse the shared exporter for successful optimizer fits if feasible; otherwise explicitly document this as unsupported in the initial delivery rather than silently implying all entry points export assignments.

### Representative document identity

Save topic ID, representative rank, and source/input identity alongside representative text whenever the fitted model exposes exact selected indices. If only text is exposed, join against documents **within that assigned topic**. A unique match can be resolved; multiple matches must remain explicitly ambiguous (for example, candidate source keys). Do not select the first identical text arbitrarily. This enhancement must not block the essential complete-assignment export.

### STM scope

STM is not needed for the initial FED reruns. Avoid breaking its existing runner. If implementing the general exporter for STM, preserve theta and its explicit document and topic-column mapping, with `argmax(theta)` documented as a derived dominant topic rather than a hard clustering result. STM topic counts currently reflect probability mass, so hard-count equality checks must not be applied to them. If deferred, document the deferral clearly.

## Phase 4: merge/archive and retrieval compatibility

- Add links/run UID fields to metrics, qualitative rows, and manifests without breaking readers of legacy rows lacking those fields.
- Verify `merge_results.py` does not ingest assignment manifests as topic JSON, remove assignment files, or leave surviving result references dangling after archival.
- Preserve per-execution manifests/artifacts even if the metrics merge retains only the latest run for a configuration. Provide a direct index/discovery path to older execution artifacts.
- Record requested K explicitly in the new manifest; do not depend on parsing `_5` from a model ID.
- Do not write assignment Parquets into a location where result collectors assume every Parquet is a source dataset or every JSON is a topic table.
- Provide a small read-only inspection command/helper that loads one run, validates its source join, shows assignments for FED indices 3367 and 5129, and summarizes metadata for their complete topics. Do not infer old assignments or rename topics automatically.

## Phase 5: focused validation

Add tests for actual failure risks rather than tests that merely mirror field construction:

1. Filtering and sampling preserve exact alignment among document keys, text, embeddings, metadata, and final labels, including empty/null inputs under the loader's existing policy.
2. Duplicate texts and repeated parent IDs remain distinct documents; FED-like chunk identities join correctly.
3. BERTopic final remapping/reduction differs from raw cluster labels and the exporter chooses the final assignments.
4. Noise is retained; label length/topic-count mismatches fail explicitly.
5. Repeated same-model/same-seed executions receive distinct identities and cannot overwrite each other.
6. Nullable confidence and different label APIs behave correctly without adding probability computation.
7. Export failure, later evaluation failure, and interrupted writes cannot produce a falsely complete run or unreadable committed artifact.
8. Both runner branches export; legacy consumers and audited optimizer paths remain compatible.
9. Existing merge/archive behavior preserves new artifacts and references.
10. Round-trip a small artifact, join it to source rows, and recover correct full-topic metadata summaries.

Run targeted tests while developing, then the full repository suite as required by repository guidance. Run a small local smoke fit for the relevant model paths if dependencies permit, or clearly document environmental blockers and the exact smoke commands. A sampled smoke run is not the scientific rerun and must carry a distinct identity.

## Phase 6: prepare the smallest useful FED reruns

Create dedicated qualitative-rerun configs (with explicit unique experiment names) rather than editing the existing production grids. Preserve the model parameters and standard unstemmed input/representation settings; narrow only the requested K and seeds. Verify how config expansion generates model identifiers.

### Stage A: three runs

Dataset: full FED corpus, standard `clean_text` and `clean_text_embedding`, representation stopword removal enabled, **seed 36201624**, requested setting **50**.

| Existing config to derive from | Model | Requested setting |
|---|---|---|
| `experiments/fed/fed_standard_mv_spectral.yaml` | CAST2 multi-view spectral | `n_clusters: 50` |
| `experiments/fed/fed_standard_baseline.yaml` | BERTopic UMAP + HDBSCAN | `nr_topics: 50` |
| `experiments/fed/fed_standard_umap_spectral.yaml` | Text-only UMAP + spectral | `n_clusters: 50` |

Prefer separate model configs/executions so the baseline's actual topic count does not automatically override either spectral run's requested K. The runner has a `baseline_n_topics` path when models are combined; inspect this before composing any combined config.

The current runner supports `--seed` and `--single-seed`, but no general K override was observed. Narrow K in the dedicated configs. Do not run the existing full five-K grids unintentionally. Document exact executable commands for the new configs using the actual implemented filenames; include the user's normal remote execution workflow if relevant, without submitting jobs automatically.

Inspect the resulting assignments for source indices 3367 and 5129 and compare complete topic membership and metadata profiles. If the baseline already separates them, report that and search other pairs using the complete assignments; do not force the historical candidate into the poster narrative.

### Stage B: robustness and broader search

Use the same three models at requested K=50 for seeds **36201624, 62613654, 57116123**: **nine total runs**, including Stage A (six additional runs if Stage A is usable). Do not expand to all datasets or all K values until these results have been inspected.

Match source snapshot, preprocessing, and selected documents across runs. Compare co-membership/separation, not numeric topic IDs across runs. HDBSCAN may produce fewer than 50 non-noise topics and has noise; report actual counts/coverage. Do not count a split into noise as a meaningful two-topic separation.

## Optional later phase: isolate the metadata effect

This is separate scientific work, not a prerequisite for assignment export or the nine-run campaign.

- Text-only spectral is a closer comparison, but its graph construction/optimization can still differ from CAST. It is not automatically a fixed-architecture metadata ablation.
- `mv_spectral_info0` is **not** metadata-off: the wrapper still supplies text and metadata views. Do not relabel it as such.
- A useful negative control is the same CAST pipeline with metadata rows jointly permuted relative to documents. Keep each metadata row intact to retain relationships among covariates; retain the same text, hyperparameters, and model seed. Record permutation seed and mapping/hash, and use repeated permutations for a robustness claim.
- A true metadata-disabled variant needs an explicitly designed implementation that avoids degenerate constant-view graph behavior; do not assume zeroing a view creates a valid control.
- Separate observed topical coherence/contextual alignment from a causal claim that metadata improved the split. The selected example has semantic clues in its full text as well as different metadata.

## Acceptance criteria and handback

Implementation is ready when a future relevant run produces recoverable final assignments for every input document, source identity survives filtering/sampling, artifacts are linked to exact run provenance, and merge/archive operations preserve those links. Model behavior and existing results remain unchanged apart from added export/provenance.

Deliver to the user:

1. A concise description of implemented changes, any explicitly deferred model/runner support, and tests/smoke results.
2. Dedicated FED rerun configs and exact Stage A/Stage B commands, with clear run counts.
3. A short artifact validation/inspection command and expected output layout.
4. Instructions to return **assignment Parquets, their manifests/input mapping, topic JSONs, metrics, and the matching source parquet snapshot (or an accessible unchanged copy)** to this workspace. Keep all three seeds when Stage B is run.
5. A statement that the saved candidates remain intact and that new topic IDs must be read from new assignments.

Whole fitted model serialization, fresh embedding generation, full-corpus probability matrices, new visualization infrastructure, and a complete rerun of every dataset/model are not required for this task.
