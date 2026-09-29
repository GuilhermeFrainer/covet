# Experiment integrity repair plan

> Historical planning document; implementation details may be superseded.
> Start experiment batches with `scripts/pipelines/slurm/queue_exp.sh`;
> use `scripts/experiments/run_optimizer.py` for individual Python configurations.

**Status: implementation handoff, not implemented.** Prepared on 2026-09-16 after a read-only inspection. The user requested a plan before reaching usage limits. This document covers existing experiments and runners; it does not depend on implementing the proposed pairwise-comparison framework.

The baseline must use **BERTopic's defaults**, rather than the defaults of independently constructed UMAP/HDBSCAN objects. The user previously said they would fix baseline dimensionality. Coordinate that work with this plan and with the other agent integrating MV-HDBSCAN; inspect the current diff before editing shared files.

Tracking: [root issue checklist](../REPOSITORY_ISSUES.md). Background: [repository findings](../pairwise_repository_findings.md).

## Outcomes required

1. State what can actually be established about old runs, including explicit unknowns.
2. Preserve the present results and presentation materials for the advisor.
3. Correct active standard UMAP-family configurations to 5 dimensions; maintain comparable PCA dimensions.
4. Audit and align other UMAP/HDBSCAN settings with the intended BERTopic reference.
5. Record effective configuration and dimensionality in future results.
6. Save floating-point results without the current three-decimal storage restriction, while retaining existing table formatting.
7. Fix configuration mutation so normalization and other settings cannot change across constructions/seeds.
8. Keep corrected runs distinguishable from old runs in execution, resumption, merging, analysis, and dashboards.

## Evidence already obtained

These counts refer to model specifications in YAML files, not completed runs, and are a snapshot while another agent is working.

| Active reducer | Specifications | `n_components` in YAML |
| --- | ---: | --- |
| UMAP | 94 | Omitted in all 94 |
| AppendUMAP | 70 | Omitted in all 70 |
| AlignedUMAP | 30 | Omitted in all 30 |
| PCA | 50 | Explicitly 5 in all 50 |

Current UMAP, AppendUMAP, and AlignedUMAP constructors default to 2 dimensions. Thus all **194 active UMAP-family specifications** have the omission in the inspected code path. This includes variants using K-means, spectral clustering, spherical clustering, and HDBSCAN downstream, not only the baseline. Actual historical behavior still requires evidence tied to each run.

Archived configurations are different: many explicitly set 5 or search over 5/10/15 dimensions. **Do not label every historical UMAP experiment as 2-dimensional or rewrite archived YAML.**

None of the **37 top-level result CSVs** inspected has a dimension/component/config-hash column. The `models/` directory contains 101 `.rds` artifacts and one `.pth` file, with no saved BERTopic estimator identified there. Those R/autoencoder artifacts do not establish a BERTopic reducer's dimensions. Historical ZIP contents, logs, other artifact locations, and run-specific source history still need investigation.

Runtime comparison of bare constructors with plain `BERTopic()` found:

| Parameter | BERTopic-created object | Bare constructor |
| --- | --- | --- |
| UMAP `n_components` | 5 | 2 |
| UMAP `metric` | `cosine` | `euclidean` |
| UMAP `min_dist` | 0.0 | 0.1 |
| UMAP `low_memory` | `False` | `True` |
| HDBSCAN `min_cluster_size` | 10 | 5 |
| HDBSCAN `prediction_data` | `True` | `False` |

The active baseline YAML already explicitly sets cosine and `min_dist=0.0`; the table above is a **constructor-default diff**, not a claim that every listed mismatch remains in every repository model. FED runtime construction verified dimensions 2 and minimum cluster size 5. Existing Trump/Yelp HDBSCAN configurations explicitly use 30/15 respectively. Compare all effective parameters to find the complete per-model mismatch set.

Reference environment: BERTopic 0.17.3, umap-learn 0.5.9.post2, hdbscan 0.8.40, scikit-learn 1.7.2, Polars 1.35.2. Recheck versions after concurrent dependency changes. “BERTopic defaults” must refer to a recorded version/backend, not an unspecified future release.

Actual repeated construction also confirmed that normalization switches from `True` to `False` because `get_algorithm` mutates its input dictionary. The new MV-HDBSCAN/feature-stacking branches currently use the same `pop` pattern and belong in the regression coverage.

## Stage 1 — Audit historical runs without changing them

Create an audit-only command/report, reusable independently of future pairwise analysis. Resolve the results actually present, rather than inferring execution from the active config inventory.

For each run, check evidence in this order:

1. Saved fitted estimator or reduced-coordinate artifact explicitly linked to the run: inspect reducer settings and the actual reduction output shape. A plotting-only 2D projection is not evidence that clustering used 2 dimensions. Only deserialize trusted project artifacts if needed.
2. Run-specific configuration snapshots, parameter columns in archived CSVs, or logs that explicitly recorded dimensions/settings.
3. Run-specific code/config/dependency revisions and launch arguments sufficient to reconstruct the effective configuration. Record missing links; a timestamp alone is not a source-revision identifier.
4. Current configuration/code as contextual evidence only, not proof of historical execution.

Search the current CSVs, ZIP contents read-only, qualitative JSON metadata, logs/SLURM logs, configuration history, and known external artifact locations if available. Do not derive UMAP dimensions from original sentence embedding length or topic count. Do not rewrite old CSVs with guessed values.

The audit table should identify source file/run/seed, configured or observed dimensions, reducer/backend/version, relevant settings, evidence location, and one of:

- `verified_run_artifact`: direct run-specific evidence, distinguishing configured and observed output dimensions.
- `reconstructed`: effective settings reconstructed from sufficiently identified historical inputs; no claim of observed fitted shape.
- `unknown`: available evidence cannot establish the setting.
- `not_applicable`: model does not use this external reducer.

Include a summary of exactly how many runs fall into each category. If a run cannot be verified, explicitly tell the user: **“The stored artifacts do not establish this run's dimensionality.”** Do not substitute “probably 2” for a measured value.

**Acceptance:** every inspected run has evidence or a stated unknown; historical files remain byte-for-byte unchanged. Reconstruction is kept in a separate audit output.

## Stage 2 — Preserve and separate the advisor snapshot

Use a snapshot of the current campaign, not a selective deletion of suspected bad rows. The archive may include valid PCA or other models: label it **pre-correction campaign**, not “all invalid.”

1. Agree on a cutoff and coordinate with active experiment/result writers. Do not stop the other agent or running jobs automatically. Files that change during copying must be retried or excluded with an explicit report; do not claim a coherent snapshot of moving inputs.
2. Enumerate the precise files first: current raw/merged result CSVs, corresponding topic JSONs, current presentation tables/figures, relevant logs, experiment configurations, dependency locks, code revision and working-tree diff, and any historical-audit report. Preserve existing presentation exports verbatim.
3. Produce `results/archive/pre_correction_<timestamp>.zip`, preserving repository-relative member paths such as `results/...`, `output/...`, and `tables/...`. Exclude the destination archive itself, existing archive trees, unrelated large data, and model artifacts unless specifically needed. Record exclusions. Existing archives remain intact.
4. Include a manifest with paths, sizes, SHA-256 hashes, cutoff, environment/config identity, and known issues. State whether dimensions are verified, reconstructed, or unknown; never label the whole snapshot definitively 2D.
5. Verify ZIP integrity and round-trip member hashes against the frozen sources. Fail rather than silently accepting missing or changed files.
6. Keep an accessible advisor copy, for example under `results/legacy/<snapshot>/` with the original result/output/table tree. Verify that snapshot-derived copies reproduce the original tables. The current presentation does not require rerunning old experiments.
7. Start corrected outputs in a separate campaign namespace, e.g. `results/bertopic_defaults_v2/` and `output/bertopic_defaults_v2/`, with corresponding table destinations. Existing top-level files can remain temporarily as the legacy working view. A ZIP alone is only a backup; **loader/resume/merge isolation is what prevents mixing**.

Add an explicit campaign/root selector to producers and consumers that need access. Corrected analyses must require the selected campaign; legacy presentation remains available via its own path. Do not let recursive discovery accidentally ingest both or re-ingest archives. Once the snapshot and selectors work, moving the exact frozen top-level files into the legacy location is optional; never delete the only usable copy.

Before any Windows move, resolve and verify every source/destination stays within the intended workspace directories. Keep operations in native PowerShell or one implementation; no cross-shell destructive path composition. No data movement is authorized by this planning document itself.

**Acceptance:** the advisor snapshot is readable and hash-verified; corrected runs cannot resume from, deduplicate against, or average with legacy runs; rollback is documented.

## Stage 3 — Fix configuration mutation first

At the start of `get_algorithm`, deep-copy the component parameter dictionary before `pop`, insertion of `n_clusters`, or any other edits. Audit other factory boundaries for nested mutation and create immutable resolved-run snapshots before construction. The basic correction is conceptually:

```python
params = copy.deepcopy(config.get("params") or {})
```

Do not fix only one normalization branch: ordinary MV K-means, MV/co-regularized spectral, MV-HDBSCAN, and feature-stacking aliases all need coverage. Preserve explicit normalization values; no global enabling of normalization.

Test reuse of the same configuration over all three seeds, repeated model validation followed by training construction, and individually dispatched optimizer runs. Confirm the original nested dictionary is unchanged and the wrapper retains `True` on every requested normalized run (`False` when requested). Test `n_clusters` injection/removal without mutating caller data too.

Historical normalized results require a launch-history audit. Fresh-process individual jobs may be unaffected; sequential jobs may have mislabeled later seeds. Correcting the factory does not repair old results. Keep uncertain cells visibly unverified and rerun only when their scientific use requires it.

**Acceptance:** regression tests demonstrate stable parameters across constructors/seeds and preserve concurrent MV-HDBSCAN integration.

## Stage 4 — Establish a versioned BERTopic default profile

Yes, defaults can be checked programmatically without fitting or downloading an embedding model: construct `BERTopic()` in the pinned target environment, read its reducer and clusterer with `get_params(deep=False)`, and compare them with the repository-created estimators. Record estimator class/backend and versions. Exclude runtime identity noise from comparisons; do not ignore meaningful settings.

Define a small centralized, versioned profile derived from that reference. Keep explicit YAML overrides visible. Add a parity test against the pinned BERTopic reference so an upgrade cannot silently change the campaign's meaning. Do not instantiate a changing external default and silently adopt it on every future run.

For active standard configurations:

- Explicitly set UMAP-family `n_components: 5` and consistent shared UMAP settings. Apply to plain UMAP, AppendUMAP, and the underlying AlignedUMAP, including their clustering combinations and stemmed counterparts.
- Keep all 50 active PCA specifications at 5 and assert equality with the standard reduction-dimension setting. Audit PCA constructor fallbacks so a future omitted `n_components` cannot mean “retain all components.” A five-component PCA policy is the user's comparability requirement, not a claim that PCA is BERTopic's default reducer.
- Compare all UMAP parameters, including `n_neighbors`, `metric`, `min_dist`, `low_memory`, initialization, and seeds. Fixed seeds are a deliberate reproducibility setting and should remain explicit, even though stock defaults may be unset.
- Compare HDBSCAN `min_cluster_size`, `min_samples`, metric, selection method, prediction-data setting, and all remaining exposed parameters. Resolve BERTopic-level inputs such as `min_topic_size` and `low_memory` before constructing its submodels, so the factory respects top-level settings.
- For the true baseline, use the reference HDBSCAN defaults, including the reference minimum cluster size instead of historical 30/15 dataset overrides. Preserve any scientifically intended tuned variants as separately labeled configurations; do not silently describe overrides as default.
- Apply matched HDBSCAN settings to appropriate comparison controls. MV-HDBSCAN is a different estimator: map only genuinely equivalent supported settings, document non-equivalences, and coordinate with its implementing agent. Do not pass an arbitrary HDBSCAN parameter dictionary to it.
- AlignedUMAP has alignment-specific settings with no BERTopic counterpart. Match shared reduction settings and record the rest as architecture-specific; it cannot be made “stock UMAP” by setting 5 dimensions.
- Do not blanket-edit TriTopic/FastTriTopic internals, STM, or archived exploratory grids. Audit their own reducers separately and mark external BERTopic reduction fields inapplicable where appropriate.

Inspect configurations added by the other agent after the initial inventory. Update tests that currently assert bare HDBSCAN `min_cluster_size=5`; decide whether the generic factory remains generic or the BERTopic assembly supplies the profile, then test that boundary explicitly. Avoid a half-fix where YAML says 5 but runtime wrappers receive 2.

Keep topic-count grids, representation stopword policy, and evaluation definitions as explicitly documented campaign choices; do not inadvertently replace them while repairing submodel defaults. Record a complete baseline-versus-reference diff so any remaining intentional differences are visible.

**Acceptance:** all active standard UMAP-family/PCA outputs are 5-dimensional; actual baseline submodel parameters match the pinned reference except declared reproducibility settings. Historical configs remain preserved in the snapshot/archive.

## Stage 5 — Record effective settings in every new run

Add a shared lightweight run-metadata collector, for example `src/run_provenance.py`, used by both experiment runners and the optimizer. Capture configured/effective values **from the constructed objects**, not just YAML. Capture fitted output dimensions where available after training.

Suggested CSV columns:

| Column | Meaning |
| --- | --- |
| `result_schema_version` | Version of the stored row format |
| `campaign_id` | Separates pre-correction and corrected experiments |
| `dim_red_n_components` | Effective constructor setting, resolved from the reducer |
| `dim_red_output_dim` | Actual reduced text output width observed during fitting |
| `dim_red_n_neighbors`, `dim_red_metric`, `dim_red_min_dist`, `dim_red_low_memory` | Common effective reducer settings, null when inapplicable |
| `cluster_min_cluster_size`, `cluster_min_samples`, `cluster_metric`, `cluster_selection_method`, `cluster_prediction_data` | Relevant effective clustering settings |
| `normalize_text_view` | Actual wrapper setting, not inferred from model name |
| `resolved_config_hash`, `run_manifest_path` | Link to the complete run specification |
| `code_revision`, `code_dirty`, `dependency_lock_hash` | Implementation identity, supplemented by snapshot/manifest for dirty code |
| `run_status` | Explicit success/failure state; failures remain distinguishable from absent runs |

Retain existing requested topic count, realized topic count, dataset, seed, and algorithm fields. If `min_samples=None` means an algorithm-specific effective choice, record both the configured value and resolved meaning in the manifest; do not invent a numeric value.

For UMAP, fitted `embedding_` can supply reduced output width; AlignedUMAP's wrapper retains the returned text embedding; PCA exposes fitted dimensionality. Prefer capturing the actual reducer output shape through a tested adapter/collector without a second fit or transform. Do not confuse original sentence embedding width, a visualization projection, or the concatenated text-plus-metadata clustering input with reduced text dimensionality. If an output was not observed, store null with an evidence/status note.

Store complete resolved config and serializable effective estimator settings in a JSON run manifest, including versions, command overrides, sampling seed, dataset/document-order and embedding fingerprints, metadata transformation settings, representation/evaluation settings, and a dirty-source snapshot identity. Avoid dumping huge arrays or relying on unstable object `repr`. Do not serialize unsupported objects silently as misleading strings.

Capture failure metadata even when training fails before a result is produced; use a separate status artifact or explicit failed rows that downstream loaders cannot mistake for valid metrics. Future result rows and their qualitative JSONs should share the run identity.

Audit resume/schema merging: the optimizer currently attempts to cast new rows to an existing file's schema. New provenance columns must survive, and corrected runs must never append to a legacy file selected only by observation count. Reject incompatible campaigns/config hashes. Legacy missing fields remain unknown, never auto-filled with corrected defaults.

New numeric metadata must not become an evaluation metric in tables: `generate_gt_table` currently infers metrics from otherwise unclassified float columns. Explicitly select metric columns or exclude the new provenance fields.

**Acceptance:** a saved row and manifest establish actual settings and observed output dimensions, survive CSV/JSON merges, and remain excluded from metric-only displays.

## Stage 6 — Full-precision storage, unchanged display precision

The specific storage-loss path is:

```text
experiment.decimal_digits = 3
  → scripts/experiments/run_optimizer.py
  → Optimizer.save_results(..., decimal_digits=3)
  → write_csv(..., float_precision=3)
```

Remove presentation precision from CSV serialization. Use Polars' full floating-point serialization without a fixed three-digit cap, and verify round-trip behavior for the installed version. Preserve Float64 metric values and avoid rounding before saving. “Full precision” means preserving the computed floating-point values, not inventing additional scientific accuracy.

Keep `decimal_digits` available for display where appropriate; deprecate its storage meaning without silently breaking callers. The removed legacy runner, STM result writers, and result merging already used uncapped `write_csv` in the inspected paths; audit all writers and intermediate rounding rather than assuming every CSV currently has the same defect.

Leave existing LaTeX/Great Tables formatting in place. For example, the general LaTeX exporter uses `float_format="%.3f"` and the Great Tables exporter uses `decimals=3`. Other exporters may have their own precision; preserve each formatter's existing policy. Perform statistical calculations on unrounded values and format only at presentation boundaries.

Tests should write/read values such as `0.1234567890123456` and `-12.34567890123456`, verify numerical preservation, then verify that existing table cells still display their established precision. Test merge/resume round-tripping and that metadata is not rendered as a metric.

Old rounded CSVs cannot recover their original digits. Check existing qualitative artifacts only if they actually contain full-precision metrics; otherwise retain and label their original precision. Do not fabricate extra decimals.

**Acceptance:** new raw/merged CSVs retain computed precision; existing display conventions remain stable. New full-precision inputs may legitimately change rankings/statistics; unchanged formatting does not promise identical scientific conclusions.

## Stage 7 — Validation and rollout

Implement in small reviewable changes: snapshot/isolation; mutation regression; default-profile/config alignment; run provenance; CSV precision; integrated validation. Coordinate file ownership with the MV-HDBSCAN agent and the user's baseline fix. Do not overwrite unrelated working-tree changes.

Required checks:

- Active config inventory and no unintended edits to archived grids.
- Runtime parameter parity against the versioned BERTopic reference.
- Small synthetic fits that observe 5 output dimensions for UMAP, AppendUMAP, AlignedUMAP, and PCA; ensure fixtures have sufficient samples/features for their algorithms.
- Repeated construction over three seeds without mutation, including new HDBSCAN wrappers.
- CSV precision, metadata schema, backward-compatible legacy reads, campaign-isolated resume/merge, and unchanged display precision.
- Archive manifest/integrity checks and an advisor snapshot read test.
- Full `uv run pytest` and repository Ruff checks/formatting required by [GEMINI.md](../../GEMINI.md), preserving other agents' work.

Only after these pass, propose a concrete rerun matrix by dataset/model/seed/topic count. Standard UMAP-family runs need corrected versions for claims about the intended 5-dimensional architecture; HDBSCAN parameter changes may affect additional cells. PCA's 5-dimensional runs are not invalidated solely by this dimensionality finding. Reuse old runs only with adequate identity/normalization evidence; do not require blanket retraining of unaffected models.

The final implementation report must separate: verified historical behavior, unresolved historical unknowns, fixed code/configuration, preserved advisor artifacts, available corrected runs, and reruns still outstanding.

## Handoff checklist

- [ ] Recheck concurrent changes, versions, and config inventory.
- [ ] Audit old run evidence; report unknowns explicitly.
- [ ] Create and verify advisor snapshot and campaign separation.
- [ ] Fix normalization/config mutation across all affected branches.
- [ ] Coordinate the user's 5-dimensional fix; align remaining intended defaults and PCA policy.
- [ ] Add effective-settings/output-dimension CSV fields and run manifests.
- [ ] Preserve full precision in CSVs while retaining table formatting.
- [ ] Run targeted regressions and required full checks.
- [ ] Present the remaining rerun matrix; retain old presentation outputs.
