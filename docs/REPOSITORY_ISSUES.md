# Repository findings tracker

Recorded on **2026-09-16**. Detailed evidence: [repository findings](pairwise_repository_findings.md). Future comparison design: [proposal index](pairwise_comparisons.md).

**History:** the [experiment integrity repair plan](archive/experiment_integrity_repair_plan.md) and its [execution guide](archive/EXPERIMENT_INTEGRITY_NEXT_STEPS.md) are archived. Their completed repairs are marked resolved below; unchecked entries remain open.

## Confirmed issues

- [x] **Baseline uses 2 UMAP dimensions instead of the intended BERTopic default of 5.** *(Resolved)* Updated all 194 active standard UMAP-family YAML configurations to explicitly configure `n_components: 5`, matching BERTopic defaults and PCA parity. Parity verified via tests in [tests/test_bertopic_defaults_parity.py](../tests/test_bertopic_defaults_parity.py).
- [x] **Other baseline defaults need checking against the intended BERTopic defaults.** *(Resolved)* Standardized `prediction_data: true`, `metric: "cosine"`, `min_dist: 0.0`, `low_memory: false`, and established an explicit dataset-sizing policy for `min_cluster_size` (10 for FED/Yelp, 30 for Trump, 5 for ANES/Gadarian). Documented in [docs/bertopic_default_parameters_and_clustering_decisions.md](bertopic_default_parameters_and_clustering_decisions.md).
- [x] **Normalization configuration is mutated during model construction.** *(Resolved)* [src/models.py](../src/models.py) previously removed `normalize_text_view` and other parameters from the caller's parameter dictionary with `pop`. This has been fixed by introducing `copy.deepcopy` at all factory and runner entry points. Regression test suite added in [tests/test_config_mutation_regression.py](../tests/test_config_mutation_regression.py).
- [x] **Stemmed STM configs trained on unstemmed BoW inputs.** *(Resolved 2026-09-30)* [scripts/experiments/run_stm.py](../scripts/experiments/run_stm.py) derived its RDS/BoW paths from the `dataset_path` stem only, which stemmed and unstemmed configs share. It now takes the `_stemmed` suffix from `text_col`, so STM matches the preprocessing level of the models it is compared against. No stemmed STM results had been merged. Tested in [tests/test_run_stm.py](../tests/test_run_stm.py).
- [x] **Stemmed MV-HDBSCAN and Feature-Stacking HDBSCAN configs were labeled as the baseline.** *(Resolved 2026-09-30)* In every `experiments/<dataset>_stemmed/`, `<dataset>_standard_mv_hdbscan.yaml` and `<dataset>_standard_feature_stacking_hdbscan.yaml` carried the correct clustering settings but `experiment.name: <dataset>_stemmed_standard_baseline` and `model.id: baseline`, so their results would have been filed as baseline runs. They were regenerated with the [generate_stemmed_configs.py](../scripts/data_prep/generate_stemmed_configs.py) transformation; only names, descriptions, and `model.id` changed. No stemmed results had been merged. The mislabeling also hid that `feature_stacking_hdbscan` had no [model catalog](../config/model_catalog.yaml) entry; it is now a primary ablation of `baseline`.

## Analysis concerns

- [ ] **Match seed/topic-count grids before averaging.** Existing delta analysis averages available rows before matching; incomplete coverage can therefore compare different conditions. Completed individual runs are not invalid merely because other cells are missing.
- [ ] **Keep sample identities separate.** Inspected Yelp results include 500-document and full-size runs (10,205 observations in inspected full-size rows). Their coexistence is fine; pooling them as the same sample is not. Resolve the intended canonical Yelp input and separate `yelp_s10000` artifacts.
- [ ] **Review exact Wilcoxon handling.** Requesting SciPy `method="exact"` alone does not guarantee the required exact treatment of ties/zeros. The helper's exception-to-`p=1` fallback can conceal errors. This concerns statistical conclusions, not trained-model validity; it does not establish that every existing p-value is wrong. See the [statistical protocol](pairwise_statistical_protocol.md).

## Comparison limitations and interpretation

- [ ] **K-means initialization differs:** single-view uses `n_init="auto"`; MV defaults to 5. Align settings for a controlled intervention or explicitly classify the comparison as architectural.
- [ ] **Spectral affinity bandwidth differs:** single-view uses RBF `gamma=1`; MV estimates bandwidth per view. Metadata is not the only changed component.
- [x] **PCA and UMAP dimensionality differ in the inspected configurations:** *(Resolved)* All 194 active standard UMAP-family YAMLs explicitly configured to `n_components: 5`, matching the 50 PCA configurations and tested in [tests/test_bertopic_defaults_parity.py](../tests/test_bertopic_defaults_parity.py).
- [ ] **`info_view=0` is not metadata-free:** it selects the text-side embedding after multi-view fitting.
- [ ] **Geometry descriptions need precision:** text normalization occurs after reduction; spherical MV K-means normalizes both views; normalized-input Euclidean/RBF methods are not generally identical to the corresponding spherical/cosine methods.
- [x] **Historical provenance is incomplete:** *(Resolved for future runs)* The run provenance system in [src/run_provenance.py](../src/run_provenance.py) records 20 effective estimator and execution fields in CSV results and writes lightweight JSON run manifests alongside qualitative topic outputs (`campaign_id: bertopic_defaults_v2`).
- [x] **Three-decimal metrics lose precision:** *(Resolved for future runs)* Removed `float_precision=decimal_digits` truncation from `Optimizer.save_results()`, preserving full `Float64` precision in CSV files while retaining 3-decimal presentation in LaTeX and Great Tables exports.

## Clarification of “quarantine”

The intended meaning was **mark potentially affected normalized runs as unverified**, and temporarily exclude specific uncertain cells from claims about normalization until their execution history is checked.

Determine whether each run received a fresh configuration. If normalization was lost, correct the bug and rerun affected cells. Preserve original files and document any analytical exclusions. Historical affected files have not yet been identified.

This did not mean deleting/moving results, rejecting every experiment, or stopping MV-HDBSCAN integration. No such action was taken. The 2-dimensional baseline issue is separately confirmed against the user's stated intent and will be fixed by the user.

Unchecked entries indicate pending work or verification; they do not all imply code bugs or mandatory reruns.
