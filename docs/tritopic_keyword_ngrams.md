# TriTopic coherence investigation and keyword configuration

## Investigation: September 2026 runs

The dashboard's error indicators initially appeared to suggest that Fed and
Trump could not run TriTopic or FastTriTopic. Inspection on September 23, 2026
showed a distinction between training completion and valid evaluation scores:

| Dataset / model | Evidence available during the investigation |
| --- | --- |
| Fed / TriTopic | 15 saved configurations marked successful; 12 had `NaN` in `u_mass` and `c_npmi`. |
| Fed / FastTriTopic | 15 saved configurations marked successful; the same 12 had those missing scores. |
| Trump / FastTriTopic | Two saved configurations (30 and 50 topics) completed, both with those missing scores. |
| Trump / TriTopic | No local logs or results found; the reason for missing runs was not established. |

Examples of completed training and saved outputs are recorded in the
[Fed TriTopic log](../logs/fed_standard_tritopic-20260918-130336.log) and
[Trump FastTriTopic log](../logs/trump_standard_fast_tritopic-20260917-234216.log).
These are historical observations, not an inventory of future reruns.

The [dashboard coverage logic](../src/experiment_tracker.py) labels runs with
missing metrics as errors or partial errors, even when training completed.
The evaluator can return a non-finite score without raising an exception, so
the saved run can have `run_status: success` alongside `NaN` metrics.
No SLURM explanation is needed for these completed runs' missing scores;
the separate [SLURM investigation](archive/slurm_fast_tritopic_concurrency_issue.md)
does not establish the cause of this issue.

## Cause: keyword and evaluation vocabulary mismatch

TriTopic's keyword extractor defaults to unigrams and bigrams. Our earlier
evaluation path used `text.lower().split()` for TriTopic, while BERTopic used
its own keyword vectorizer's analyzer. Whitespace splitting cannot produce a
single token such as `federal reserve`, and it retains punctuation that the
keyword vectorizer removes. Exact keyword matching can therefore discard terms.

In the saved Fed run `fast_tritopic_1_seed36201624`, topic 9's ten keywords
included `secretarymr`, `general counselmr`, and `assistant secretarymr`.
Only `assistant` matched the whitespace-tokenized evaluation vocabulary.
This leaves no pairs of distinct recognized words for pairwise coherence.
All 12 affected Fed configurations in each implementation had at least one
topic with fewer than two distinct in-vocabulary keywords.

A small Gensim reproduction with one recognized keyword and one missing phrase
returned `NaN` for `u_mass` and `c_npmi`, but a finite `c_v`. This explains how
the observed metric pattern can occur. The same per-topic diagnosis was not
established for the saved Trump runs.

## Implemented keyword configuration

Active TriTopic and FastTriTopic experiments explicitly use single-word keywords:

```yaml
model:
  params:
    keyword_ngram_range: [1, 1]
```

The bounds are inclusive: `[1, 2]` allows words and two-word phrases. The model
factories validate the pair and apply it to the keyword extractor before fitting.
This is a project-level option, removed before constructing the upstream
`TriTopicConfig`, which does not expose this setting. FastTriTopic inherits the
same extractor. Graph text features are unaffected.

The optimizer treats a flat pair as one setting; nested pairs such as
`[[1, 1], [1, 2]]` define a search grid. Configurations omitting the option retain
the upstream `(1, 2)` default. Archived experiments and existing results are not
rewritten. New results record the option in their resolved configuration.

Unigram extraction aligns keyword length with the BERTopic experiment defaults.
Coherence evaluation for both TriTopic implementations uses the fitted keyword
vectorizer's analyzer, matching its punctuation, stopword, and n-gram rules.
Methods without a vectorizer retain lowercase whitespace tokenization with a
warning. BERTopic variants continue to use their own `vectorizer_model` analyzer.
Matching tokenization does not guarantee finite scores for every topic.

## Diagnostics and validation

[Training and evaluation](../src/training.py) now warn when a configured coherence
or diversity metric is `NaN` or infinite, naming the model and metric while
preserving the score and continuing. Non-finite coherence also triggers a bounded
diagnostic summary of missing keywords and evaluation topic indices with fewer
than two recognized words. These indices are positions in the evaluated topic
list, not necessarily the model's topic IDs. Exceptions raised by evaluators
retain their existing behavior.

The implementation uses the existing upstream `TriTopicConfig`; there is no
project subclass. The factories remove `keyword_ngram_range` from its arguments
and apply it to the private keyword extractor. This dependency integration is
covered by tests and should be checked when upgrading TriTopic.

Tests cover [n-gram configuration and actual extraction](../tests/test_tritopic_keywords.py),
[coherence tokenization and BERTopic regression behavior](../tests/test_coherence_tokenization.py),
and [non-finite metric warnings](../tests/test_metric_warnings.py).
The full suite passed with 579 tests after the tokenization change; Ruff checks
and formatting also passed.

## Reruns, merging, and document assignments

Existing scores are not retroactively corrected. Rerun the affected experiments
to obtain results under the new extraction and evaluation settings. After a
successful default merge, newer rows replace older rows with the same dataset,
model identifier, and seed. Configurations not rerun retain their previous rows;
configuration hashes do not create separate deduplication identities.

Raw contributing and superseded CSV/JSON files are zipped into `results/archive/`
and `output/archive/`, respectively, then removed from the top-level directories.
The prior merged file is not itself backed up by that invocation. See the
[merge lifecycle](merge_results_lifecycle.md).

Assignment archival was discussed but **not implemented**. Bundles in
`output/document_assignments/<dataset>/<run_uid>/` remain on disk and are not
included in merge ZIPs. They are inspectable through their manifests, including
bundles for superseded runs. "Unarchived runs" can mean either:

- Runs with raw top-level result files still present: select those files and
  follow their `run_uid` / assignment links.
- Runs currently selected in merged results: select the merged rows and follow
  their assignment links, even if their raw result files were already archived.

The assignment directories alone do not distinguish these sets. Historical
runs without assignment exports cannot be reconstructed from representative
documents. See [document assignment exports](document_assignment_exports.md).
