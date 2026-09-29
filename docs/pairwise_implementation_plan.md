# Pairwise comparisons: implementation plan and open decisions

Part of the [pairwise comparison proposal](pairwise_comparisons.md). **No implementation is approved by these documents.** The current task only records the design for review.

## Recommended stages

| Stage | Work | Verifiable outcome |
| --- | --- | --- |
| 1. Freeze the scientific specification | Agree on nodes/final CAST architectures, claims, primary metrics, conditions, families, and provenance policy | Reviewed registry design and explicit decisions, without result-driven choices |
| 2. Registry and audit-only CLI | Schema validation, concrete configuration resolution, effective-parameter parity, result discovery, coverage reports | The CLI explains which comparisons are possible and why others are blocked; no inferential output |
| 3. Correctness/provenance prerequisites | Fix configuration mutation, add its regression test, classify historical normalized runs, add agreed future-run metadata | Repeated construction preserves settings; uncertain historical runs remain visibly uncertain |
| 4. Matching and aggregation | Deterministic source selection, expected grids, sample/condition checks, paired and dataset-level deltas | Synthetic fixtures yield known matched cells and equal-weight summaries |
| 5. Exact inference | Exhaustive signed ranks, zero/tie handling, Holm, effect sizes, small-sample notes | Known exact p-values and corrected families; no asymptotic or exception-to-success fallback |
| 6. Exports and graph | Machine-readable tables, Markdown/LaTeX summaries, minimal plots and DAG from the registry | Paper figures and tested contrasts use the same source definitions |
| 7. End-to-end verification | Synthetic fixtures, read-only current-result audit, documentation, Ruff, full pytest suite | Reproducible outputs and successful required checks |

The first recommended deliverable is stage 2 after the scientific choices needed for it are settled. Architectural comparisons can remain available while strict-ablation status is reserved for verified interventions.

New model controls and reruns are separate work items identified in the [comparison trees](pairwise_comparison_trees.md). Do not launch them merely because the analysis pipeline can describe them. MV-HDBSCAN remains the user's separate implementation work until integrated by agreement.

## Test coverage

At minimum, cover:

- Registry parsing, schema versions, unknown fields, and duplicate YAML keys.
- Duplicate model/comparison IDs and unknown node/model references.
- Cycles where the display is required to be acyclic.
- Invalid metric directions.
- Configuration resolution and undeclared effective-parameter differences.
- Repeated model construction without mutation, including normalization across seeds.
- Incomplete paired grids, failed/unknown cells, and missing/nonfinite metrics.
- Sample conflicts, preprocessing conflicts, ambiguous duplicates, and provenance policies.
- Correct matching by dataset, seed, requested topic count, and count mechanism.
- Correct improvement-direction transformation.
- Equal-weight dataset aggregation on a complete synthetic grid.
- Exact behavior on small synthetic examples, tied magnitudes, zeros, and all-zero differences.
- Holm adjustment and exact registered family membership, including multiple primary metrics.
- Deterministic output ordering and source selection.
- Consistency between graph references and tested comparisons.

Use focused tests for each stage and full integration fixtures for the final pipeline. Follow [AGENTS.md](../AGENTS.md): after implementation/configuration changes, run the full `uv run pytest` suite and apply the repository's Ruff conventions. Documentation-only registration does not constitute implementation or a completed test-suite validation.

## Decisions still needed from the user

1. **Meaning of Default BERTopic — resolved, 2026-09-16:** the user wants BERTopic defaults and will fix the 2-versus-5 dimensionality issue. Other effective default settings still need verification; the baseline definition is no longer an open naming choice.
2. **Primary metrics:** which coherence metric (`c_v` or `c_npmi`) and distinctiveness metric (IRBO or topic diversity) should be primary?
3. **Preprocessing scope:** should `standard` be primary, with stemmed experiments reported separately as robustness analyses?
4. **Canonical sample and pending results:** what is the intended Yelp sample, and are the missing Trump/Yelp spectral runs still running or stored elsewhere?
5. **Historical provenance:** should unverified results remain descriptive-only until verified, or may an explicitly qualified historical analysis use them?
6. **Final CAST architectures:** which variants are considered final, determining which metadata-off controls deserve priority?

The proposal also requires review of the recommended complete-grid equal-weight aggregation, two-sided exhaustive signed ranks with Pratt zeros, and hypothesis-family membership. These are proposed defaults, not silently finalized decisions.

## Scientific safeguards to retain

- Do not select metrics, aggregation, tests, graph edges, or multiplicity boundaries by which gives the strongest observed result.
- Keep dataset independence distinct from repeated seed/topic-count conditions.
- Keep standard, stemmed, and legacy no-stopword-removal regimes isolated unless a specifically registered question compares conditions.
- Preserve Python/R preprocessing parity and document alignment.
- Report coverage and historical uncertainty even when a comparison cannot proceed.
- Never call an edge a strict ablation based on its name alone.
- Preserve raw data, existing result artifacts, and unrelated user changes.

For the current evidence behind these requirements, see [repository findings](pairwise_repository_findings.md). For the proposed registry, modules, and statistical procedure, follow the [proposal index](pairwise_comparisons.md).
