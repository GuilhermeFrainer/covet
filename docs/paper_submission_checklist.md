---
title: Paper submission checklist (October 12)
created: 2026-10-08
status: in-progress
tags: [paper, covet, results, checklist]
---

# Paper submission checklist (October 12)

Working checklist for putting the results into `covet_2026-10-08.tex` (ACL long paper, 8 pages).
It follows on from [paper_results_plan.md](paper_results_plan.md), whose findings and sanity checks still apply.

> [!important] Goal for today (Oct 8)
> Preliminary tables and figures are in the paper, built from the results on disk.
> The FED/Yelp re-run (chunking fix `29a6e6f`) and STM land later and only need the generator to be re-run.

---

## Decisions

- [x] Trump: `trump_s25000` replaces full Trump as the fifth dataset.
- [x] Main-text metrics: $C_\text{NPMI}$ and $C_V$ (coherence), IRBO (exclusivity). Topic Diversity and $C_\text{UMass}$ go to the appendix.
- [x] Pairwise tests: exact Wilcoxon, Holm-adjusted within one family per metric across all presented edges.
- [x] Free-for-all models (may change, so the list lives in one config file):
  - BERTopic₁, TriTopic, STM
  - COVET$_\text{Ap}$, COVET$_\text{Ap}^{w=0.1}$, COVET$_\text{Al}$, COVET$_\text{MH}$, COVET$_\text{CR}$
- [x] COVET variant names: COVET with a subscript for the technique (see [Naming](#naming)); NaiveFusion becomes COVET$_\text{Ap}$.
- [ ] Update the `latex_label`s in [config/model_catalog.yaml](../config/model_catalog.yaml) to the new names.
- [x] Missing runs: listed in [Missing runs](#missing-runs); the outputs use whatever is available and flag incomplete rows.
- [ ] STM variant reported (stemmed or unstemmed), once the cluster results are in.

---

## Today (Oct 8): preliminary outputs in the paper

All outputs come from `uv run python scripts/analysis/make_paper_outputs.py` (Trump-25k by default), written to `~/Downloads/covet_paper_outputs/`, so the later re-runs only mean re-running it.

- [x] **Free-for-all model list in config**: [config/paper_benchmark.yaml](../config/paper_benchmark.yaml), read by the generator.
- [x] **T1 – Pairwise ablations** (main text, `table*`): `tables/pairwise_ablation.tex`; appendix metrics in `pairwise_ablation_appendix.tex`, r and Holm p in `pairwise_ablation_stats.tex`
  - [x] Rows: HDBSCAN family (BERTopic₁ → COVET$_\text{Ap}$, COVET$_\text{Ap}^{w}$, COVET$_\text{Al}$, COVET$_\text{MH}$, COVET$_\text{FS}$) and spectral family (UMAP + Spectral → COVET$_\text{CR}$, COVET$_\text{CT}$)
  - [x] Holm families: one per metric, 7 tests each
  - [x] Cells: mean Δ (W/T/L) and Holm p for NPMI, $C_V$ and IRBO
  - [x] Extra columns: Δ realized topics, Δ noise share, ΔAMI (marked † where AMI covers fewer datasets)
  - [x] Stats merged in: Holm p in the main table; rank-biserial r in the stats table
  - [x] Show each row's dataset count and mark incomplete rows (until the [missing runs](#missing-runs) are in: COVET$_\text{Ap}^{w}$ 3, COVET$_\text{MH}$ 4, COVET$_\text{FS}$ 4, COVET$_\text{CT}$ 4)
- [x] **T2 – Free-for-all** (main text, single column): `tables/benchmark.tex`; appendix metrics in `benchmark_appendix.tex`; per-dataset scores in `benchmark_datasets.csv`
  - [x] Mean score and average Friedman rank per metric (only models complete on all 5 datasets are ranked; the others are marked)
  - [x] Iman–Davenport p and Kendall's W per metric; no post-hoc pairwise tests
  - [x] Realized topic count column (TriTopic: counted from its exported `topics.json`, since its results record the requested count)
  - [ ] Includes STM automatically once its results are merged (check that its requested K is read from `n_topics`, `nr_topics` or `n_clusters`)
- [ ] **F1 – Metadata alignment** (main text): dose-response figure, Δ metrics and AMI against w
- [x] **T3 – Noise coverage** (appendix): `tables/noise_coverage.tex`; presented HDBSCAN models × datasets, mean ± SD noise share, plus the mean over datasets
- [ ] **T4 – Qualitative example**: keep `fed_qualitative_example_v3`
- [ ] *(optional)* **F2 – Trade-off scatter**: Δ NPMI against Δ IRBO
- [ ] **Appendix tables**: Diversity and UMass versions of T1/T2
- [ ] Fill the AMI gaps: backfill Fed Aligned UMAP and Fed Append UMAP (see [Missing runs](#missing-runs))
- [ ] Tests, ruff, and a pdflatex compile of every table
- [ ] `\input` the outputs into the tex; remove `combined_avg5`, `demsar_delta_yelp_no_stopword_removal` and `all_datasets_demsar_all_vs_all`

---

## Oct 9–10: methodology edits

- [ ] **Models**
  - [ ] Describe MV-HDBSCAN, Feature-Stacking and weighted Append (resolves `\todo{adicionar MV-HDBSCAN}`)
  - [ ] Apply the new variant names everywhere, including the extensions figure
  - [ ] Drop the variants that are not presented
  - [ ] Justify excluding K-Means with the topic collapse (about 7 of 30 requested topics), not "stop words"
- [ ] **Experimental Design**
  - [ ] HDBSCAN merges topics towards k only when it can, so the realized count can be lower (Gadarian for every model; Yelp and ANES at large k for the metadata-heavy variants)
  - [ ] The noise cluster counts as one of the k topics (realized = k − 1 when exact); verify in BERTopic first
  - [ ] Trump is the 25k sample ([trump_downsampling.md](trump_downsampling.md))
- [ ] **Evaluation Measures**
  - [ ] Unpadded protocol: topics with fewer than two scorable words are excluded from coherence; Diversity keeps the K × 10 denominator
  - [ ] AMI: what it measures, why AMI over NMI (chance-adjusted, comparable across cluster counts), and the circularity caveat (the models saw the covariates)
  - [ ] STM's AMI uses each document's most probable topic (argmax of θ)
  - [ ] Primary metrics are NPMI, $C_V$ and IRBO; Diversity and UMass in the appendix
- [ ] **Statistical Analysis**
  - [ ] Replace `[FAMILY DEFINITION]` with the Holm family (one per metric)
  - [ ] Add Iman–Davenport and Kendall's W for the omnibus test
  - [ ] State that pairwise claims come only from the planned comparisons
- [ ] **Preprocessing**: clean up the STM vocabulary-parity paragraph; resolve the "STM stemmed" `\todo`
- [ ] **Resource usage**: cluster hardware and the LLMs used
- [ ] **Limitations**: N = 5 significance floor, embedding truncation (if FED/Yelp are not re-run in time), single-membership assumption

---

## Oct 10–11: results text and framing

- [ ] Replace the May per-dataset subsections with one subsection per question:
  - [ ] Paired ablations (T1, F2 if there is room)
  - [ ] External comparison (T2)
  - [ ] Metadata alignment (F1 and the ΔAMI column)
  - [ ] Qualitative example (T4)
- [ ] Abstract, introduction and conclusion: trade-off framing; drop "consistently improves" and "strong performance of COVET₃"
- [ ] Delete the draft notes and resolved `\todo`s

---

## Oct 12: final pass

- [ ] Merge STM and the FED/Yelp re-runs (if they arrived), re-run the generator, recompile
- [ ] Check every number quoted in the text against the regenerated tables
- [ ] Page count (8 pages excluding Limitations, references and appendix)

---

## Naming

COVET plus a subscript for the technique. Rule: two capitals when the technique is two words, a capitalized abbreviation when it is one word. The weight of weighted Append is a parameter, so it goes in a superscript.

| Model | Catalog id | Name |
|---|---|---|
| Naive Append (formerly NaiveFusion) | `append_umap` | COVET$_\text{Ap}$ |
| Weighted Append | `append_umap_w*` | COVET$_\text{Ap}^{w}$ (e.g. COVET$_\text{Ap}^{w=0.1}$) |
| Aligned UMAP | `aligned_umap` | COVET$_\text{Al}$ |
| MV-HDBSCAN | `mv_hdbscan` | COVET$_\text{MH}$ |
| Feature-Stacking HDBSCAN | `feature_stacking_hdbscan` | COVET$_\text{FS}$ |
| Co-Reg MV Spectral | `mv_co_reg_spectral` | COVET$_\text{CR}$ |
| MV Spectral (co-training) | `mv_spectral` | COVET$_\text{CT}$ |

In tables, the variants can sit under a "COVET" group header so only the subscript appears.

---

## Missing runs

Coverage of the standard results on 2026-10-08 (15 runs = 3 seeds × 5 topic counts). Everything not listed is complete for the presented models.

- [ ] **STM**: all five datasets (results are on the cluster; merge them)
- [ ] **COVET$_\text{Ap}^{w}$ (weighted Append sweep)**
  - [ ] Fed: merge the 15 `append_umap_w000` and 7 `append_umap_w005` raw files; run the rest of w005 (8) and w010, w020, w030, w050 (15 each)
    - These raw files predate the evaluation fix: after merging, run `recompute_topic_metrics.py --datasets fed`. Until then they mark the dose-response figure preliminary.
  - [ ] Trump-25k: w000, w005, w010, w020, w030, w050 (15 each); w010 alone is enough for T1
- [ ] **COVET$_\text{MH}$**: Trump-25k (15)
- [ ] **COVET$_\text{FS}$**: Trump-25k (15)
- [ ] **COVET$_\text{CT}$ (MV Spectral)**: Yelp (15)
- [ ] **AMI backfill** (no training, a backfill run): Fed `aligned_umap` and `append_umap`
- [x] Trump-25k metrics recomputed under the current evaluation protocol (2026-10-08; its runs came from revision `17fde45`, before the fix). Original archived as `results/archive/trump_s25000_standard_merged_pre_recompute_20261008_151449.zip`.
- [ ] *(later)* FED and Yelp: rebuild and re-run every model after the chunking fix (`29a6e6f`)

---

## Risks

- [ ] FED/Yelp re-run lands on Oct 10–11: every FED/Yelp number in the text changes, and the qualitative example may need regenerating.
- [ ] STM does not arrive in time: T2 goes out without it, with a sentence in Limitations.
