---
title: Paper results plan (October submission)
created: 2026-10-01
status: in-progress
tags: [paper, covet, results, plan]
---

# Paper results plan (October submission)

Working plan for deciding which results go into `covet_2026-10-01.tex`.
It is based on a snapshot of `results/` on 2026-10-01: the standard regime only, 3 seeds × k ∈ {10, 20, 30, 40, 50} = 15 runs per cell.
Tick items off as they are resolved and add notes inline.

> [!summary] Proposed framing
> Present the work as a trade-off, not an improvement. *Where and how strongly metadata enters BERTopic determines a trade-off between metadata alignment and lexical topic quality.*
> The current data do not support "consistently improves coherence and exclusivity".

---

## 1. Coverage snapshot

| Model | ANES | Fed | Gadarian | Trump | Yelp |
|---|---|---|---|---|---|
| BERTopic₁ (UMAP + HDBSCAN) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Naive Append UMAP | ✓ | ✓ | ✓ | ✓ | ✓ |
| COVET₃ (Aligned UMAP) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Feature-Stacking HDBSCAN | ✓ | ✓ | ✓ | ✓ | ✓ |
| PCA + K-Means | ✓ | ✓ | ✓ | ✓ | ✓ |
| TriTopic (FastTriTopic on Trump) | ✓ | ✓ | ✓ | 10/15 | ✓ |
| MV-HDBSCAN | ✓ | ✓ | ✓ | ✗ | ✓ |
| UMAP + Spectral | ✓ | ✓ | ✓ | 4/15 | ✓ |
| Co-Reg MV Spectral | ✓ | ✓ | ✓ | ✗ | ✓ |
| PCA + MV K-Means | ✓ | ✓ | ✓ | ✗ | ✓ |
| MV Spectral / MV Spectral info0 | ✓ | ✓ | ✓ | ✗ | ✗ |
| Weighted Append sweep (w = 0 … 0.5) | ✓ | ✗ | ✓ | ✗ | ✓ |
| **STM** | ✗ | ✗ | ✗ | ✗ | ✗ |
| `trump_s25000` | — | — | — | — | — |

- [ ] Re-run the coverage check after the Trump merge (see [[#4. Pending runs and merges]])

---

## 2. Key findings so far

These are dataset-level paired deltas: each variant minus its reference, matched by seed and k, then averaged per dataset.

- **No variant improves on its reference on every dataset for any metric.** The best record is 4 of 5.
- **The large coherence gains come from metadata dominating the clustering.**
  - ANES, MV-HDBSCAN: NPMI goes from −0.21 to +0.03, but diversity falls from 0.92 to 0.54.
  - Yelp, Naive Append / Feature-Stacking / MV-HDBSCAN: the topic count drops from 29 to 13–16, IRBO from 0.99 to 0.42–0.67, and noise to 0%.
  - Gadarian, Naive Append: AMI goes from 0.03 to 0.59, so the topics approximate a metadata partition.
- **COVET₃ (Aligned UMAP) is close to the baseline.** All deltas are about ±0.01, AMI is unchanged, and noise is 2–9 points higher. It barely uses the metadata, which contradicts the current text saying it "dominates exclusivity".
- **Spectral family:** Co-Reg is broadly neutral against UMAP + Spectral. MV Spectral mostly loses (Fed C_V −0.10).
- **K-Means family:** MV K-Means returns about 7 topics when 30 are requested. This degenerate result is a stronger reason for exclusion than "stop words".
- **The weighted Append sweep is the clearest mechanism result.**
  - w = 0 reproduces the baseline (parity check).
  - As w increases, AMI rises and noise falls.
  - At w = 0.1–0.2, coherence rises slightly on Yelp (C_V 0.535 → 0.563) without any diversity loss.
  - At w ≥ 0.3, diversity and topic count collapse.
- **FastTriTopic matches TriTopic exactly** on every seed and k, and runs 2–16× faster. Report a single TriTopic row.

---

## 3. Data sanity checks (do before generating tables)

- [x] **Co-Reg vs Co-Reg info0 give identical partitions.** *(Resolved 2026-10-01: they are the same model.)*
  - mvlearn's `MultiviewCoRegSpectralClustering` accepts `info_view` (inherited from `MultiviewSpectralClustering.__init__`), but its `fit` never reads it: it always clusters the stacked eigenvectors of all views (`np.hstack(U_mats)`). Plain MV Spectral does use it (`embedding_ = U_mats[info_view]`), so `mv_spectral_info0` is a genuine variant.
  - Evidence: on Fed, all 15 paired runs have identical document assignments (ARI = 1.0). Across ANES, Fed and Gadarian, all 45 pairs have identical topic words.
  - The stored scores differ only where topics were padded (ANES 15/15 pairs, Gadarian 8/15, Fed 0/15), because `random.choices` was unseeded. Padding noise alone moved C_V by up to 0.029 for identical topics, which is evidence for the evaluation-change disclosure.
  - → Present only Co-Reg. The `*_co_reg_spectral_info0` configs (plain, Append UMAP and PCA variants) waste compute.
  - [x] Archived the 24 active Co-Reg info0 configs into `experiments/archive/<dir>/`, removed them from the queue lists, and demoted `mv_co_reg_spectral_info0` to secondary in the catalog (with a corrected `change` text). The single Trump info0 run is ignored, because Trump is being re-run on `trump_s25000`. (2026-10-01)
- [x] **Naive Append is identical to w = 0.5 on Gadarian** (15 of 15 runs). *(Resolved 2026-10-01: a real effect, not a bug.)*
  - Not a config mix-up: the w = 0.5 runs recorded `euclidean` and their own config hash, and on ANES and Yelp the two models don't match a single metric.
  - Both models produce the **same partition** in all 15 pairs (ARI = 1.0; only the topic IDs differ). That partition *is* the metadata: for k ≥ 20 it matches Gadarian's 14 metadata profiles (7 party levels × 2 treatment arms) exactly (AMI = 1.0). At k = 10, both merge the profiles identically. At w = 0.3, AMI with the profiles is already about 0.95.
  - Same partition → same topic words → same metrics. The NPMI differences in 4 pairs are 2.8e-17 (summation order).
  - → For the paper: this is the far end of the dose-response curve. On Gadarian, from w ≥ 0.3 the topics essentially reproduce the metadata groups, which supports the trade-off framing.
- [ ] **NaN NPMI/UMass** (investigated 2026-10-01; three separate causes)
  - [ ] **Trump FastTriTopic is stale.** The cluster ran revision `cd85c54` (Sept 21), which predates the unigram fix `2d872c3` (Sept 23). Its keywords include bigrams, which reproduces the mechanism in [tritopic_keyword_ngrams.md](tritopic_keyword_ngrams.md). *All* of its metrics are incomparable, not only the NaN ones. Every other dataset's TriTopic results include the fix. → Re-run (no code change needed).
  - [x] **ANES weighted Append (3 runs): tokenization mismatch.** *(Fixed in code 2026-10-01; existing runs still need the recompute below.)*
    - BERTopic builds topic words after *deleting* punctuation (`_preprocess_text`: "economy.no" → "economyno"), but evaluation tokenized the raw text, which *splits* on punctuation.
    - Gensim silently drops the words it doesn't know. A topic left with fewer than 2 tokens produces NaN (reproduced on ANES topic 9: NPMI/UMass NaN, C_V 1.0).
    - Silent effect on every run: about 11–15% of BERTopic topic words on ANES were dropped from scoring, and about 0–3% elsewhere. TriTopic is at 0%.
  - [x] **Padding inflates coherence.** *(Fixed in code 2026-10-01; existing runs still need the recompute below.)* Topics with fewer than 10 words were padded by repeating their own words (with an unseeded `random.choices`), and self-pairs score the maximum (`['war']*10` → NPMI 1.0, UMass 0.0).
    - Share of padded topics on ANES: baseline 8%, Aligned 12%, Append 16%, Spectral 26–36%, MV-HDBSCAN 54%, Feature-Stacking 51%. Gadarian Spectral: 3–8%. Close to 0% elsewhere.
    - The large ANES coherence "gains" of MV-HDBSCAN and Feature-Stacking are likely an artifact of this padding.
    - The padding was added in April (`36d7bbf`) because OCTIS refuses to score when the *first* topic has fewer than `topk` words. The new code calls gensim directly, so it no longer needs it.
    - Reviewers: xxbv listed the padding as a weakness ("artificially modifies topic representations"); Yiq2 praised the paper for disclosing it. → Disclose the change and its effect in the paper.
- [ ] **TODO: recompute topic metrics for existing runs** (no retraining). The scores are updated **in place in the merged result CSVs**, so the dashboard, `get_results.ps1` and `make_paper_outputs.py` pick them up with no further wiring (decided 2026-10-01).
  ```bash
  # 1. Merge first: the script refuses to run while unmerged raw CSVs exist (Trump has ~100).
  uv run python scripts/analysis/merge_results.py
  # 2. Optional rehearsal: every computation and safeguard, nothing written.
  uv run python scripts/analysis/recompute_topic_metrics.py --dry-run
  # 3. The real run (or --datasets anes fed gadarian yelp first; Trump is the slowest).
  uv run python scripts/analysis/recompute_topic_metrics.py
  ```
  - Updates `c_npmi`, `c_v`, `u_mass`, `irbo` and `topic_diversity`. It keeps the previous values as `<metric>_padded`, and adds `n_topics_short`, `n_topics_unscored`, `n_keywords_oov` and `evaluation_protocol` (`unpadded_2026_10`). Rows already at that protocol are skipped, so re-running is safe.
  - Safeguards. A file is left untouched unless all of these pass:
    - unique run keys;
    - BERTopic topic counts match the stored `n_topics`;
    - wherever no topic needed padding, the previous protocol reproduces the stored NPMI exactly, which proves each row is matched to its own topic words.
    The original file is zipped into `results/archive/*_pre_recompute_*.zip` (verified byte for byte), and the new file is written, re-read and validated before it replaces the original.
  - Checked on 2026-10-01 on real data, with nothing written:
    - Gadarian dry run: 360 rows; 275 verified exactly, 85 unverifiable (they had padded topics), 0 failures.
    - Full write test on a copy of Gadarian: only the metric columns changed. The archive equals the original. TriTopic rows without short topics are unchanged (within 1e-16), and IRBO is unchanged for unpadded rows. No topic words are missing from the evaluation vocabulary any more.
  - Runs whose topic words weren't saved stay at the old protocol (`evaluation_protocol` empty), and the paper outputs stay "Preliminary" while any exist.
  - [ ] After running it: re-check the ANES conclusions (MV-HDBSCAN / Feature-Stacking gains, NaiveFusion's coherence wins).
  - [ ] Paper: describe the evaluation (no padding; topics with fewer than two distinct words excluded from coherence; Topic Diversity keeps the K × 10 denominator), report `n_topics_short` / `n_topics_unscored` per model and dataset, and make NPMI and C_V the primary coherence metrics, with UMass secondary (Yiq2: UMass correlates worst with human judgment).
- [ ] **TriTopic realized topic count is not recorded.**
  - Cause: the grid parameter `n_topics` has the same name as the realized-count metric, and `src/optimizer.py` overwrites the metric with the requested value. The `n_topics` column therefore holds the *requested* k.
  - No runs are lost, and the other metrics are correct.
  - 31 of 130 runs differ from the requested k; 24 of them are on Gadarian (k=50 → 25–26 topics).
  - The realized counts can be recovered from the assignment manifests via `run_uid` (2 Trump rows have no `run_uid`).
  - [x] Code fix: grid parameters no longer overwrite measured values, and every run records an explicit `requested_topics` column (2026-10-01)
  - [ ] Existing rows are left as they are (no footnote). Gadarian and ANES may be re-run later with smaller topic counts.
- [ ] **Trump FastTriTopic is incomplete:** 10 unique runs; k=20 (all seeds) and k=30 (2 seeds) are missing.
- [ ] **The dashboard coverage tab is misleading.**
  - It shows "✅ Done" whenever any valid run exists (e.g. "Done (1 runs)").
  - It counts duplicate rows from unmerged reruns (e.g. "Done (45 runs)").
  - It should count unique seed × k runs against the expected 15.
- [ ] **AMI coverage is partial.** The `meta_ami_mean` column plus `results/derived/metadata_alignment_backfill.csv` cover only some model × dataset cells. Decide whether to complete the backfill or limit AMI to the weighted-append and HDBSCAN-family figures.
- [ ] Decide how to report the confound from the realized topic count. HDBSCAN variants and MV K-Means return fewer topics than requested, which shifts UMass and diversity. Plan: always show Δ#topics next to the metric deltas.

---

## 4. Pending runs and merges

- [ ] Merge the ~100 unmerged Trump raw CSVs (`uv run python scripts/analysis/merge_results.py`)
- [ ] Weighted Append sweep on **Fed** (~40 s per run, cheap)
- [ ] Weighted Append sweep on **Trump** (~420 s per run)
- [ ] *(optional)* Finish FastTriTopic on Trump (5 runs missing)
- [ ] *(optional)* MV-HDBSCAN / Co-Reg on `trump_s25000`, to get the 5th dataset for those families
- [ ] *(optional, decide)* STM. Run the stemmed/unstemmed STM, or drop it from the quantitative results (see [[#6. Drop or defer]])

---

## 5. What goes into the paper

### Main text

- [ ] **RQ1, HDBSCAN-family paired analysis** (the only family complete on 5 datasets)
  - Variants: Naive Append, COVET₃, Feature-Stacking, and MV-HDBSCAN (marked as 4 datasets)
  - Delta table: mean Δ, W/T/L, rank-biserial r, exact Wilcoxon p (Holm), plus Δ#topics and Δnoise columns
  - Frame the p-values as consistency summaries (minimum p = 0.0625 with n = 5, and 0.125 with n = 4)
- [ ] **Trade-off figure:** Δcoherence (NPMI) vs Δdiversity, one point per variant × dataset
- [ ] **Weighted Append dose-response figure:** metrics and AMI vs w (ANES, Gadarian, Yelp, plus Fed and Trump if they are run)
- [ ] **Spectral family, compact:** Co-Reg vs UMAP + Spectral on 4 datasets, stated as broadly neutral
- [ ] **External comparison, descriptive only:** absolute means and Friedman average ranks for BERTopic₁, the selected COVET variants and TriTopic. No Nemenyi or all-vs-all pairwise tests.
- [ ] **Qualitative Fed example** (keep the existing `fed_qualitative_example_v3`)

### Appendix

- [ ] HDBSCAN noise-coverage table (requested by a reviewer), restricted to the presented models
- [ ] K-Means family collapse (~7 of 30 requested topics) as the justification for excluding it
- [ ] MV Spectral and MV Spectral info0 results
- [ ] Per-dataset absolute metric tables
- [ ] Note that FastTriTopic reproduces TriTopic exactly, with timings

---

## 6. Drop or defer

- [ ] Remove STM from the results section. Keep it in related work and mention it in Limitations (unless it is run in time).
- [ ] Remove `demsar_delta_yelp_no_stopword_removal` (stale)
- [ ] Remove the `all_datasets_demsar_all_vs_all` / Nemenyi tables
- [ ] Remove the May-submission tables (`combined_avg5`, `combined_best4`)
- [ ] Move RQ2 (modality-specific geometry, Decoupled MV Spectral) to future work, since there are no current results

---

## 7. Manuscript edits

- [ ] Abstract: replace "improves topic quality" with the trade-off framing
- [ ] Introduction: rewrite the contribution bullet on "consistently improves coherence and exclusivity"
- [ ] Introduction: state the RQs explicitly (RQ1 paired ablations, RQ3 descriptive benchmark, RQ4 via AMI/dose-response), and drop RQ2
- [ ] Section 3: describe only the presented variants (add MV-HDBSCAN, Feature-Stacking and weighted Append; resolve the `\todo{adicionar MV-HDBSCAN}`)
- [ ] Renumber the COVET variants consecutively, or switch to descriptive names (the catalog currently uses ₁,₂,₃,₇,₈,₉), and update `config/model_catalog.yaml` `latex_label`s to match
- [ ] Section 3.4: replace the "stop words" justification for excluding K-Means with the topic-collapse evidence
- [ ] Statistical analysis: define the Holm family (proposal: one family per metric across the presented edges)
- [ ] Evaluation measures: add the AMI description (why AMI over NMI: chance-adjusted, comparable across different cluster counts)
- [ ] Results: rewrite the per-dataset subsections from the old May numbers, or replace them with family-based subsections
- [ ] Limitations: n = 5 significance floor, missing STM, Trump coverage gaps, single-membership assumption
- [ ] Resource usage: update to cluster hardware and the list of LLMs used
- [ ] Conclusion: drop "given the strong performance of COVET₃"

---

## Next session (2026-10-02, 8:00–10:00, before the 10:00 meeting)

Both sanity checks were finished on 2026-10-01. **The RQ1 dashboard work below was also done on the evening of 2026-10-01** (logic in `src/comparisons/views.py`, tested; the dashboard was driven headlessly through every scope). The 8:00 slot is free to review it before the meeting.

> [!warning] Trump is excluded from the cross-dataset tests (coverage 4/5) until the Trump raw files are merged. The Sept 17 merged rows and the Sept 29 raw reruns are two different runs for the same cells, so the comparison code rejects them as ambiguous.

- [x] **8:00–8:30: compact comparison table**
  - Drop the "Intended change" and "Type" columns.
  - Split "Comparison" into **Reference** and **Proposed**, using the catalog's `short_label`. Keep "Chain" as the grouping column.
  - Add a **metric selector** (default NPMI). The table shows only that metric: reference score, proposed score, mean Δ, median Δ, W/T/L, rank-biserial r, exact p, Holm p and coverage.
  - The dot plot **follows the selector**: a single panel replaces the fixed NPMI / IRBO / AMI panels.
- [x] **8:30–10:00: dataset / topic-count scope**
  - `src/comparisons/analysis.py`: add an optional `requested_topics` filter to `compute_ablation_comparisons` and `compute_registered_edge_comparisons`. "Complete" then means all 3 seeds at the chosen topic counts. Add tests.
  - **Scope controls** at the top of the tab: Dataset (All / one) and Requested topics (All / one).
    - All datasets + fixed k: dataset deltas average 3 seeds; W/T/L and tests still apply.
    - One dataset: descriptive table (scores mean ± SD over seeds, Δ, seed-level wins) and a plot of Δ against topic count (one line per comparison, error bars over seeds).
    - One dataset + one k: per-seed values, realized topic count and noise share.
  - **Heatmap**: comparisons × the 25 dataset × topic-count cells, coloured by the seed-averaged Δ of the selected metric, with realized topic counts in the tooltip.
  - Label all single-slice p-values "exploratory". Show realized topic counts (HDBSCAN requested ≠ realized, e.g. Gadarian ≈ 13).
- **Remaining for the 2026-10-02 8:00–10:00 slot** (from the original pre-meeting list; the dashboard work was done the evening before):
  - [x] Paper output generators (done the evening of 2026-10-01): `uv run python scripts/analysis/make_paper_outputs.py` writes to `tables/paper/` (git-ignored):
    - `hdbscan_ablation.tex`: mean Δ with W/T/L for $C_\text{NPMI}$, $C_V$, IRBO and Diversity, plus Δ realized topics and Δ noise share, against UMAP + HDBSCAN
    - `hdbscan_ablation_stats.tex`: rank-biserial r and Holm p (Holm across the 4 HDBSCAN variants)
    - `tradeoff.{pdf,png}`: dataset-level Δ NPMI vs Δ Topic Diversity
    - `dose_response.{pdf,png}`: weighted Append, Δ NPMI / Diversity / noise share / AMI against w
    - Marked "Preliminary" automatically until every standard row has `evaluation_protocol = unpadded_2026_10`. Tested; the tables compile with pdflatex.
  - [ ] After the meeting: the descriptive benchmark table with TriTopic (average ranks).
  - [x] ~~Wire a recomputed-metrics sidecar into the dashboard and comparisons~~: no longer needed. The recompute now updates the merged CSVs in place, so every consumer reads the corrected values (2026-10-01).
  - [ ] 9:15–10:00: review the dashboard and the outputs together.
- Deferred to the weekend (user): run the recompute, merge the Trump raw files, optional runs (weighted Append on Fed/Trump, Trump TriTopic), paper writing.

## Log

- 2026-10-01: Initial plan from a results snapshot (standard regime only; no stemmed or STM results merged).
- 2026-10-01: Replaced the coherence evaluation: no padding, BERTopic-aligned tokenization, unscorable topics excluded and counted. Added the offline recompute script, which has not been run yet (see the TODO in section 3).
- 2026-10-01: Investigated TriTopic. The topic-count overwrite is real but minor, and no runs are lost in merging. The coverage-tab gaps come from NaN metrics, runs that were never produced, and the tab's status/count logic.
