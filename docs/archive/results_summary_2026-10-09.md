---
title: Results summary (standard regime)
created: 2026-10-09
status: superseded
tags: [paper, covet, results, summary]
---

# Results summary (standard regime, 2026-10-09)

> Historical snapshot; its numbers are superseded. It is kept for its reasoning,
> in particular the AMI/noise confound (finding 5) and the framing of the
> contribution as a trade-off.
>
> - The Fed and Yelp results it used were trained on the old chunking and were
>   archived on 2026-10-10 (`results/archive/*_old_chunking_*.zip`). Their reruns
>   replace them.
> - Gadarian and ANES were rerun on dataset-specific topic-count grids from
>   2026-10-10 (see [src/topic_grids.py](../../src/topic_grids.py)).
> - The Trump warning below is outdated: the `trump_s25000` metrics were already
>   recomputed with the unpadded protocol (`unpadded_2026_10`).

All numbers are paired deltas against each variant's catalog reference. Runs are matched by seed and requested topic count, averaged per dataset (3 seeds × k ∈ {10, 20, 30, 40, 50}), and then averaged across datasets. Trump is the 25k sample (`trump_s25000`). The numbers match the tables written by `scripts/analysis/make_paper_outputs.py` (see [paper_results_plan.md](../paper_results_plan.md)).

> [!warning] Preliminary
> - Every `trump_s25000` run was produced at revision `17fde45`. That revision predates the unpadded coherence evaluation (`0e3fd08`), so Trump's coherence is still on the old protocol. Run `scripts/analysis/recompute_topic_metrics.py` on it before the numbers are final.
> - The Fed weighted-Append runs (w = 0, w = 0.05) are still unmerged raw files and have not been recomputed.

## Headline

Adding metadata trades **topic–metadata alignment** against **topic diversity**, and how much metadata enters the model sets where on that curve you land. Coherence stays roughly flat everywhere on the curve. Noise and topic count fall as alignment rises, but mostly where the clusters collapse onto metadata groups. No variant improves on its reference on every dataset for any quality metric.

## Findings

### 1. Coherence is roughly unchanged

Mean ΔNPMI is between −0.03 and +0.01 for every variant, and mean ΔC_V is between −0.05 and +0.01. No variant wins consistently. Every Holm-adjusted p is ≥ 0.88. With 5 datasets the smallest attainable two-sided p is 0.0625, so the tests describe consistency across datasets rather than confirm effects.

### 2. Alignment rises only when metadata drives the clustering, and diversity pays for it

| Variant (vs reference) | ΔAMI | ΔNPMI | ΔIRBO | ΔTopic Diversity | ΔK | Δnoise (pp) |
|---|---|---|---|---|---|---|
| Append UMAP (vs UMAP + HDBSCAN) | +0.20 | −0.021 | −0.073 | −0.120 (1/4) | −2.5 | −14.4 |
| Weighted Append, w = 0.1 | +0.02 | +0.008 | +0.004 | +0.009 | +0.1 | +3.7 |
| Aligned UMAP | +0.00 | +0.002 | +0.002 | +0.001 | +0.3 | +5.1 |
| MV-HDBSCAN | +0.22 | −0.029 | −0.159 | −0.279 (0/4) | −4.1 | −22.2 |
| Feature-Stacking HDBSCAN | +0.21 | −0.012 | −0.186 | −0.296 (0/4) | −5.3 | −16.8 |
| Co-Reg MV Spectral (vs UMAP + Spectral) | +0.00 | −0.018 | −0.012 | −0.024 | 0 | — |
| Co-Train MV Spectral | +0.15 | −0.053 | −0.031 | −0.069 | 0 | — |

- The variants that lose nothing (Aligned UMAP, Co-Reg) also gain no alignment. They effectively ignore the metadata.
- The variants that gain about +0.2 AMI lose 0.12–0.30 diversity and 0.07–0.19 IRBO. On Yelp, diversity drops by 0.43–0.60.
- Weighted Append at w = 0.1 is the only setting with a small alignment gain at no measurable quality cost. The dose-response sweep shows AMI rising with w, and diversity collapsing from w = 0.2 on Gadarian and from w = 0.3 on Yelp and ANES.
- Part of the AMI gain is by construction. On Gadarian, Append's topics reproduce the 14 metadata profiles exactly (7 party levels × 2 treatment arms).

### 3. Fewer topics mostly means Yelp collapsing

The HDBSCAN models are capped by BERTopic's `nr_topics` reduction, so on ANES, Fed and Trump the realized topic count stays at the requested k − 1. Realized topic counts:

| Dataset | Reference | Append | MV-HDBSCAN | Feature-Stacking |
|---|---|---|---|---|
| Yelp | 29 | 16.5 | 13.6 | 13.0 |
| Gadarian | 13 | 13.2 | 13.8 | 10.3 |
| ANES | 29 | 29 | 27.3 | 26.4 |
| Fed | 29 | 29 | 29 | 29 |

The mean ΔK of −2.5 to −5.3 is therefore driven almost entirely by Yelp, and that collapse explains much of the diversity loss. PCA + MV K-Means (about 7 of 30 requested topics on every dataset) is a different, degenerate failure and is excluded.

### 4. Less noise is not universal

Mean noise share for the reference is 27.4%. It drops to 13.0% for Append, 8.2% for Feature-Stacking and 2.8% for MV-HDBSCAN. Aligned UMAP *increases* noise on all 5 datasets (32.5%), and Append increases it on ANES (7.3% → 20.0%). The near-zero noise on Yelp and Gadarian comes together with the topic collapse: the documents are absorbed into a few metadata-shaped clusters.

### 5. AMI counts HDBSCAN noise as a cluster, which confounds it with noise share

The noise label is scored as one more cluster. A large noise cluster that is mostly independent of the metadata adds partition entropy without adding mutual information, so it lowers AMI. Absorbing those documents into metadata-shaped clusters raises AMI even if the non-noise topics are unchanged. Consequences:

- **Within the HDBSCAN family, the AMI gain and the noise reduction partly measure the same thing.** The +0.20 to +0.22 AMI of Append, MV-HDBSCAN and Feature-Stacking cannot be separated from their −14 to −22 pp noise change with the current metric.
- **Across families, AMI is not like-for-like.** UMAP + Spectral, Co-Reg, Co-Train, TriTopic and STM have no noise cluster.
- **The gains are not only a noise effect.** Co-Train MV Spectral has no noise and still gains +0.15. Weighted Append at w = 0.1 *increases* noise (+3.7 pp) and still gains +0.02. On Gadarian, Append's topics reproduce the metadata partition exactly.
- **The bias can also run the other way.** Aligned UMAP's extra noise (+5.1 pp) may hide a small alignment gain.

**Recommended sensitivity check:** recompute AMI on non-noise documents only, and report it next to the current AMI. That restriction has its own bias, because it keeps only the documents HDBSCAN was confident about, so present it as a bound rather than as a replacement.

## Implications for the paper

- Frame the contribution as a controllable trade-off, not as a quality improvement.
- Describe AMI as a measure of how strongly metadata shapes the partition, never as quality. State that noise is scored as a cluster, and show Δnoise and ΔK next to every ΔAMI.
- Present weighted Append (w ≈ 0.1) as the operating point with low-cost alignment, and the w sweep as the mechanism.
- Recompute the `trump_s25000` metrics, and add the noise-excluded AMI before finalizing.
