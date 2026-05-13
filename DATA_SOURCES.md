# Data sources, licenses, and redistribution policy

The Aiki-Sol Dataset assembles a license-clean *E. coli* solubility corpus
at two tiers: an Apache-tier 147K-row distribution and a research-tier
229K-row extension. This document is the authoritative inclusion table.

## The Aiki-Sol Dataset

### Canonical tier (Apache 2.0) — `aikisol_canonical_147k_train.csv`

The training pool of the released `aikisol_v2_canonical_147k_full.pt`
checkpoint and of the five per-fold checkpoints. 147,574 rows, all
license-clean for derivative redistribution under CC-BY 4.0.

| Upstream source family | Rows (≈) | License | Notes |
|---|---:|---|---|
| eSol/PURE (Niwa et al.) | ~3,200 | CC BY 4.0 | Cell-free intrinsic-aggregation labels; continuous mg/mL routed to MSE head during training |
| NESG SG (TargetTrack) | ~5,600 | CC0 | |
| MCSG / CSGID / NYSGRC (TargetTrack) | ~7,000 | CC0 | Centrifugation stringencies HIGH-confidence per consortium review |
| PSI:Biology (TargetTrack, NetSolP-style) | ~11,000 | BSD 3-Clause | |
| DeepSol (Khurana et al., ESMFold-augmented) | ~8,200 | CC BY 4.0 | |
| DeepSoluE | ~900 | CC BY 4.0 | |
| ProgSol-eSol / ProgSol-yeast | ~3,000 | unverified | License posture documented as unverified; included on conservative reading of the data deposit |
| Foldit (designed proteins) | ~120 | public | |
| Additional Apache-tier curated sources | varies | mixed CC-BY-OK | Per-source manifest in the deposit |
| **Total canonical tier** | **147,574** | | Mixed annotations: binary at known stringency, binary at unknown stringency, continuous eSol mg/mL |

### Research-tier extension (CC-BY-NC-ND 4.0) — training CSV NOT redistributed

The training pool of the research-tier
`aikisol_v2_research_n3v2_229k_full.pt` checkpoint. Adds 81,775 rows to
the canonical tier, dominated by:

| Upstream source family | Rows added (≈) | License |
|---|---:|---|
| ProtSolM training corpus (pdbsol / external / DSResSol / SoluProt / eSol splits) | ~57,000 | CC BY-NC-ND 4.0 |
| SoluProtMutDB mutational dataset | ~15,000 | CC BY-NC-ND 4.0 |
| Aikium curation rebalance over the canonical base | ~10,000 (non-exclusive memberships) | mixed; see deposit manifest |

The research-tier training CSV is **not redistributed in this deposit** —
its upstream sources prohibit derivative redistribution. The trained
checkpoint is shipped under CC-BY-NC-ND 4.0 for research use only. For
reproductions that require the underlying rows, follow the per-source
manifest in the deposit (`n3v2_source_manifest.csv`) to re-fetch from
the original distributors under your institutional licensing terms.

## Released checkpoints (`checkpoints/`)

| File | License | Notes |
|---|---|---|
| `aikisol_v2_canonical_147k_full.pt` | Apache 2.0 | Full-pool deployment; recommended default |
| `aikisol_v2_canonical_147k_fold{0..4}.pt` | Apache 2.0 | 5-fold cluster-disjoint per-fold checkpoints; supports leakage-free cross-cohort scoring without re-clustering |
| `aikisol_v2_research_n3v2_229k_full.pt` | CC-BY-NC-ND 4.0 | Research-tier; higher cohort-mean but in-distribution on ProtSolM-derived cohorts |

ESM-2 650M backbone weights are NOT redistributed here — the `aikisol`
package downloads the tokenizer and config from HuggingFace
(`facebook/esm2_t33_650M_UR50D`, licensed under MIT by Meta) on first use;
the model architecture is instantiated locally and the Aiki-Sol checkpoint
file supplies all weight values.

## Splits and partitions

| File | License | Notes |
|---|---|---|
| `splits/canonical_147k_5fold/fold_{0..4}.csv` | CC BY 4.0 | Sequence MD5 → fold assignment over the canonical tier, MMseqs2 25%-id 80%-cov clustering |
| `splits/protocol_strict_85k_5fold/fold_{0..4}.csv` | CC BY 4.0 | The 5-fold partition of the protocol-stratified-85K substrate used for the §sec:scaleup headline benchmark |

## Predictions and results

| Path | License | Notes |
|---|---|---|
| `predictions/canonical_147k_full/cohort_<cohort>_predictions.csv` | CC BY 4.0 | Per-row predictions of the canonical-tier full-pool checkpoint on each external cohort |
| `predictions/canonical_147k_fold{0..4}/cohort_<cohort>_predictions.csv` | CC BY 4.0 | Same for the 5 per-fold checkpoints |
| `predictions/n3v2_229k_full/cohort_<cohort>_predictions.csv` | CC-BY-NC-ND 4.0 | Research-tier per-row predictions; ProtSolM-derived cohort predictions are flagged in-distribution |
| `results/aikisol_v2_*.json` | CC BY 4.0 | Aggregated metrics + bootstrap CIs |

## Held-out in-house cohorts (NOT in this deposit)

The manuscript reports per-construct AUCs on five in-house cohorts (two
engineered-scaffold datasets and three nanobody datasets) used for
evaluation only. Their sequences and labels are not in any released
training pool and are not redistributed.

## Audit support

`canonical_147k_md5.txt` (in the companion Zenodo deposit) is one MD5
per training-pool sequence in the canonical tier. Use it for exact-match
overlap reporting against user-submitted sequences when auditing whether
a held-out evaluation set leaks into training.

## Licensing summary

- Code (companion repo): Apache 2.0.
- Released Apache-tier checkpoint + per-fold checkpoints: Apache 2.0.
- Released research-tier checkpoint: CC-BY-NC-ND 4.0.
- Aiki-Sol Dataset canonical tier (training rows, splits, predictions,
  manifests): CC BY 4.0.
- ESM-2 backbone weights: MIT (Meta), downloaded separately.
- Aiki-Sol Dataset research-tier extension: NOT redistributed verbatim
  due to upstream source-license restrictions; per-source manifest names
  them for independent re-collection.
