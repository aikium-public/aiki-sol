# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-05-12

### Changed (breaking)

The released model is now **Aiki-Sol-v2**: a single fine-tuned ESM-2 650M
backbone with a 6-output head (5 per-stringency + 1 stringency-marginal),
trained on the 147,574-row canonical-clean pool. This supersedes the
3-seed curve-head ensemble released in v0.1.x, which was trained on a
much smaller 21,571-row license-clean pool. The new checkpoint reaches
cohort-mean AUC 0.825 on the five external benchmarks (vs. 0.694–0.705
for the v0.1.x lineage on the same cohorts) and lifts the cohort-mean by
+0.10 to +0.16 AUC over the strongest published binary comparators on the
three measurably-zero-leakage cohorts.

API changes from v0.1.x:

- `Aikisol(checkpoint=...)` replaces `Aikisol(checkpoint_dir=...)`. The
  v2 checkpoint is a single `.pt` file, not a directory of seed files;
  the `seeds=` parameter is removed.
- `Aikisol.from_pretrained()` now expects a single file
  `aikisol_v2_canonical_147k_full.pt` (or, for research-tier use, the
  CC-BY-NC-ND `aikisol_v2_research_n3v2_229k_full.pt`).
- `PredictionResult` now carries a `prob_marginal` field
  (`np.ndarray [N]`) from the dedicated marginal head. The recommended
  single number for callers with unspecified downstream protocol.
- `Aikisol.predict(...)` no longer takes `return_per_seed`; it takes
  `return_raw` instead, which exposes the full `[N, 6]` output tensor
  if needed.
- `aikisol-predict` CLI: `--checkpoint` replaces `--checkpoint-dir` and
  `--seeds`. The new output CSV columns include `prob_marginal`
  alongside the per-stringency probabilities.
- HF backbone weight download skipped at inference (the released `.pt`
  contains the full fine-tuned backbone; we only need the tokenizer and
  config from HuggingFace).

### Companion Zenodo deposit (v2 version)

- New version of `10.5281/zenodo.20102408` shipping:
  - `aikisol_v2_canonical_147k_full.pt` (~2.4 GB, Apache 2.0)
  - 5 per-fold checkpoints `aikisol_v2_canonical_147k_fold{0..4}.pt`
    (Apache 2.0)
  - `aikisol_v2_research_n3v2_229k_full.pt` (~2.4 GB, CC-BY-NC-ND 4.0)
  - Canonical-147K training pool CSV (CC BY 4.0)
  - 5-fold cluster-disjoint split assignments
  - Per-cohort predictions and aggregated result JSONs

### Architecture summary (v2)

- Backbone: ESM-2 650M, full fine-tune.
- Head: `Linear(1280→256) → ReLU → Dropout(0.1) → Linear(256→6) → sigmoid`.
- Loss: BCE on the row's stringency output; BCE on the marginal head;
  cross-supervision of the mean of per-stringency outputs on
  stringency-unknown rows at weight 0.5; MSE on the eSol output for
  continuous mg/mL rows; squared-ReLU soft ordinal-monotonicity penalty
  at weight 0.2. Trained for 2 epochs (canonical) / 3 epochs (research
  tier); no early stopping; validation monitor-only.

## [0.1.1] — 2026-05-10

### Changed

- `aikisol.inference._build_model` now constructs the ESM-2 650M architecture
  from `AutoConfig.from_pretrained()` (downloads only the small `config.json`
  from HuggingFace) and instantiates the backbone with random initialization
  via `AutoModel.from_config()`. The released `.pt` checkpoint is a full
  fine-tune containing every backbone parameter, so the previous
  `AutoModel.from_pretrained(...)` call was downloading ~2.5 GB of HF
  weights only to immediately overwrite all of them with the .pt's saved
  state. Cuts cold-start by 30–90 s and ~2.5 GB of HF traffic per fresh
  container; same model, same predictions.
- `_load_seed` is now strict on missing keys (the random-init backbone
  requires every parameter to come from the .pt). Raises a clear error if
  any non-trivial backbone weight is absent from the checkpoint, so
  silently running with random-init weights is now structurally impossible.
  HuggingFace's non-persistent `position_ids` buffer is allowlisted.

### Compatibility

- Same checkpoint files (`released_curve_head_s{42,7,13}.pt`); no Zenodo
  re-upload needed.
- Same Python API; no caller-facing changes.
- Same predictions (verified via local test on representative sequences).

## [0.1.0] — 2026-05-09

Initial public release accompanying the manuscript
*"Protein solubility prediction has been over-engineered and under-curated:
a cluster-disjoint benchmark"* (Mysore et al., 2026). The journal/preprint
citation will be filled in on acceptance.

### Added

- `aikisol` Python package with:
  - `Aikisol.from_pretrained()` 3-seed ensemble loader (ESM-2 650M backbone +
    5-output curve head).
  - `Aikisol.predict(sequences)` returning per-protein `mean_prob` and
    per-stringency probabilities for the 5 centrifugation regimes
    (3,000×g/10 min, 6,000×g, 32,000×g, eSol/PURE 21,600×g/30 min, 100,000×g).
  - `normalize_sequence` and `normalize_sequences` for tag-aware sequence
    normalization (His6, HiBit, Strep, common linkers, initiator M);
    `normalize_sequences` raises if more than 5% of inputs retain tags
    after stripping.
  - One-shot `predict(sequence)` top-level helper.
  - `aikisol-predict` CLI accepting `--sequence`, `--fasta`, or `--csv`.
- Docker images:
  - `docker/Dockerfile` — CPU inference (no checkpoints baked in).
  - `docker/Dockerfile.full` — CPU inference with checkpoints baked in.
- Hosted live demo at `https://aikium--aikisol-landing-page.modal.run/`
  (read-only; weights and source served via Zenodo and this repository).
- Validation harness:
  - `validation/reproduce_paper_numbers.py` + manifest — verifies that every
    quantitative claim in the manuscript is reproduced by a deposited
    `result.json`.
- Companion Zenodo deposit (DOI assigned at publication) holding the
  license-clean training-pool parquet, cluster-disjoint splits, the
  3-seed checkpoint archive, predictions, results, and external benchmark
  artefacts. Sources with explicit redistribution restrictions
  (CC BY-NC-ND 4.0) are excluded; the `union_inventory_manifest.csv`
  documents inclusion/exclusion per source.

### Notes

- Inference convention: ESM-2 650M tokenization, max_length=1022, masked-mean
  pooling, sigmoid over the 5-dim curve head, arithmetic mean of seeds.
- Single-seed mode (`--seeds 42`) is documented as within ±0.012 AUC of the
  3-seed ensemble on every cohort we measured; useful when latency matters.
- The released artefact is the curve-head ensemble trained directly on the
  license-clean training pool. The distilled-student variant discussed in
  the manuscript is described in the paper but not part of this release.
