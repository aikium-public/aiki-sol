# Release notes

## v0.2.0

Aiki-Sol v2 is a single fine-tuned ESM-2 650M backbone with a 6-output
head: five per-stringency probabilities (3,000×g/10 min, 6,000×g,
32,000×g, eSol/PURE 21,600×g/30 min, 100,000×g) plus a dedicated
stringency-marginal output for inputs without a specified protocol.

### Model

- Backbone: ESM-2 650M, full fine-tune.
- Head: `Linear(1280→256) → ReLU → Dropout(0.1) → Linear(256→6) → sigmoid`.
- Loss: BCE on the row's stringency output; BCE on the marginal head;
  cross-supervision of the mean of per-stringency outputs on
  stringency-unknown rows at weight 0.5; MSE on the eSol output for
  continuous mg/mL rows; squared-ReLU soft ordinal-monotonicity penalty
  at weight 0.2. Two epochs on the canonical-tier pool, three on the
  research-tier pool; validation monitor-only.

### Training data — the Aiki-Sol Dataset

- Apache-tier `aikisol_canonical_147k_train.csv` — 147,574 license-clean
  *E. coli* rows with per-protein centrifugation-regime annotation,
  cluster-disjoint at 25% identity into 5 folds.
- Research-tier extension to 229,349 rows under CC-BY-NC-ND 4.0; the
  training CSV is not redistributed (upstream licences restrict
  derivative redistribution); the checkpoint is.

### Companion Zenodo deposit

`10.5281/zenodo.20151817` ships:

- `aikisol_v2_canonical_147k_full.pt` (~2.4 GB, Apache 2.0)
- 5 per-fold checkpoints `aikisol_v2_canonical_147k_fold{0..4}.pt`
- `aikisol_v2_research_n3v2_229k_full.pt` (~2.4 GB, CC-BY-NC-ND 4.0)
- Canonical-147K training-pool CSV and 5-fold cluster-disjoint split
  assignments
- Per-cohort prediction CSVs and aggregated result JSONs

### Other

- Hosted live demo at
  `https://aikium--aikisol-landing-page.modal.run/`.
- Validation harness: `validation/reproduce_paper_numbers.py` verifies
  every quantitative claim in the companion paper against a deposited
  `result.json`.
