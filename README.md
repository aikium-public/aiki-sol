# Aiki-Sol

**Per-protocol calibrated protein solubility prediction.**

Aiki-Sol scores a protein sequence against five centrifugation-defined solubility
regimes (3,000×g/10 min, 6,000×g, 32,000×g, eSol/PURE 21,600×g/30 min, 100,000×g)
and returns an ensemble-averaged probability per regime, plus a stringency-
marginal probability for when the downstream protocol is unspecified.

The released checkpoint (v0.2.0, **Aiki-Sol-v2**) is a single fine-tuned ESM-2
650M backbone with a 6-output head: outputs 1–5 are per-stringency
`P(soluble | regime)` and output 6 is a dedicated marginal head trained on
stringency-unknown rows, recommended when the caller doesn't know which
protocol matters. Trained on the 147,574-row canonical-clean pool under
Apache 2.0; the loss adds a soft monotonicity penalty across consecutive
per-stringency outputs during training (see the companion paper §Methods).

A research-tier checkpoint (`aikisol_v2_research_n3v2_229k_full.pt`) trained
on the larger 229,349-row pool (which adds CC-BY-NC-ND-tier upstream sources)
is available in the same Zenodo deposit under a research-use notice; load it
by passing the path to `Aikisol.from_pretrained(checkpoint=...)`.

For methods, benchmarks, and limitations, see the companion paper *(citation
below)*.

---

## Install

```bash
pip install aikisol
```

Requires Python ≥ 3.10. The first `Aikisol.from_pretrained()` call downloads
the ESM-2 650M tokenizer + config from HuggingFace (~1 MB total; backbone
weights come from the Aiki-Sol checkpoint, not from HF). The Aiki-Sol
deployment checkpoint (`aikisol_v2_canonical_147k_full.pt`, ~2.4 GB) must be
obtained from the Zenodo deposit listed below and placed at
`~/.cache/aikisol/checkpoints/` (or set `AIKISOL_CKPT_DIR`).

## Quick start

```python
from aikisol import Aikisol

model = Aikisol.from_pretrained()
out = model.predict([
    "MKLITVLVLALLAVAVAFPV",
    "MAEILVTQNMK...",
])
out.mean_prob              # np.ndarray [N], mean of the 5 per-stringency outputs
out.prob_marginal          # np.ndarray [N], the dedicated stringency-marginal output
out.per_stringency         # {3000g_10min: [...], 6000g: [...], ...}
out.normalized_sequences   # tag-stripped sequences actually scored
```

One-shot:

```python
from aikisol import predict
result = predict("MAEILVTQNMK...")
# {'mean_prob': 0.735, 'prob_marginal': 0.71, 'prob_3000g_10min': 0.81, ...}
```

CLI:

```bash
aikisol-predict --sequence "MAEILVTQNMK..." --pretty
aikisol-predict --fasta my.fasta --out predictions.csv
aikisol-predict --csv input.csv --seq-col sequence --id-col id --out predictions.csv
```

To score with the research-tier checkpoint (CC-BY-NC-ND, larger training
pool, in-distribution on ProtSolM-derived test cohorts so generalization
numbers are inflated on those):

```bash
aikisol-predict --sequence "MAEIL..." \
    --checkpoint /path/to/aikisol_v2_research_n3v2_229k_full.pt
```

## Tag-aware sequence normalization (important)

Aiki-Sol was trained on tag-stripped sequences. Scoring raw tagged sequences
(His6, HiBit, Strep, common linkers) is a silent train/inference distribution
mismatch and silently degrades AUC by 0.01–0.05.

The package strips these tags by default (`normalize=True` in `predict`) and
raises if more than 5% of inputs still retain them post-strip. Disable only
if your inputs are already clean (`--no-normalize`).

```python
from aikisol import normalize_sequence
normalized, meta = normalize_sequence("MGSSHHHHHHGSGSGEDQAEILVTQNMK")
# normalized = "MGSGEDQAEILVTQNMK"
# meta.had_tag, meta.tag_type, meta.tag_position
```

## Which output to report

- **`prob_marginal`** is the recommended single number when the downstream
  protocol is unspecified (the marginal head was trained explicitly for this
  case).
- **`per_stringency[regime]`** is the right number when the assay protocol
  matches one of the five labeled regimes.
- **`mean_prob`** is the average across the 5 per-stringency outputs; useful
  for ranking proteins symmetrically against binary comparators.

## Model details

- Backbone: ESM-2 650M (`facebook/esm2_t33_650M_UR50D`), masked-mean pooling.
- Head: `Linear(1280→256) → ReLU → Dropout(0.1) → Linear(256→6) → sigmoid`.
- Released checkpoint (Apache-tier): `aikisol_v2_canonical_147k_full.pt`.
- Research-tier checkpoint (CC-BY-NC-ND): `aikisol_v2_research_n3v2_229k_full.pt`.
- Training pool sizes: 147,574 (canonical, Apache); 229,349 (n3v2, research-tier).
- Splits: MMseqs2 cluster-disjoint at 25% identity, 5 folds for evaluation.

## Redistribution & license policy

| Asset | License |
|---|---|
| Code (this repo) | Apache 2.0 |
| Released Apache-tier checkpoint | Apache 2.0 |
| Research-tier checkpoint | CC-BY-NC-ND 4.0 |
| Canonical-147K training pool (Zenodo) | CC-BY 4.0 |
| n3v2-229K training pool | not redistributed (upstream license restrictions) |
| ESM-2 backbone weights | MIT (Meta), downloaded from HuggingFace |

Sources with explicit prohibitions on commercial use or repackaging are
excluded from the Apache-tier deposit. The n3v2-229K pool extends the
canonical pool with these sources for the research-tier checkpoint only;
its training CSV is not redistributed verbatim. For reproductions that
require those rows, follow the manifest pointers to the original
distributors.

The in-house held-out cohorts (two engineered-scaffold datasets and three
nanobody datasets) used in the paper for *evaluation only* are not in any
released training pool and not redistributed.

## Citation

If you use Aiki-Sol in your work, please cite:

```bibtex
@article{aikisol2026,
  title   = {Protein solubility prediction is bottlenecked by
             training-pool curation, not architectural complexity},
  author  = {Mysore, Venkatesh and others},
  year    = {2026},
  note    = {Companion code/data: github.com/aikium-public/aiki-sol;
             Zenodo DOI: 10.5281/zenodo.20102408;
             journal/preprint citation added on acceptance.}
}
```

## Links

- Paper: *(arXiv/bioRxiv link added on submission)*
- Zenodo deposit: [10.5281/zenodo.20102408](https://doi.org/10.5281/zenodo.20102408)
- Hosted demo: `https://aikium--aikisol-landing-page.modal.run/`
- Issues / questions: https://github.com/aikium-public/aiki-sol/issues

## Contact

Venkatesh Mysore — venkatesh@aikium.com
