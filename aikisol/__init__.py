"""Aiki-Sol v2 — per-protocol calibrated protein solubility prediction.

Quick start:

    from aikisol import Aikisol
    model = Aikisol.from_pretrained()
    out = model.predict(["MAEILVTQ...", "MKLITV..."])
    out.mean_prob              # mean of the 5 per-stringency outputs
    out.prob_marginal          # the dedicated stringency-marginal head
    out.per_stringency         # {3000g_10min: array, 6000g: array, ...}

Or one-shot for a single sequence:

    from aikisol import predict
    result = predict("MAEILVTQNMK...")
    # {'mean_prob': 0.735, 'prob_marginal': 0.71, 'prob_3000g_10min': 0.81, ...}

Inputs are tag-stripped before scoring (His6, HiBit, Strep, common linkers,
N-terminal Met handled). Batch helpers raise if the retained-tag rate after
normalization exceeds 5% — passing tagged sequences silently degrades AUC by
0.01-0.05 because the training pool was tag-stripped.

Released checkpoint (v2): single fine-tuned ESM-2 650M with a 6-output head
(5 per-stringency outputs + a dedicated marginal output for
stringency-unknown queries). Trained on the 147,574-row canonical-clean
training pool under Apache 2.0.

A research-tier checkpoint trained on the larger 229,349-row n3v2 pool
(which includes CC-BY-NC-ND-tier source data) is available via the
companion Zenodo deposit under a research-use notice; pass its path to
`Aikisol(checkpoint=...)` if you have institutional clearance for the
upstream license terms.

See README for installation, citation, license terms.
"""
from aikisol.inference import (
    Aikisol,
    PredictionResult,
    STRINGENCY_LABELS,
    predict,
)
from aikisol.sequence_normalization import (
    normalize_sequence,
    normalize_sequences,
)

__version__ = "0.2.0"
__all__ = [
    "Aikisol",
    "PredictionResult",
    "STRINGENCY_LABELS",
    "predict",
    "normalize_sequence",
    "normalize_sequences",
    "__version__",
]
