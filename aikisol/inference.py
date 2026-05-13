"""Aiki-Sol v2 inference — single fine-tuned ESM-2 650M, six-output head.

The deployment artefact (since v0.2.0) is a single ESM-2 650M backbone
fine-tuned end-to-end on the 147,574-row canonical-clean training pool
(`canonical-147K`, Apache 2.0). Output head:

    ESM-2 650M -> masked-mean pool over residues
                -> Linear(1280, 256) -> ReLU -> Dropout(0.1)
                -> Linear(256, 6) -> sigmoid

The 6 outputs are:

    p_0 = P(soluble | 3,000 g x 10 min)
    p_1 = P(soluble | 6,000 g)
    p_2 = P(soluble | 32,000 g)
    p_3 = P(soluble | 21,600 g x 30 min, eSol/PURE)
    p_4 = P(soluble | 100,000 g)
    p_5 = P(soluble | sequence)  -- marginal head for stringency-unknown rows

The marginal head exists because the training pool contains a large
fraction of rows whose source record does not recover the centrifugation
stringency; the marginal output is the right number to report when the
caller does not know which protocol's prediction they need.

Inference convention:
    1. Tag-aware sequence normalization (sequence_normalization.normalize_sequence)
    2. ESM-2 650M tokenization, max_length=1022, truncation=True
    3. Single forward pass through the released checkpoint
    4. Sigmoid over the 6-dim head; per-stringency probs are p_0..p_4 and
       the stringency-marginal prob is p_5.

Usage:

    from aikisol import Aikisol
    model = Aikisol.from_pretrained()  # loads aikisol_v2_canonical_147k_full.pt
    out = model.predict(["MKLI...", "MAEN..."])
    out.mean_prob               # per-protein mean across the 5 per-stringency outputs
    out.prob_marginal           # the dedicated marginal head, p_5
    out.per_stringency          # dict of all five per-stringency outputs
    out.normalized_sequences    # tag-stripped sequences actually scored
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from aikisol.sequence_normalization import normalize_sequences

ESM2_650M = "facebook/esm2_t33_650M_UR50D"
EMBED_DIM = 1280
HEAD_HIDDEN = 256
N_OUTPUTS = 6
N_STRINGENCIES = 5  # outputs 0..4; output 5 is the marginal head
STRINGENCY_LABELS = ("3000g_10min", "6000g", "32000g", "eSol_21600g_30min", "100000g")

# Default checkpoint dir — set AIKISOL_CKPT_DIR env var to override.
DEFAULT_CKPT_DIR = os.environ.get(
    "AIKISOL_CKPT_DIR",
    str(Path.home() / ".cache" / "aikisol" / "checkpoints"),
)
DEFAULT_CKPT_NAME = "aikisol_v2_canonical_147k_full.pt"


def _build_model() -> nn.Module:
    """Build the architecture (no pretrained weights downloaded).

    The released checkpoint is a full fine-tune — every backbone parameter
    is in the .pt file and is restored by `_load_checkpoint`. We download
    only the ~1 KB `config.json` from HuggingFace to instantiate the right
    architecture with random init.
    """
    config = AutoConfig.from_pretrained(ESM2_650M)
    backbone = AutoModel.from_config(config)

    class _V2Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.esm = backbone
            self.head = nn.Sequential(
                nn.Linear(EMBED_DIM, HEAD_HIDDEN),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(HEAD_HIDDEN, N_OUTPUTS),
            )

        def forward(self, input_ids, attention_mask):
            out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return torch.sigmoid(self.head(pooled))  # [B, 6]

    return _V2Model()


# State-dict keys that legitimately won't round-trip through the .pt:
#   - "coral_head.*" was a training-time auxiliary head; not reconstructed
#     at inference.
#   - "esm.embeddings.position_ids" and "*.rotary_embeddings.inv_freq" are
#     non-persistent buffers that HuggingFace regenerates at model
#     construction (they were never in the .pt to begin with).
#   - "esm.embeddings.position_embeddings.weight" is an unused absolute-
#     position-embedding parameter that HF still instantiates even when
#     `position_embedding_type='rotary'` (ESM-2's setting); its values are
#     never read at forward time.
_BENIGN_UNEXPECTED_PREFIXES = ("coral_head.",)
_BENIGN_UNEXPECTED_SUFFIXES = (".rotary_embeddings.inv_freq",)
_BENIGN_MISSING_PREFIXES = (
    "esm.embeddings.position_ids",
    "esm.embeddings.position_embeddings.weight",
)
_BENIGN_MISSING_SUFFIXES = (
    ".rotary_embeddings.inv_freq",
    ".position_ids",
)


def _load_checkpoint(ckpt_path: Path) -> nn.Module:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Aiki-Sol v2 checkpoint not found at {ckpt_path}. Set AIKISOL_CKPT_DIR "
            f"or call Aikisol.from_pretrained(checkpoint=...) with the path to the "
            f"deposit's `aikisol_v2_canonical_147k_full.pt` file (or the research-"
            f"tier `aikisol_v2_research_n3v2_229k_full.pt` for CC-BY-NC-ND use)."
        )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = sd.get("model_state_dict", sd)

    model = _build_model()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    real_missing = [
        k for k in missing
        if not any(k.startswith(p) for p in _BENIGN_MISSING_PREFIXES)
        and not any(k.endswith(s) for s in _BENIGN_MISSING_SUFFIXES)
    ]
    if real_missing:
        head = real_missing[:5]
        rest = max(0, len(real_missing) - 5)
        suffix = f" (and {rest} more)" if rest else ""
        raise RuntimeError(
            f"Checkpoint {ckpt_path} is missing weights for: {head}{suffix}. "
            f"The released Aiki-Sol-v2 checkpoint should contain a full backbone "
            f"+ 6-output head state dict."
        )
    real_unexpected = [
        k for k in unexpected
        if not any(k.startswith(p) for p in _BENIGN_UNEXPECTED_PREFIXES)
        and not any(k.endswith(s) for s in _BENIGN_UNEXPECTED_SUFFIXES)
    ]
    if real_unexpected:
        head = real_unexpected[:5]
        rest = max(0, len(real_unexpected) - 5)
        suffix = f" (and {rest} more)" if rest else ""
        raise RuntimeError(
            f"Checkpoint {ckpt_path} has unexpected state-dict keys: "
            f"{head}{suffix}. Loader and checkpoint architectures disagree."
        )
    return model.eval()


@dataclass
class PredictionResult:
    mean_prob: np.ndarray        # [N], mean of the 5 per-stringency outputs per protein
    prob_marginal: np.ndarray    # [N], the dedicated marginal head (p_5)
    per_stringency: dict         # {stringency_label: np.ndarray[N]} for outputs 0..4
    raw: np.ndarray | None       # optional [N, 6] full output tensor if requested
    normalized_sequences: list[str]


class Aikisol:
    """Aiki-Sol v2 — single fine-tuned ESM-2 650M with a 6-output head.

    Args:
        checkpoint: path to a single .pt file
            (`aikisol_v2_canonical_147k_full.pt` for the Apache-tier deployment
             checkpoint, or `aikisol_v2_research_n3v2_229k_full.pt` for the
             research-tier checkpoint trained on the broader CC-BY-NC-ND pool).
        device: 'cuda', 'cpu', or 'auto' (default: pick CUDA if available).
        max_length: max ESM-2 input length, default 1022.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        max_length: int = 1022,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_650M)
        self.checkpoint = Path(checkpoint)
        self.model = _load_checkpoint(self.checkpoint).to(device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path | None = None,
        *,
        checkpoint_dir: str | Path | None = None,
        **kwargs,
    ) -> "Aikisol":
        """Load the released v2 deployment checkpoint.

        If `checkpoint` is provided, load that file directly. Otherwise we
        look in `checkpoint_dir` (or `AIKISOL_CKPT_DIR`, or
        `~/.cache/aikisol/checkpoints/`) for
        `aikisol_v2_canonical_147k_full.pt`.
        """
        if checkpoint is None:
            d = Path(checkpoint_dir) if checkpoint_dir else Path(DEFAULT_CKPT_DIR)
            checkpoint = d / DEFAULT_CKPT_NAME
        return cls(checkpoint=checkpoint, **kwargs)

    def _score_one(
        self,
        sequences: list[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Returns [N, 6] probability tensor."""
        N = len(sequences)
        out = np.zeros((N, N_OUTPUTS), dtype=np.float32)
        for i in range(0, N, batch_size):
            batch = sequences[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                p = self.model(ids, mask)
            out[i:i + len(batch)] = p.float().cpu().numpy()
        if not np.isfinite(out).all():
            raise RuntimeError("Non-finite predictions emitted; check input.")
        return out

    def predict(
        self,
        sequences: str | Sequence[str],
        *,
        normalize: bool = True,
        batch_size: int = 8,
        return_raw: bool = False,
    ) -> PredictionResult:
        """Score sequences. Returns PredictionResult.

        Args:
            sequences: string or list of strings.
            normalize: if True (default), apply tag-aware sequence
                normalization; raises if retained-tag rate > 5% post-strip.
            batch_size: inference batch size.
            return_raw: if True, the result includes the raw [N, 6] tensor.

        Output structure:
            mean_prob:     mean of the 5 per-stringency outputs (p_0..p_4)
            prob_marginal: the dedicated marginal head (p_5) — recommended
                           single number for a stringency-unknown caller
            per_stringency: dict mapping stringency name to per-protein prob
            raw:           [N, 6] if return_raw=True, else None
        """
        single = isinstance(sequences, str)
        seqs = [sequences] if single else list(sequences)
        if not seqs:
            raise ValueError("Empty sequence list.")
        normalized = normalize_sequences(seqs) if normalize else list(seqs)

        probs = self._score_one(normalized, batch_size=batch_size)  # [N, 6]
        per_strg = probs[:, :N_STRINGENCIES]                          # [N, 5]
        marginal = probs[:, N_STRINGENCIES]                           # [N]
        mean_prob = per_strg.mean(axis=1)                             # [N]
        per_stringency = {
            STRINGENCY_LABELS[k]: per_strg[:, k] for k in range(N_STRINGENCIES)
        }
        return PredictionResult(
            mean_prob=mean_prob,
            prob_marginal=marginal,
            per_stringency=per_stringency,
            raw=probs if return_raw else None,
            normalized_sequences=normalized,
        )


def predict(
    sequence: str,
    *,
    checkpoint: str | Path | None = None,
) -> dict:
    """One-shot top-level API. Score a single sequence with the released
    deployment checkpoint.

    Returns a flat dict with keys ``mean_prob``, ``prob_marginal``,
    ``prob_<stringency>``, and ``normalized_sequence``.
    """
    model = Aikisol.from_pretrained(checkpoint=checkpoint)
    out = model.predict(sequence)
    flat = {
        "mean_prob": float(out.mean_prob[0]),
        "prob_marginal": float(out.prob_marginal[0]),
        "normalized_sequence": out.normalized_sequences[0],
    }
    for k, v in out.per_stringency.items():
        flat[f"prob_{k}"] = float(v[0])
    return flat


__all__ = ["Aikisol", "PredictionResult", "predict", "STRINGENCY_LABELS"]
