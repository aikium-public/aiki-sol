"""Tag-aware sequence normalization for AikiSol inference.

AikiSol training stripped purification and detection appendages from every
sequence in the training pool (hexahistidine, HiBit, Strep, FLAG, common
linkers, N-terminal Met). The deployment checkpoint is therefore in a
distribution that has no tags. Inference on raw tagged sequences is a
silent train/inference distribution mismatch and degrades AUC by
0.01-0.05 depending on the construct mix.

This module is the single canonical entrypoint:

    from aikisol.sequence_normalization import normalize_sequence

    norm, meta = normalize_sequence(raw_seq)

The batched form runs a 5%-retained-tag preflight that raises if the
input set still contains tags after normalization (the same invariant the
training pipeline enforced).

Reference: Manuscript Methods - sequence normalization.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# ── Tag patterns ──────────────────────────────────────────────────────────
HIBIT_RE = re.compile(r"(SGGGG)*VSGWRLFKKIS(GGGGS)*")
STREP_RE = re.compile(r"(SGGGG)*WSHPQFEK(GGGGS)*")

NTERM_WINDOW = 30
CTERM_WINDOW = 25

_LINKER_MOTIFS = (
    "TDPALRA", "ENLYFQG", "ENLYFQS", "ENLYFQ",
    "LEVLFQG", "LEVLFQP", "LEVLFQ",
    "LVPRGS", "LVPRG",
    "IEGR", "DDDDK", "SSGLVPR", "SSG", "GS",
)


@dataclass
class NormalizationMeta:
    had_tag: bool
    tag_type: str | None
    tag_position: str | None  # 'N' or 'C'
    extended_hibit: bool
    extended_strep: bool


def _strip_terminal_his(s: str) -> str:
    """Strip HHHHHH runs at either terminus within the windowed region.

    Internal H-runs are preserved (legitimate H-rich biology elsewhere).
    """
    head = s[:NTERM_WINDOW]
    if "HHHHHH" in head:
        for i in range(NTERM_WINDOW - 1, 4, -1):
            if i < len(s) and s[i] == "H" and s[i - 5:i + 1] == "HHHHHH":
                j = i + 1
                while j < len(s) and s[j] == "H":
                    j += 1
                for motif in _LINKER_MOTIFS:
                    if s[j:j + len(motif)] == motif:
                        j += len(motif)
                        break
                s = s[j:]
                break

    tail_start = max(0, len(s) - CTERM_WINDOW)
    tail = s[tail_start:]
    if "HHHHHH" in tail:
        idx = tail.find("HHHHHH")
        s = s[:tail_start + idx].rstrip("GS*")
    return s


def normalize_sequence(seq: str) -> tuple[str, NormalizationMeta]:
    """Strip purification and detection appendages and re-add initiator M.

    Pipeline (in order):
      1. Strip HiBit pattern (VSGWRLFKKIS, optionally flanked by GGGGS linkers).
      2. Strip Strep tag (WSHPQFEK, optionally flanked).
      3. Strip terminal His tag (HHHHHH within first 30 / last 25 residues),
         consuming common linker motifs immediately downstream of an N-terminal His.
      4. Re-add initiator M if missing.
    """
    if not isinstance(seq, str) or not seq:
        return "", NormalizationMeta(False, None, None, False, False)

    s = seq.upper().strip()
    extended_hibit = bool(HIBIT_RE.search(s))
    extended_strep = bool(STREP_RE.search(s))
    had_tag = False
    tag_type = None
    tag_position = None

    if extended_hibit:
        had_tag = True
        tag_type = "HiBit"
        tag_position = "C" if HIBIT_RE.search(s).start() > len(s) // 2 else "N"
        s = HIBIT_RE.sub("", s)
    if extended_strep:
        had_tag = True
        tag_type = (tag_type + "+Strep") if tag_type else "Strep"
        s = STREP_RE.sub("", s)
    s_before_his = s
    s = _strip_terminal_his(s)
    if s != s_before_his:
        had_tag = True
        tag_type = (tag_type + "+His") if tag_type else "His"

    if s and not s.startswith("M"):
        s = "M" + s

    return s, NormalizationMeta(had_tag, tag_type, tag_position, extended_hibit, extended_strep)


def normalize_sequences(
    seqs: Iterable[str],
    *,
    raise_on_retained_tags: bool = True,
    retained_tag_threshold: float = 0.05,
) -> list[str]:
    """Batch normalize + retained-tag preflight.

    Args:
        seqs: iterable of raw sequences.
        raise_on_retained_tags: if True (default), raise ValueError if
            either His6 or HiBit retention exceeds threshold.
        retained_tag_threshold: max retained-tag fraction (default 5%).

    Returns:
        list of normalized sequences (same length as input).

    Raises:
        ValueError if retention exceeds threshold and raise_on_retained_tags.
    """
    norm = [normalize_sequence(s)[0] for s in seqs]
    if not norm:
        return norm
    n = len(norm)
    n_his = sum(1 for s in norm if "HHHHHH" in s)
    n_hibit = sum(1 for s in norm if "VSGWRLFKKIS" in s)
    his_rate = n_his / n
    hibit_rate = n_hibit / n
    if raise_on_retained_tags and (
        his_rate > retained_tag_threshold or hibit_rate > retained_tag_threshold
    ):
        raise ValueError(
            f"Tag-stripping preflight failed: {his_rate:.1%} of normalized "
            f"sequences retain His6 and {hibit_rate:.1%} retain HiBit "
            f"(threshold {retained_tag_threshold:.1%}). The training pool "
            f"saw only stripped sequences; scoring tagged inputs is a "
            f"distribution mismatch."
        )
    return norm


__all__ = [
    "normalize_sequence",
    "normalize_sequences",
    "NormalizationMeta",
    "HIBIT_RE",
    "STREP_RE",
]
