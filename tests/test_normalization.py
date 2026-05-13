"""Unit tests for tag-aware sequence normalization.

These run in CI without any model checkpoints — the normalization module is
pure-Python and self-contained.
"""
from __future__ import annotations

import pytest

from aikisol import normalize_sequence, normalize_sequences


def test_strip_n_terminal_his_with_linker():
    raw = "MGSSHHHHHHGSGSGEDQAEILVTQNMK"
    norm, meta = normalize_sequence(raw)
    assert norm.startswith("M")
    assert "HHHHHH" not in norm
    assert "GEDQAEILVTQNMK" in norm
    assert meta.had_tag is True
    assert meta.tag_type is not None and "His" in meta.tag_type


def test_strip_hibit_with_linkers():
    # HiBit + (GGGGS) repeat embedded between an Fv-like body and the tag.
    raw = "AVQLQESGGGSVQAGGSLKLTCAAGGSHHHHHHGGSVSGWRLFKKISLE"
    norm, meta = normalize_sequence(raw)
    assert "VSGWRLFKKIS" not in norm
    assert "HHHHHH" not in norm
    assert norm.startswith("M")
    assert meta.had_tag is True
    assert meta.extended_hibit is True


def test_no_tag_returns_unchanged_apart_from_initiator_m():
    raw = "AEILVTQNMKK"
    norm, meta = normalize_sequence(raw)
    assert norm == "M" + raw
    assert meta.had_tag is False
    assert meta.tag_type is None


def test_internal_h_run_is_preserved():
    # Long H-run beyond both terminal windows (first 30 / last 25 residues)
    # is preserved — only terminal His tags are stripped.
    raw = "MAEILVTQNMKAEILVTQNMKAEILVTQNMK" + "HHHHHH" + "AEILVTQNMKAEILVTQNMKAEILVTQNMKAEIL"
    norm, meta = normalize_sequence(raw)
    assert "HHHHHH" in norm
    assert meta.had_tag is False  # internal H-run is not flagged as a tag


def test_empty_input():
    norm, meta = normalize_sequence("")
    assert norm == ""
    assert meta.had_tag is False


def test_batch_preflight_raises_above_threshold():
    # Sequence with an internal H6-run that survives normalization (because it
    # falls outside both terminal windows). One such input out of one is 100%
    # retention and must trigger the preflight.
    internal_h = (
        "MAEILVTQNMKAEILVTQNMKAEILVTQNMK"   # 31aa pre-run
        "HHHHHH"                            # internal H6
        "AEILVTQNMKAEILVTQNMKAEILVTQNMKAEIL" # 34aa post-run (>25 -> outside C window)
    )
    with pytest.raises(ValueError):
        normalize_sequences([internal_h])


def test_batch_preflight_disabled_when_flag_off():
    internal_h = (
        "MAEILVTQNMKAEILVTQNMKAEILVTQNMK"
        "HHHHHH"
        "AEILVTQNMKAEILVTQNMKAEILVTQNMKAEIL"
    )
    out = normalize_sequences([internal_h] * 3, raise_on_retained_tags=False)
    assert len(out) == 3
    assert all("HHHHHH" in s for s in out)


def test_batch_preflight_clean_input():
    seqs = ["MAEILVTQNMK", "MGSSHHHHHHGSGSGEDQAEILVTQNMK"]
    out = normalize_sequences(seqs)
    assert all(s.startswith("M") for s in out)
    assert all("HHHHHH" not in s and "VSGWRLFKKIS" not in s for s in out)
