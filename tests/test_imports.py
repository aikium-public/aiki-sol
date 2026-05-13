"""Smoke-import test — package surface must remain stable across releases."""
from __future__ import annotations


def test_package_surface():
    import aikisol
    assert aikisol.__version__
    # Re-exports.
    assert hasattr(aikisol, "Aikisol")
    assert hasattr(aikisol, "PredictionResult")
    assert hasattr(aikisol, "predict")
    assert hasattr(aikisol, "normalize_sequence")
    assert hasattr(aikisol, "normalize_sequences")
    assert isinstance(aikisol.STRINGENCY_LABELS, tuple)
    assert len(aikisol.STRINGENCY_LABELS) == 5


def test_cli_imports():
    from aikisol.cli import main, parse_fasta
    assert callable(main)
    assert callable(parse_fasta)


def test_stringency_label_names():
    from aikisol import STRINGENCY_LABELS
    expected = ("3000g_10min", "6000g", "32000g", "eSol_21600g_30min", "100000g")
    assert STRINGENCY_LABELS == expected
