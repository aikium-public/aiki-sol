"""FASTA parser tests for the CLI helper."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aikisol.cli import parse_fasta


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "x.fasta"
    p.write_text(text)
    return p


def test_two_records(tmp_path):
    p = _write(tmp_path, ">a\nMKLI\nTV\n>b\nMAE\n")
    rec = list(parse_fasta(p))
    assert rec == [("a", "MKLITV"), ("b", "MAE")]


def test_label_takes_first_token(tmp_path):
    p = _write(tmp_path, ">a description text\nMK\n")
    rec = list(parse_fasta(p))
    assert rec[0][0] == "a"


def test_blank_lines_tolerated(tmp_path):
    p = _write(tmp_path, "\n\n>a\nMK\n\nLI\n")
    rec = list(parse_fasta(p))
    assert rec == [("a", "MKLI")]


def test_empty_file(tmp_path):
    p = _write(tmp_path, "")
    rec = list(parse_fasta(p))
    assert rec == []
