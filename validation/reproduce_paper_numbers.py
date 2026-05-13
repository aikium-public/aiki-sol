#!/usr/bin/env python3
"""Verify deposited result JSONs reproduce the manuscript headlines.

Reads a manifest file (default: `validation/paper_numbers_manifest.yaml`,
shipped as `.example.yaml` until the maintainer fills in real expected
values). The manifest lists every quantitative claim in the paper that must
be backed by a deposited artefact. Each entry points to a JSON file
(relative to --deposit-root), a dotted field path inside that JSON, and the
expected value with a tolerance. Mismatches fail loudly so that submit-day
catches them.

Usage (from repo root, after staging the Zenodo bundle):
    cp validation/paper_numbers_manifest.example.yaml validation/paper_numbers_manifest.yaml
    # ...edit the .yaml with real expected values...
    python validation/reproduce_paper_numbers.py \
        --deposit-root ~/aiki-sol-zenodo-staging \
        --manifest validation/paper_numbers_manifest.yaml

Manifest schema (YAML list, one entry per claim):
    - tag: "table-1: 5-fold mean (license-clean release ensemble)"
      file: results/cluster_disjoint_5fold/aikisol_release_ensemble.json
      field: cv.mean_prob_auc.mean
      expected: 0.847
      tolerance: 0.001        # +/- absolute; default 0.001
      manuscript_section: "Table 1, row 'AikiSol (released)'"
      sub_field_check:        # optional further required keys
        - cv.mean_prob_auc.std
        - n_folds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")


def get_dotted(d: Any, path: str):
    cur = d
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                raise KeyError(f"List index {part} not valid in {path}")
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"Missing key {part!r} in {path}")
            cur = cur[part]
        else:
            raise KeyError(f"Cannot descend into {path} at {part}")
    return cur


def check_entry(entry: dict, deposit_root: Path) -> tuple[bool, str]:
    file = entry.get("file")
    field = entry.get("field")
    expected = entry.get("expected")
    tol = float(entry.get("tolerance", 0.001))
    sub_field_checks = entry.get("sub_field_check", [])
    if file is None or field is None:
        return False, f"manifest entry missing file/field: {entry}"
    p = (deposit_root / file).resolve()
    if not p.exists():
        return False, f"missing artefact: {p}"

    with p.open() as f:
        data = json.load(f)

    try:
        actual = get_dotted(data, field)
    except KeyError as e:
        return False, f"{file}: {e}"

    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)):
            return False, f"{file}: {field} expected number, got {type(actual).__name__}={actual!r}"
        if abs(actual - expected) > tol:
            return False, (f"{file}: {field} = {actual} differs from expected "
                            f"{expected} (tol +/- {tol})")
    else:
        if actual != expected:
            return False, f"{file}: {field} = {actual!r} != expected {expected!r}"

    for sub in sub_field_checks:
        try:
            get_dotted(data, sub)
        except KeyError as e:
            return False, f"{file}: required sub-field {sub} missing -- {e}"

    return True, f"OK  {entry.get('tag', '')}  ({field}={actual})"


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deposit-root", required=True,
                    help="Path to the staged Zenodo deposit root.")
    ap.add_argument("--manifest", default=str(here / "paper_numbers_manifest.yaml"))
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"manifest {manifest_path} not found; create one first.")
    deposit_root = Path(args.deposit_root).resolve()
    if not deposit_root.exists():
        sys.exit(f"deposit root {deposit_root} not found.")

    with manifest_path.open() as f:
        entries = yaml.safe_load(f) or []
    if not entries:
        sys.exit(f"manifest {manifest_path} is empty.")

    print(f"[reproduce] {len(entries)} manuscript claim(s) to verify "
          f"against {deposit_root}.\n")
    n_ok = n_fail = 0
    for entry in entries:
        ok, msg = check_entry(entry, deposit_root)
        if ok:
            n_ok += 1
            print(f"  [PASS] {msg}")
        else:
            n_fail += 1
            print(f"  [FAIL] {entry.get('tag', '?')}: {msg}")

    print(f"\n{n_ok} passed, {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
