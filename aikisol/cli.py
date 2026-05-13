"""User-facing CLI for Aiki-Sol scoring.

Inputs (mutually exclusive):
  --sequence "MAEILVTQ..."     : single sequence
  --fasta /path/to/file.fasta  : multiple sequences
  --csv   /path/to/file.csv    : per-row scoring (column-name configurable)

Outputs (default: stdout JSON; --out CSV):
  per-protein mean_prob (across the 5 per-stringency outputs),
  prob_marginal (the dedicated stringency-marginal output), plus
  per-stringency probs.

Examples:
  aikisol-predict --sequence "MAEILVTQNMK..." --pretty
  aikisol-predict --fasta my.fasta --out predictions.csv
  aikisol-predict --csv input.csv --seq-col sequence --out predictions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

from aikisol import Aikisol, STRINGENCY_LABELS


def parse_fasta(path: Path) -> Iterator[tuple[str, str]]:
    label, buf = None, []
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if label is not None:
                    yield label, "".join(buf)
                label = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if label is not None:
            yield label, "".join(buf)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sequence", help="Single amino-acid sequence string")
    g.add_argument("--fasta", help="Path to a FASTA file")
    g.add_argument("--csv", help="Path to a CSV file with a sequence column")

    ap.add_argument("--seq-col", default="sequence",
                    help="Column name when --csv is used (default: sequence)")
    ap.add_argument("--id-col", default=None,
                    help="ID column when --csv is used (default: row index)")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: stdout)")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to a single .pt file. Default looks for "
                         "$AIKISOL_CKPT_DIR/aikisol_v2_canonical_147k_full.pt "
                         "(or ~/.cache/aikisol/checkpoints/...). Pass an "
                         "alternative path to score with the research-tier "
                         "n3v2 checkpoint instead (CC-BY-NC-ND).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-normalize", action="store_true",
                    help="Skip tag-aware normalization (NOT recommended; the model "
                         "was trained on tag-stripped sequences and tagged inputs "
                         "silently degrade AUC).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pretty", action="store_true",
                    help="Pretty-print the JSON for --sequence mode.")
    args = ap.parse_args(argv)

    ids: list[str] = []
    seqs: list[str] = []
    if args.sequence:
        ids, seqs = ["query"], [args.sequence.strip().upper()]
    elif args.fasta:
        for label, s in parse_fasta(Path(args.fasta)):
            ids.append(label)
            seqs.append(s)
    elif args.csv:
        df = pd.read_csv(args.csv)
        if args.seq_col not in df.columns:
            ap.error(f"--seq-col {args.seq_col!r} not in {args.csv}")
        seqs = df[args.seq_col].astype(str).str.strip().str.upper().tolist()
        if args.id_col and args.id_col in df.columns:
            ids = df[args.id_col].astype(str).tolist()
        else:
            ids = [str(i) for i in df.index]

    if not seqs:
        ap.error("no sequences found")

    print(f"[aikisol] scoring {len(seqs)} sequence(s) on {args.device}",
          file=sys.stderr)
    model = Aikisol.from_pretrained(checkpoint=args.checkpoint, device=args.device)

    out = model.predict(
        seqs,
        normalize=not args.no_normalize,
        batch_size=args.batch_size,
    )

    rows = []
    for i, sid in enumerate(ids):
        row = {
            "id": sid,
            "mean_prob": float(out.mean_prob[i]),
            "prob_marginal": float(out.prob_marginal[i]),
            "normalized_sequence": out.normalized_sequences[i],
        }
        for k, v in out.per_stringency.items():
            row[f"prob_{k}"] = float(v[i])
        rows.append(row)

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[aikisol] wrote {args.out}", file=sys.stderr)
    elif len(rows) == 1:
        print(json.dumps(rows[0], indent=2 if args.pretty else None))
    else:
        print(json.dumps(rows, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
