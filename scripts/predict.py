#!/usr/bin/env python3
"""Thin wrapper for the aikisol CLI when running from a cloned repo.

The actual implementation lives in `aikisol.cli`; the installed entry point
`aikisol-predict` calls the same `main()` and accepts the same flags.
"""
from aikisol.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
