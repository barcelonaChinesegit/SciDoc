#!/usr/bin/env python3
"""Evaluate the five self-contained result components using sxz v4 rules."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
