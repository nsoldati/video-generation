#!/usr/bin/env python3
"""Convenience entry point for benchmark generation."""

from pathlib import Path
import sys

# Allow ``./generate.py`` from a source checkout without installing this package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from video_bench.cli import main


if __name__ == "__main__":
    main()
