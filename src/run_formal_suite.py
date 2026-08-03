"""One command for frozen random-sample extraction followed by paper_v1 analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run a frozen formal paper_v1 suite")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    src = Path(__file__).resolve().parent
    suite = str(Path(args.suite).resolve())
    extraction = [sys.executable, str(src / "run_extraction_suite.py"), "--suite", suite]
    analysis = [sys.executable, str(src / "run_paper_suite.py"), "--suite", suite]
    if args.resume:
        extraction.append("--resume")
        analysis.append("--resume")
    subprocess.run(extraction, check=True)
    subprocess.run(analysis, check=True)
    print("Formal paper_v1 suite complete.")


if __name__ == "__main__":
    main()

