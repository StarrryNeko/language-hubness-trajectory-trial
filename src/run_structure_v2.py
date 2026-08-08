"""Single-model structure_v2 entry point."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    src = Path(__file__).resolve().parent
    for script in (src / "compute_structure_v2.py", src / "validate_structure_v2.py"):
        subprocess.run([sys.executable, str(script), "--config", args.config], check=True)


if __name__ == "__main__":
    main()
