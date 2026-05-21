"""
Regenerate aggregated CSVs, final tables, and figures.

Usage:
  python app/experiments/results/scripts/run_all.py
  python app/experiments/results/scripts/run_all.py --skip-collect
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full results pipeline")
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip DB export; use existing aggregated/*.csv",
    )
    args = parser.parse_args()

    py = sys.executable
    steps = []
    if not args.skip_collect:
        steps.append([py, str(_SCRIPTS / "collect_results.py")])
    steps.extend(
        [
            [py, str(_SCRIPTS / "generate_tables.py")],
            [py, str(_SCRIPTS / "generate_figures.py")],
        ]
    )

    for cmd in steps:
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
