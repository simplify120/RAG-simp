"""
Generate publication figures under app/experiments/results/figures/.

Implements a multi-folder layout (see app/experiments/results/publication_figures.py):
  01_phase_evolution/ … 06_rankings/

Prefers aggregated/results_*.csv from collect_results.py; falls back to DB queries.
Includes T5 and Plan-Simp baselines from comparison_models/*/outputs/*_full_dataset_analysis.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.experiments.analysis.aggregate_metrics import aggregate_by_model_and_strategy
from app.experiments.results.phase_utils import add_phase_columns, canonicalize_phase_column, filter_models
from app.experiments.results.publication_figures import generate_all
from app.experiments.visualization.data_loader import load_detailed_results


def _results_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def load_inputs(aggregated_dir: Path):
    agg_p = aggregated_dir / "results_all_experiments.csv"
    det_p = aggregated_dir / "results_detailed.csv"
    if agg_p.exists() and det_p.exists():
        import pandas as pd

        agg = canonicalize_phase_column(pd.read_csv(agg_p))
        det = canonicalize_phase_column(pd.read_csv(det_p))
        return agg, det
    agg = aggregate_by_model_and_strategy(None)
    agg = add_phase_columns(filter_models(agg))
    det = load_detailed_results(None)
    det = add_phase_columns(filter_models(det))
    return agg, det


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures under results/figures/")
    parser.add_argument(
        "--aggregated-dir",
        type=Path,
        default=None,
        help="Directory with results_all_experiments.csv and results_detailed.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: ../figures); subfolders 01_…06_ are created inside",
    )
    args = parser.parse_args()

    agg_dir = args.aggregated_dir or (_results_dir() / "aggregated")
    out_dir = args.output_dir or (_results_dir() / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    agg, det = load_inputs(agg_dir)
    generate_all(agg, det, output_root=out_dir)


if __name__ == "__main__":
    main()
