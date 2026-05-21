"""
Collect evaluation aggregates and per-item details from the database into CSVs.

Reuses app.experiments.analysis.aggregate_metrics and visualization phase logic.
Excludes sonar-pro from exports (use sonar only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root (…/LLM-simplification-PE)
_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.experiments.analysis.aggregate_metrics import (
    aggregate_by_model_and_strategy,
)
from app.experiments.visualization.data_loader import load_detailed_results
from app.experiments.results.phase_utils import add_phase_columns, filter_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Export aggregated and detailed results CSVs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregated/ (default: ../aggregated next to this script)",
    )
    args = parser.parse_args()

    base = args.output_dir or (Path(__file__).resolve().parent.parent / "aggregated")
    base.mkdir(parents=True, exist_ok=True)

    print("Loading aggregate by model and strategy (all descriptions)...")
    df_agg = aggregate_by_model_and_strategy(description=None)
    df_agg = filter_models(df_agg)
    df_agg = add_phase_columns(df_agg)
    agg_path = base / "results_all_experiments.csv"
    df_agg.to_csv(agg_path, index=False)
    print(f"Wrote {agg_path} ({len(df_agg)} rows)")

    print("Loading per-item detailed results (includes LENS)...")
    df_det = load_detailed_results(description=None)
    df_det = filter_models(df_det)
    df_det = add_phase_columns(df_det)
    det_path = base / "results_detailed.csv"
    df_det.to_csv(det_path, index=False)
    print(f"Wrote {det_path} ({len(df_det)} rows)")


if __name__ == "__main__":
    main()
