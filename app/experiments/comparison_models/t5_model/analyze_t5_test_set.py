"""
Aggregate T5 evaluation metrics from t5_large_text_simplification_evaluation and export to CSV.

- test: 40-item BGE split (same as RAG / prompt engineering, random.seed(42)).
- all: every row in the evaluation table (e.g. 40 + 149 = 189).

Output format matches app/experiments/comparison_models/plan_simp/analyze_plan_simp_test_set.py.

Usage:
    python -m app.experiments.comparison_models.t5_model.analyze_t5_test_set
    python -m app.experiments.comparison_models.t5_model.analyze_t5_test_set --split all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from app.db.session import SessionLocal
from app.experiments.comparison_models.plan_simp.run_plan_simp_pipeline import (
    get_test_items_bge_split,
)
from app.models.t5_evaluation import T5LargeTextSimplificationEvaluation


def _aggregate_rows(rows: List[T5LargeTextSimplificationEvaluation], description: str) -> pd.DataFrame:
    def safe_values(field: str):
        vals = [getattr(r, field) for r in rows if getattr(r, field) is not None]
        return np.array(vals) if vals else np.array([np.nan])

    metrics = {
        "bertscore_f1": safe_values("bertscore_f1"),
        "bleu": safe_values("bleu"),
        "sari": safe_values("sari"),
        "perplexity": safe_values("perplexity"),
        "delta_fkgl": safe_values("delta_fkgl"),
        "fre_delta": safe_values("fre_delta"),
        "fkgl_output": safe_values("fkgl_output"),
        "fre_output": safe_values("fre_output"),
        "lens": safe_values("lens"),
    }

    def mean_std(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return np.nan, np.nan
        return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0

    row = {
        "model": "t5-large-text-simplification",
        "description": description,
        "count": len(rows),
        "BERTScore": mean_std(metrics["bertscore_f1"])[0],
        "BERTScore_std": mean_std(metrics["bertscore_f1"])[1],
        "BLEU": mean_std(metrics["bleu"])[0],
        "BLEU_std": mean_std(metrics["bleu"])[1],
        "SARI": mean_std(metrics["sari"])[0],
        "SARI_std": mean_std(metrics["sari"])[1],
        "Perplexity": mean_std(metrics["perplexity"])[0],
        "Perplexity_std": mean_std(metrics["perplexity"])[1],
        "FKGL_Delta": mean_std(metrics["delta_fkgl"])[0],
        "FKGL_Delta_std": mean_std(metrics["delta_fkgl"])[1],
        "FRE_Delta": mean_std(metrics["fre_delta"])[0],
        "FRE_Delta_std": mean_std(metrics["fre_delta"])[1],
        "FKGL_Output": mean_std(metrics["fkgl_output"])[0],
        "FKGL_Output_std": mean_std(metrics["fkgl_output"])[1],
        "FRE_Output": mean_std(metrics["fre_output"])[0],
        "FRE_Output_std": mean_std(metrics["fre_output"])[1],
        "Entity_Additions_Rate": "",
        "Number_Mismatch_Rate": "",
        "LENS": mean_std(metrics["lens"])[0],
        "LENS_std": mean_std(metrics["lens"])[1],
    }

    return pd.DataFrame([row])


def aggregate_t5_test_set_metrics() -> pd.DataFrame:
    db = SessionLocal()
    try:
        test_item_ids = {item[0] for item in get_test_items_bge_split(db)}
        if len(test_item_ids) != 40:
            raise ValueError(
                f"Expected 40 test items, got {len(test_item_ids)}. "
                "Ensure the dataset and get_test_items_bge_split() match."
            )

        rows = (
            db.query(T5LargeTextSimplificationEvaluation)
            .filter(T5LargeTextSimplificationEvaluation.item_id.in_(test_item_ids))
            .all()
        )

        if len(rows) < 40:
            print(
                f"Warning: Found {len(rows)} T5 results for test set (expected 40). "
                "Proceeding with available data."
            )

        return _aggregate_rows(rows, "T5 fine-tuned (test set)")
    finally:
        db.close()


def aggregate_t5_full_dataset_metrics() -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = db.query(T5LargeTextSimplificationEvaluation).all()
        if not rows:
            print("Warning: No T5 evaluation rows in database.")
        return _aggregate_rows(rows, "T5 fine-tuned (full dataset)")
    finally:
        db.close()


def aggregate_t5_metrics(split: Literal["test", "all"]) -> pd.DataFrame:
    if split == "test":
        return aggregate_t5_test_set_metrics()
    return aggregate_t5_full_dataset_metrics()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export aggregated T5 metrics to CSV")
    parser.add_argument(
        "--split",
        choices=("test", "all"),
        default="test",
        help="test: 40-item BGE sample; all: every evaluation row (e.g. 189)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: outputs/t5_test_set_analysis.csv or t5_full_dataset_analysis.csv)",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        output_path = args.output.resolve()
    elif args.split == "test":
        output_path = output_dir / "t5_test_set_analysis.csv"
    else:
        output_path = output_dir / "t5_full_dataset_analysis.csv"

    label = "40 test items" if args.split == "test" else "full evaluation table"
    print(f"Aggregating T5 metrics ({label})...")
    df = aggregate_t5_metrics(args.split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Exported to {output_path}")
    print(f"  Rows: {len(df)}, Count: {df['count'].iloc[0]}")


if __name__ == "__main__":
    main()
