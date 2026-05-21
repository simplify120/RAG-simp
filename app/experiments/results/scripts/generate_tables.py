"""
Build paper-oriented summary tables from aggregated/results_all_experiments.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.experiments.results.phase_utils import canonicalize_phase_column

PE_PHASES = {"PE_v1", "PE_v2"}
RAG_PHASES = {
    "Open AI RAG v2",
    "Open AI RAG v3",
    "E5 RAG v3",
    "BGE RAG v3",
}


def _load_agg(aggregated_dir: Path) -> pd.DataFrame:
    path = aggregated_dir / "results_all_experiments.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run collect_results.py first.")
    return canonicalize_phase_column(pd.read_csv(path))


def table_01_pe_by_model_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Prompt-engineering only: model × strategy × prompt version (v1/v2)."""
    sub = df[df["phase"].isin(PE_PHASES)].copy()
    sub = sub.sort_values(["phase", "model", "strategy"])
    cols = [
        "phase",
        "version",
        "model",
        "strategy",
        "description",
        "count",
        "sari_mean",
        "sari_std",
        "bertscore_mean",
        "bertscore_std",
        "bleu_mean",
        "bleu_std",
    ]
    # LENS may be missing in aggregate_by_model_and_strategy — include if present
    for c in ("lens_mean", "lens_std"):
        if c in sub.columns:
            cols.append(c)
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].reset_index(drop=True)


def table_02_rag_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """All RAG experiment groups: long format for side-by-side comparison."""
    sub = df[df["phase"].isin(RAG_PHASES)].copy()
    sub["rag_setting"] = sub["phase"].map(
        {
            "Open AI RAG v2": "OpenAI-RAG-k3",
            "Open AI RAG v3": "OpenAI-RAG-v3-clean",
            "E5 RAG v3": "E5-RAG-full",
            "BGE RAG v3": "BGE-RAG-full",
        }
    )
    sub = sub.sort_values(["model", "rag_setting", "strategy"])
    base_cols = [
        "rag_setting",
        "phase",
        "experiment_group",
        "retrieval",
        "description",
        "model",
        "strategy",
        "count",
        "sari_mean",
        "sari_std",
        "bertscore_mean",
        "bertscore_std",
        "bleu_mean",
        "bleu_std",
        "perplexity_mean",
        "perplexity_std",
    ]
    out_cols = [c for c in base_cols if c in sub.columns]
    for c in ("lens_mean", "lens_std"):
        if c in sub.columns:
            out_cols.append(c)
    return sub[out_cols].reset_index(drop=True)


def table_03_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """One best row per model by highest sari_mean (ties: prefer lower perplexity if present)."""
    rows = []
    for model, g in df.groupby("model"):
        g2 = g[g["sari_mean"].notna()].copy()
        if g2.empty:
            continue
        g2 = g2.sort_values(
            ["sari_mean", "perplexity_mean"],
            ascending=[False, True],
            na_position="last",
        )
        best = g2.iloc[0].to_dict()
        best["rank_metric"] = "sari_mean"
        rows.append(best)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = [
        "model",
        "phase",
        "experiment_group",
        "strategy",
        "description",
        "retrieval",
        "count",
        "sari_mean",
        "sari_std",
        "bertscore_mean",
        "bertscore_std",
        "bleu_mean",
        "bleu_std",
        "lens_mean",
        "lens_std",
        "rank_metric",
    ]
    front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final_tables/*.csv from aggregated data.")
    parser.add_argument(
        "--aggregated-dir",
        type=Path,
        default=None,
        help="Directory containing results_all_experiments.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for final_tables/ (default: ../final_tables)",
    )
    args = parser.parse_args()

    agg_dir = args.aggregated_dir or (Path(__file__).resolve().parent.parent / "aggregated")
    out_dir = args.output_dir or (Path(__file__).resolve().parent.parent / "final_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_agg(agg_dir)

    t1 = table_01_pe_by_model_strategy(df)
    t1_path = out_dir / "table_01_pe_by_model_strategy.csv"
    t1.to_csv(t1_path, index=False)
    print(f"Wrote {t1_path} ({len(t1)} rows)")

    t2 = table_02_rag_comparison(df)
    t2_path = out_dir / "table_02_rag_comparison.csv"
    t2.to_csv(t2_path, index=False)
    print(f"Wrote {t2_path} ({len(t2)} rows)")

    t3 = table_03_best_configs(df)
    t3_path = out_dir / "table_03_best_configs.csv"
    t3.to_csv(t3_path, index=False)
    print(f"Wrote {t3_path} ({len(t3)} rows)")


if __name__ == "__main__":
    main()
