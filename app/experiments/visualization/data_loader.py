"""
data_loader.py — Shared data loading module for visualizations.
Pulls from PostgreSQL via SQLAlchemy and existing CSV outputs.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.db.session import SessionLocal
from app.models.evaluation import Evaluation
from app.models.prompt import PromptResult, PromptVersion, Prompt
from app.experiments.visualization.config import DESCRIPTION_TO_PHASE


def _assign_phase(row):
    """Assign phase label based on description and version."""
    desc = row.get("description")
    version = row.get("version")
    
    phase = DESCRIPTION_TO_PHASE.get(desc)
    if phase is not None:
        return phase
        
    # Handle Step 1 branching
    if desc == "step 1 - simple prompt engineering":
        if version == "v1": return "PE_v1"
        if version == "v2": return "PE_v2"
        
    return "UNKNOWN"


def load_detailed_results(description=None):
    """
    Load detailed per-item evaluation records from the database.
    Includes LENS, which was recently added to the Evaluation model.
    """
    db = SessionLocal()
    
    try:
        query = db.query(
            PromptResult.model_name,
            PromptVersion.version,
            PromptResult.description,
            Prompt.strategy_type,
            PromptResult.item_id,
            Evaluation.bertscore_f1,
            Evaluation.bleu,
            Evaluation.sari,
            Evaluation.perplexity,
            Evaluation.delta_fkgl,
            Evaluation.fre_delta,
            Evaluation.fkgl_input,
            Evaluation.fkgl_output,
            Evaluation.fre_input,
            Evaluation.fre_output,
            Evaluation.lens,
        ).join(
            PromptResult, Evaluation.result_id == PromptResult.result_id
        ).join(
            PromptVersion, PromptResult.prompt_version_id == PromptVersion.prompt_version_id
        ).join(
            Prompt, PromptVersion.prompt_id == Prompt.prompt_id
        ).filter(
            PromptResult.model_name.isnot(None),
            Evaluation.bertscore_f1.isnot(None) # ignore failed evals
        )
        
        if description is not None:
            query = query.filter(PromptResult.description == description)
            
        results = query.all()
        
        df = pd.DataFrame([{
            'model': r.model_name,
            'version': r.version,
            'description': r.description,
            'strategy': r.strategy_type,
            'item_id': str(r.item_id),
            'bertscore': float(r.bertscore_f1) if r.bertscore_f1 is not None else np.nan,
            'bleu': float(r.bleu) if r.bleu is not None else np.nan,
            'sari': float(r.sari) if r.sari is not None else np.nan,
            'perplexity': float(r.perplexity) if r.perplexity is not None else np.nan,
            'fkgl_delta': float(r.delta_fkgl) if r.delta_fkgl is not None else np.nan,
            'fre_delta': float(r.fre_delta) if r.fre_delta is not None else np.nan,
            'fkgl_input': float(r.fkgl_input) if r.fkgl_input is not None else np.nan,
            'fkgl_output': float(r.fkgl_output) if r.fkgl_output is not None else np.nan,
            'fre_input': float(r.fre_input) if r.fre_input is not None else np.nan,
            'fre_output': float(r.fre_output) if r.fre_output is not None else np.nan,
            'lens': float(r.lens) if r.lens is not None else np.nan,
        } for r in results])
        
        return df
        
    finally:
        db.close()


def load_t5_results():
    """Load T5 baseline aggregated results from CSV."""
    t5_csv_path = Path(__file__).parent.parent / "comparison_models" / "t5_model" / "outputs" / "t5_test_set_analysis.csv"
    
    if not t5_csv_path.exists():
        print(f"Warning: T5 baseline CSV not found at {t5_csv_path}")
        return pd.DataFrame()
        
    df = pd.read_csv(t5_csv_path)
    # The format is aggregated (mean/std). 
    # Return it directly for horizontal bar comparisons.
    return df


def load_t5_detailed_results():
    """Load T5 per-item results from the DB."""
    db = SessionLocal()
    from app.models.t5_evaluation import T5LargeTextSimplificationEvaluation
    
    try:
        results = db.query(T5LargeTextSimplificationEvaluation).all()
        
        df = pd.DataFrame([{
            'model': 't5-large-text-simplification',
            'version': 'v1',
            'description': 'T5 fine-tuned (test set)',
            'strategy': 'baseline',
            'item_id': str(r.item_id),
            'bertscore': float(r.bertscore_f1) if r.bertscore_f1 is not None else np.nan,
            'bleu': float(r.bleu) if r.bleu is not None else np.nan,
            'sari': float(r.sari) if r.sari is not None else np.nan,
            'perplexity': float(r.perplexity) if r.perplexity is not None else np.nan,
            'fkgl_delta': float(r.delta_fkgl) if r.delta_fkgl is not None else np.nan,
            'fre_delta': float(r.fre_delta) if r.fre_delta is not None else np.nan,
            'fkgl_input': float(r.fkgl_input) if r.fkgl_input is not None else np.nan,
            'fkgl_output': float(r.fkgl_output) if r.fkgl_output is not None else np.nan,
            'fre_input': float(r.fre_input) if r.fre_input is not None else np.nan,
            'fre_output': float(r.fre_output) if r.fre_output is not None else np.nan,
            'lens': float(r.lens) if r.lens is not None else np.nan,
        } for r in results])
        
        return df
    except Exception as e:
        print(f"Warning: Could not load T5 detailed results: {e}")
        return pd.DataFrame()
    finally:
        db.close()


def load_all_phases(include_t5=False):
    """
    Load all detailed results, compute phase labels.
    """
    df = load_detailed_results()
    df["phase"] = df.apply(_assign_phase, axis=1)
    
    # Filter out UNKNOWN phases
    df = df[df["phase"] != "UNKNOWN"].copy()
    
    if include_t5:
        t5_df = load_t5_detailed_results()
        if not t5_df.empty:
            t5_df["phase"] = "Baseline"
            df = pd.concat([df, t5_df], ignore_index=True)
            
    return df


def load_aggregated_by_phase_strategy():
    """
    Return mean for each metric grouped by (phase, strategy, model).
    """
    df = load_all_phases()
    
    # List of numeric metrics to aggregate
    from app.experiments.visualization.config import METRICS_ALL
    
    # Group by
    agg_funcs = {m: 'mean' for m in METRICS_ALL if m in df.columns}
    agg_funcs['item_id'] = 'count'  # to safely get count
    
    df_agg = df.groupby(["phase", "strategy", "model"]).agg(agg_funcs).reset_index()
    df_agg.rename(columns={'item_id': 'count'}, inplace=True)
    
    return df_agg
