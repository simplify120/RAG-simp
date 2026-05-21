"""
Aggregate evaluation metrics for model comparison.
Provides multiple levels of aggregation:
1. Overall by model (across all prompts)
2. By model + strategy (per prompt strategy)
3. Detailed per-item data

Use --description to filter to a specific run (e.g. "step 2 - RAG top k=3").
When omitted, aggregates all results; description is included in grouping so
step 1 and step 2 are kept separate.
"""

import argparse
import sys
from pathlib import Path
from sqlalchemy import case, func
import pandas as pd
import numpy as np

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.db.session import SessionLocal
from app.models.evaluation import Evaluation
from app.models.prompt import PromptResult, PromptVersion, Prompt


def aggregate_overall_by_model(description=None):
    """
    Aggregate metrics overall by model (ignoring prompt strategy).
    Returns DataFrame with mean, std, min, max, count for each metric.
    """
    db = SessionLocal()
    
    try:
        query = db.query(
            PromptResult.model_name,
            PromptVersion.version,
            PromptResult.description,
            # Quality metrics (higher is better)
            func.avg(Evaluation.bertscore_f1).label('avg_bertscore'),
            func.stddev(Evaluation.bertscore_f1).label('std_bertscore'),
            func.min(Evaluation.bertscore_f1).label('min_bertscore'),
            func.max(Evaluation.bertscore_f1).label('max_bertscore'),
            
            func.avg(Evaluation.bleu).label('avg_bleu'),
            func.stddev(Evaluation.bleu).label('std_bleu'),
            func.min(Evaluation.bleu).label('min_bleu'),
            func.max(Evaluation.bleu).label('max_bleu'),
            
            func.avg(Evaluation.sari).label('avg_sari'),
            func.stddev(Evaluation.sari).label('std_sari'),
            func.min(Evaluation.sari).label('min_sari'),
            func.max(Evaluation.sari).label('max_sari'),
            
            func.avg(Evaluation.perplexity).label('avg_perplexity'),
            func.stddev(Evaluation.perplexity).label('std_perplexity'),
            func.min(Evaluation.perplexity).label('min_perplexity'),
            func.max(Evaluation.perplexity).label('max_perplexity'),
            
            # Readability deltas (FKGL: negative is better, FRE: positive is better)
            func.avg(Evaluation.delta_fkgl).label('avg_delta_fkgl'),
            func.stddev(Evaluation.delta_fkgl).label('std_delta_fkgl'),
            func.min(Evaluation.delta_fkgl).label('min_delta_fkgl'),
            func.max(Evaluation.delta_fkgl).label('max_delta_fkgl'),
            
            func.avg(Evaluation.fre_delta).label('avg_fre_delta'),
            func.stddev(Evaluation.fre_delta).label('std_fre_delta'),
            func.min(Evaluation.fre_delta).label('min_fre_delta'),
            func.max(Evaluation.fre_delta).label('max_fre_delta'),
            
            # Output readability levels
            func.avg(Evaluation.fkgl_output).label('avg_fkgl_output'),
            func.avg(Evaluation.fre_output).label('avg_fre_output'),
            
            # Count
            func.count(Evaluation.evaluation_id).label('count')
        ).join(
            PromptResult, Evaluation.result_id == PromptResult.result_id
        ).join(
            PromptVersion, PromptResult.prompt_version_id == PromptVersion.prompt_version_id
        ).filter(
            PromptResult.model_name.isnot(None),
            Evaluation.bertscore_f1.isnot(None)
        )
        if description is not None:
            query = query.filter(PromptResult.description == description)
        results = query.group_by(
            PromptResult.model_name,
            PromptVersion.version,
            PromptResult.description
        ).order_by(
            PromptResult.model_name,
            PromptVersion.version,
            PromptResult.description
        ).all()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'model': r.model_name,
            'version': r.version,
            'description': r.description,
            'count': r.count,
            # BERTScore
            'bertscore_mean': float(r.avg_bertscore) if r.avg_bertscore else None,
            'bertscore_std': float(r.std_bertscore) if r.std_bertscore else None,
            'bertscore_min': float(r.min_bertscore) if r.min_bertscore else None,
            'bertscore_max': float(r.max_bertscore) if r.max_bertscore else None,
            # BLEU
            'bleu_mean': float(r.avg_bleu) if r.avg_bleu else None,
            'bleu_std': float(r.std_bleu) if r.std_bleu else None,
            'bleu_min': float(r.min_bleu) if r.min_bleu else None,
            'bleu_max': float(r.max_bleu) if r.max_bleu else None,
            # SARI
            'sari_mean': float(r.avg_sari) if r.avg_sari else None,
            'sari_std': float(r.std_sari) if r.std_sari else None,
            'sari_min': float(r.min_sari) if r.min_sari else None,
            'sari_max': float(r.max_sari) if r.max_sari else None,
            # Perplexity
            'perplexity_mean': float(r.avg_perplexity) if r.avg_perplexity else None,
            'perplexity_std': float(r.std_perplexity) if r.std_perplexity else None,
            'perplexity_min': float(r.min_perplexity) if r.min_perplexity else None,
            'perplexity_max': float(r.max_perplexity) if r.max_perplexity else None,
            # FKGL Delta
            'fkgl_delta_mean': float(r.avg_delta_fkgl) if r.avg_delta_fkgl else None,
            'fkgl_delta_std': float(r.std_delta_fkgl) if r.std_delta_fkgl else None,
            'fkgl_delta_min': float(r.min_delta_fkgl) if r.min_delta_fkgl else None,
            'fkgl_delta_max': float(r.max_delta_fkgl) if r.max_delta_fkgl else None,
            # FRE Delta
            'fre_delta_mean': float(r.avg_fre_delta) if r.avg_fre_delta else None,
            'fre_delta_std': float(r.std_fre_delta) if r.std_fre_delta else None,
            'fre_delta_min': float(r.min_fre_delta) if r.min_fre_delta else None,
            'fre_delta_max': float(r.max_fre_delta) if r.max_fre_delta else None,
            # Output readability
            'fkgl_output_mean': float(r.avg_fkgl_output) if r.avg_fkgl_output else None,
            'fre_output_mean': float(r.avg_fre_output) if r.avg_fre_output else None,
        } for r in results])
        
        return df
        
    finally:
        db.close()


def aggregate_by_model_and_strategy(description=None):
    """
    Aggregate metrics by model and prompt strategy.
    Returns DataFrame with metrics grouped by model + strategy_type.

    Rows are anchored on ``prompt_results`` (left join to ``evaluation``) so
    ``count`` matches the number of model outputs for that bucket (e.g. 189).
    Means/stddevs use SQL aggregates that ignore NULLs per metric; if some
    items lack an evaluation or a given metric, use ``count_with_bertscore``
    to see how many rows had BERTScore for overlap with those aggregates.
    """
    db = SessionLocal()
    
    try:
        query = db.query(
            PromptResult.model_name,
            PromptVersion.version,
            PromptResult.description,
            Prompt.strategy_type,
            # Quality metrics
            func.avg(Evaluation.bertscore_f1).label('avg_bertscore'),
            func.stddev(Evaluation.bertscore_f1).label('std_bertscore'),
            func.avg(Evaluation.bleu).label('avg_bleu'),
            func.stddev(Evaluation.bleu).label('std_bleu'),
            func.avg(Evaluation.sari).label('avg_sari'),
            func.stddev(Evaluation.sari).label('std_sari'),
            func.avg(Evaluation.perplexity).label('avg_perplexity'),
            func.stddev(Evaluation.perplexity).label('std_perplexity'),
            func.avg(Evaluation.lens).label('avg_lens'),
            func.stddev(Evaluation.lens).label('std_lens'),
            # Readability deltas
            func.avg(Evaluation.delta_fkgl).label('avg_delta_fkgl'),
            func.stddev(Evaluation.delta_fkgl).label('std_delta_fkgl'),
            func.avg(Evaluation.fre_delta).label('avg_fre_delta'),
            func.stddev(Evaluation.fre_delta).label('std_fre_delta'),
            # Output readability
            func.avg(Evaluation.fkgl_output).label('avg_fkgl_output'),
            func.avg(Evaluation.fre_output).label('avg_fre_output'),
            # Counts: total outputs vs rows with BERTScore (legacy queries filtered the latter only)
            func.count(PromptResult.result_id).label('count'),
            func.sum(
                case((Evaluation.bertscore_f1.isnot(None), 1), else_=0)
            ).label('count_with_bertscore'),
        ).select_from(PromptResult).join(
            PromptVersion,
            PromptResult.prompt_version_id == PromptVersion.prompt_version_id,
        ).join(
            Prompt,
            PromptVersion.prompt_id == Prompt.prompt_id,
        ).outerjoin(
            Evaluation,
            PromptResult.result_id == Evaluation.result_id,
        ).filter(
            PromptResult.model_name.isnot(None),
        )
        if description is not None:
            query = query.filter(PromptResult.description == description)
        results = query.group_by(
            PromptResult.model_name,
            PromptVersion.version,
            PromptResult.description,
            Prompt.strategy_type
        ).order_by(
            PromptResult.model_name,
            PromptVersion.version,
            PromptResult.description,
            Prompt.strategy_type
        ).all()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'model': r.model_name,
            'version': r.version,
            'description': r.description,
            'strategy': r.strategy_type,
            'count': int(r.count) if r.count is not None else 0,
            'count_with_bertscore': int(r.count_with_bertscore)
            if r.count_with_bertscore is not None
            else 0,
            # BERTScore
            'bertscore_mean': float(r.avg_bertscore) if r.avg_bertscore else None,
            'bertscore_std': float(r.std_bertscore) if r.std_bertscore else None,
            # BLEU
            'bleu_mean': float(r.avg_bleu) if r.avg_bleu else None,
            'bleu_std': float(r.std_bleu) if r.std_bleu else None,
            # SARI
            'sari_mean': float(r.avg_sari) if r.avg_sari else None,
            'sari_std': float(r.std_sari) if r.std_sari else None,
            # Perplexity
            'perplexity_mean': float(r.avg_perplexity) if r.avg_perplexity else None,
            'perplexity_std': float(r.std_perplexity) if r.std_perplexity else None,
            # LENS
            'lens_mean': float(r.avg_lens) if r.avg_lens else None,
            'lens_std': float(r.std_lens) if r.std_lens else None,
            # FKGL Delta
            'fkgl_delta_mean': float(r.avg_delta_fkgl) if r.avg_delta_fkgl else None,
            'fkgl_delta_std': float(r.std_delta_fkgl) if r.std_delta_fkgl else None,
            # FRE Delta
            'fre_delta_mean': float(r.avg_fre_delta) if r.avg_fre_delta else None,
            'fre_delta_std': float(r.std_fre_delta) if r.std_fre_delta else None,
            # Output readability
            'fkgl_output_mean': float(r.avg_fkgl_output) if r.avg_fkgl_output else None,
            'fre_output_mean': float(r.avg_fre_output) if r.avg_fre_output else None,
        } for r in results])
        
        return df
        
    finally:
        db.close()


def get_detailed_results(description=None):
    """
    Get detailed per-item results for deep analysis.
    Returns DataFrame with all metrics for each individual result.
    """
    db = SessionLocal()
    
    try:
        query = db.query(
            PromptResult.model_name,
            PromptVersion.version,
            PromptResult.description,
            Prompt.strategy_type,
            PromptResult.item_id,
            Evaluation.result_id,
            # Quality metrics
            Evaluation.bertscore_f1,
            Evaluation.bleu,
            Evaluation.sari,
            Evaluation.perplexity,
            # Readability deltas
            Evaluation.delta_fkgl,
            Evaluation.fre_delta,
            # Input/output readability
            Evaluation.fkgl_input,
            Evaluation.fkgl_output,
            Evaluation.fre_input,
            Evaluation.fre_output,
        ).join(
            PromptResult, Evaluation.result_id == PromptResult.result_id
        ).join(
            PromptVersion, Evaluation.prompt_version_id == PromptVersion.prompt_version_id
        ).join(
            Prompt, PromptVersion.prompt_id == Prompt.prompt_id
        ).filter(
            PromptResult.model_name.isnot(None),
            Evaluation.bertscore_f1.isnot(None)
        )
        if description is not None:
            query = query.filter(PromptResult.description == description)
        results = query.order_by(
            PromptResult.model_name,
            PromptVersion.version,
            Prompt.strategy_type
        ).all()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'model': r.model_name,
            'version': r.version,
            'description': r.description,
            'strategy': r.strategy_type,
            'item_id': str(r.item_id),
            'result_id': str(r.result_id),
            # Quality metrics
            'bertscore': float(r.bertscore_f1) if r.bertscore_f1 else None,
            'bleu': float(r.bleu) if r.bleu else None,
            'sari': float(r.sari) if r.sari else None,
            'perplexity': float(r.perplexity) if r.perplexity else None,
            # Readability deltas
            'fkgl_delta': float(r.delta_fkgl) if r.delta_fkgl else None,
            'fre_delta': float(r.fre_delta) if r.fre_delta else None,
            # Input/output readability
            'fkgl_input': float(r.fkgl_input) if r.fkgl_input else None,
            'fkgl_output': float(r.fkgl_output) if r.fkgl_output else None,
            'fre_input': float(r.fre_input) if r.fre_input else None,
            'fre_output': float(r.fre_output) if r.fre_output else None,
        } for r in results])
        
        return df
        
    finally:
        db.close()


def aggregate_by_prompt_version(description=None):
    """
    Aggregate metrics by prompt_version_id and model_name.
    Returns a dictionary where keys are prompt_version_id (as string) and values are DataFrames
    with models as rows and metrics as columns.
    Each DataFrame represents one table for a specific prompt_version_id.
    """
    db = SessionLocal()
    
    try:
        query = db.query(
            Evaluation.prompt_version_id,
            PromptResult.model_name,
            PromptResult.description,
            Prompt.strategy_type,
            PromptVersion.version,
            # Quality metrics
            func.avg(Evaluation.bertscore_f1).label('bertscore_mean'),
            func.stddev(Evaluation.bertscore_f1).label('bertscore_std'),
            func.avg(Evaluation.bleu).label('bleu_mean'),
            func.stddev(Evaluation.bleu).label('bleu_std'),
            func.avg(Evaluation.sari).label('sari_mean'),
            func.stddev(Evaluation.sari).label('sari_std'),
            func.avg(Evaluation.perplexity).label('perplexity_mean'),
            func.stddev(Evaluation.perplexity).label('perplexity_std'),
            # Readability deltas
            func.avg(Evaluation.delta_fkgl).label('delta_fkgl_mean'),
            func.stddev(Evaluation.delta_fkgl).label('delta_fkgl_std'),
            func.avg(Evaluation.fre_delta).label('fre_delta_mean'),
            func.stddev(Evaluation.fre_delta).label('fre_delta_std'),
            # Output readability
            func.avg(Evaluation.fkgl_output).label('fkgl_output_mean'),
            func.stddev(Evaluation.fkgl_output).label('fkgl_output_std'),
            func.avg(Evaluation.fre_output).label('fre_output_mean'),
            func.stddev(Evaluation.fre_output).label('fre_output_std'),
            # Additional metrics
            func.avg(Evaluation.entity_additions_rate).label('entity_additions_rate_mean'),
            func.avg(Evaluation.number_mismatch_rate).label('number_mismatch_rate_mean'),
            # LENS
            func.avg(Evaluation.lens).label('lens_mean'),
            func.stddev(Evaluation.lens).label('lens_std'),
            # Count
            func.count(Evaluation.evaluation_id).label('count')
        ).join(
            PromptResult, Evaluation.result_id == PromptResult.result_id
        ).join(
            PromptVersion, Evaluation.prompt_version_id == PromptVersion.prompt_version_id
        ).join(
            Prompt, PromptVersion.prompt_id == Prompt.prompt_id
        ).filter(
            PromptResult.model_name.isnot(None),
            Evaluation.bertscore_f1.isnot(None)
        )
        if description is not None:
            query = query.filter(PromptResult.description == description)
        results = query.group_by(
            Evaluation.prompt_version_id,
            PromptResult.model_name,
            PromptResult.description,
            Prompt.strategy_type,
            PromptVersion.version
        ).all()
        
        # Group by prompt_version_id
        tables_by_prompt_version = {}
        
        for r in results:
            prompt_version_id_str = str(r.prompt_version_id)
            
            if prompt_version_id_str not in tables_by_prompt_version:
                tables_by_prompt_version[prompt_version_id_str] = {
                    'strategy_type': r.strategy_type,
                    'version': r.version,
                    'description': r.description,
                    'data': []
                }
            
            tables_by_prompt_version[prompt_version_id_str]['data'].append({
                'model': r.model_name,
                'description': r.description,
                'count': r.count,
                # Quality metrics
                'BERTScore': float(r.bertscore_mean) if r.bertscore_mean else None,
                'BERTScore_std': float(r.bertscore_std) if r.bertscore_std else None,
                'BLEU': float(r.bleu_mean) if r.bleu_mean else None,
                'BLEU_std': float(r.bleu_std) if r.bleu_std else None,
                'SARI': float(r.sari_mean) if r.sari_mean else None,
                'SARI_std': float(r.sari_std) if r.sari_std else None,
                'Perplexity': float(r.perplexity_mean) if r.perplexity_mean else None,
                'Perplexity_std': float(r.perplexity_std) if r.perplexity_std else None,
                # Readability deltas
                'FKGL_Delta': float(r.delta_fkgl_mean) if r.delta_fkgl_mean else None,
                'FKGL_Delta_std': float(r.delta_fkgl_std) if r.delta_fkgl_std else None,
                'FRE_Delta': float(r.fre_delta_mean) if r.fre_delta_mean else None,
                'FRE_Delta_std': float(r.fre_delta_std) if r.fre_delta_std else None,
                # Output readability
                'FKGL_Output': float(r.fkgl_output_mean) if r.fkgl_output_mean else None,
                'FKGL_Output_std': float(r.fkgl_output_std) if r.fkgl_output_std else None,
                'FRE_Output': float(r.fre_output_mean) if r.fre_output_mean else None,
                'FRE_Output_std': float(r.fre_output_std) if r.fre_output_std else None,
                # Additional metrics
                'Entity_Additions_Rate': float(r.entity_additions_rate_mean) if r.entity_additions_rate_mean else None,
                'Number_Mismatch_Rate': float(r.number_mismatch_rate_mean) if r.number_mismatch_rate_mean else None,
                # LENS
                'LENS': float(r.lens_mean) if r.lens_mean else None,
                'LENS_std': float(r.lens_std) if r.lens_std else None,
            })
        
        # Convert each group to a DataFrame
        result_dict = {}
        for prompt_version_id, info in tables_by_prompt_version.items():
            df = pd.DataFrame(info['data'])
            # Set model as index for cleaner table display
            df.set_index('model', inplace=True)
            result_dict[prompt_version_id] = {
                'dataframe': df,
                'strategy_type': info['strategy_type'],
                'version': info['version'],
                'description': info.get('description')
            }
        
        return result_dict
        
    finally:
        db.close()


def print_summary_table(df_overall):
    """Print a formatted summary table of overall metrics."""
    print("\n" + "="*100)
    print("OVERALL MODEL COMPARISON SUMMARY")
    print("="*100)
    
    for _, row in df_overall.iterrows():
        desc = f" [{row['description']}]" if row.get('description') else ""
        print(f"\n{row['model'].upper()} ({row['version']}){desc}:")
        print(f"  Samples: {row['count']}")
        print(f"\n  Quality Metrics (higher is better):")
        print(f"    BERTScore: {row['bertscore_mean']:.4f} ± {row['bertscore_std']:.4f} "
              f"[{row['bertscore_min']:.4f}, {row['bertscore_max']:.4f}]")
        print(f"    BLEU:      {row['bleu_mean']:.4f} ± {row['bleu_std']:.4f} "
              f"[{row['bleu_min']:.4f}, {row['bleu_max']:.4f}]")
        print(f"    SARI:      {row['sari_mean']:.4f} ± {row['sari_std']:.4f} "
              f"[{row['sari_min']:.4f}, {row['sari_max']:.4f}]")
        print(f"    Perplexity: {row['perplexity_mean']:.4f} ± {row['perplexity_std']:.4f} "
              f"[{row['perplexity_min']:.4f}, {row['perplexity_max']:.4f}] (lower is better)")
        print(f"\n  Readability Improvement:")
        print(f"    FKGL Δ:    {row['fkgl_delta_mean']:.2f} ± {row['fkgl_delta_std']:.2f} "
              f"(negative = simpler, better)")
        print(f"    FRE Δ:     {row['fre_delta_mean']:.2f} ± {row['fre_delta_std']:.2f} "
              f"(positive = easier, better)")
        print(f"\n  Output Readability:")
        print(f"    FKGL:      {row['fkgl_output_mean']:.2f} (lower = simpler)")
        print(f"    FRE:       {row['fre_output_mean']:.2f} (higher = easier)")


def print_strategy_table(df_by_strategy):
    """Print a formatted breakdown of metrics by model, version, and strategy."""
    print("\n" + "="*100)
    print("BREAKDOWN BY MODEL × VERSION × STRATEGY (40 samples each)")
    print("="*100)

    current_group = None
    for _, row in df_by_strategy.iterrows():
        desc = f" [{row.get('description')}]" if row.get('description') else ""
        group = f"{row['model'].upper()} ({row['version']}){desc}"
        if group != current_group:
            print(f"\n{'─'*100}")
            print(f"  {group}")
            print(f"{'─'*100}")
            print(f"  {'Strategy':<14} {'Count':>6}  {'BERTScore':>10}  {'BLEU':>8}  {'SARI':>8}  {'FKGL Δ':>8}  {'FRE Δ':>8}")
            current_group = group

        fkgl = f"{row['fkgl_delta_mean']:.2f}" if row['fkgl_delta_mean'] == row['fkgl_delta_mean'] else "  nan"
        fre  = f"{row['fre_delta_mean']:.2f}"  if row['fre_delta_mean']  == row['fre_delta_mean']  else "  nan"
        print(
            f"  {row['strategy']:<14} {row['count']:>6}  "
            f"{row['bertscore_mean']:>10.4f}  "
            f"{row['bleu_mean']:>8.4f}  "
            f"{row['sari_mean']:>8.4f}  "
            f"{fkgl:>8}  "
            f"{fre:>8}"
        )


def export_to_csv(df, filename, output_dir=None):
    """Export DataFrame to CSV file."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs" / "csv"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    print(f"✓ Exported to {filepath}")
    return filepath


def print_tables_by_prompt_version(tables_dict):
    """
    Print tables grouped by prompt_version_id.
    
    Args:
        tables_dict: Dictionary returned by aggregate_by_prompt_version()
    """
    print("\n" + "="*120)
    print("EVALUATION RESULTS BY PROMPT VERSION")
    print("="*120)
    
    for prompt_version_id, info in tables_dict.items():
        df = info['dataframe']
        strategy = info['strategy_type']
        version = info['version']
        desc = info.get('description')
        
        print(f"\n{'='*120}")
        print(f"Prompt Version ID: {prompt_version_id}")
        print(f"Strategy Type: {strategy}")
        print(f"Version: {version}")
        if desc:
            print(f"Description: {desc}")
        print(f"{'='*120}\n")
        
        # Display the table
        print(df.to_string())
        print(f"\nTotal models: {len(df)}")
        print(f"{'='*120}\n")


def _description_to_slug(description):
    """Convert description to filename-safe slug."""
    if description == "step 1 - simple prompt engineering":
        return "step1"
    if description == "step 2 - RAG top k=3":
        return "step2"
    if description:
        return description.replace(" ", "_").replace("-", "_").lower()[:30]
    return "all"


def export_tables_by_prompt_version(tables_dict, output_dir=None):
    """
    Export tables grouped by prompt_version_id to CSV files.
    Filename format: {step_slug}_{strategy}_v{version}.csv (e.g. step1_zeroshot_v1.csv)
    
    Args:
        tables_dict: Dictionary returned by aggregate_by_prompt_version()
        output_dir: Directory to save CSV files (default: outputs/csv/prompt_versions/)
    
    Returns:
        List of exported file paths
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs" / "csv" / "prompt_versions"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = []
    
    for prompt_version_id, info in tables_dict.items():
        df = info['dataframe']
        strategy = info['strategy_type']
        version = info['version']
        description = info.get('description')
        slug = _description_to_slug(description)
        
        # Create filename: step1_zeroshot_v1.csv or step2_zeroshot_v2.csv
        safe_strategy = strategy.replace(" ", "_").lower()
        safe_version = version.replace(" ", "_").lower() if version else "unknown"
        filename = f"{slug}_{safe_strategy}_v{safe_version}.csv"
        
        filepath = output_dir / filename
        df.to_csv(filepath, index=True)  # index=True to include model names
        exported_files.append(filepath)
        print(f"✓ Exported table for prompt_version_id {prompt_version_id[:8]}... to {filepath}")
    
    return exported_files


def generate_results_by_prompt_version(description=None, print_tables=True, export_csv=True):
    """
    Generate and optionally display/export results tables grouped by prompt_version_id.
    
    Args:
        print_tables: If True, print tables to console
        export_csv: If True, export tables to CSV files
    
    Returns:
        Dictionary of tables by prompt_version_id
    """
    print("\nGenerating results tables by prompt_version_id...")
    tables_dict = aggregate_by_prompt_version(description=description)
    
    if not tables_dict:
        print("⚠ No data found for prompt version aggregation")
        return {}
    
    print(f"✓ Found {len(tables_dict)} prompt version(s)")
    
    if print_tables:
        print_tables_by_prompt_version(tables_dict)
    
    if export_csv:
        exported_files = export_tables_by_prompt_version(tables_dict)
        print(f"\n✓ Exported {len(exported_files)} table(s) to CSV")
    
    return tables_dict


def main():
    """Main function to run all aggregations and exports."""
    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics")
    parser.add_argument("--description", type=str, default=None, help="Only aggregate results with this description")
    args = parser.parse_args()
    
    print("Starting metric aggregation...")
    if args.description:
        print(f"Filtering by description: {args.description}")
    
    # 1. Overall aggregation by model
    print("\n1. Aggregating overall metrics by model...")
    df_overall = aggregate_overall_by_model(description=args.description)
    if not df_overall.empty:
        print_summary_table(df_overall)
        export_to_csv(df_overall, "model_comparison_overall.csv")
    else:
        print("⚠ No data found for overall aggregation")
    
    # 2. Aggregation by model + version + strategy
    print("\n2. Aggregating metrics by model, version, and strategy...")
    df_by_strategy = aggregate_by_model_and_strategy(description=args.description)
    if not df_by_strategy.empty:
        print_strategy_table(df_by_strategy)
        export_to_csv(df_by_strategy, "model_comparison_by_strategy.csv")
    else:
        print("⚠ No data found for strategy-based aggregation")
    
    # 3. Detailed per-item results
    print("\n3. Extracting detailed per-item results...")
    df_detailed = get_detailed_results(description=args.description)
    if not df_detailed.empty:
        print(f"✓ Found {len(df_detailed)} individual results")
        export_to_csv(df_detailed, "model_comparison_detailed.csv")
    else:
        print("⚠ No detailed data found")
    
    print("\n" + "="*100)
    print("✓ Aggregation complete!")
    print("="*100)


if __name__ == "__main__":
    main()

