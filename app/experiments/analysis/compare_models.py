"""
Statistical comparison between models.
Performs t-tests, Mann-Whitney U tests, and calculates effect sizes.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.db.session import SessionLocal
from app.models.evaluation import Evaluation
from app.models.prompt import PromptResult, PromptVersion, Prompt


def get_detailed_results(description=None):
    """Get detailed per-item results for statistical testing."""
    db = SessionLocal()
    
    try:
        query = db.query(
            PromptResult.model_name,
            PromptVersion.version,
            PromptResult.description,
            Prompt.strategy_type,
            Evaluation.bertscore_f1,
            Evaluation.bleu,
            Evaluation.sari,
            Evaluation.perplexity,
            Evaluation.delta_fkgl,
            Evaluation.fre_delta,
            Evaluation.fkgl_output,
            Evaluation.fre_output,
        ).join(
            PromptResult, Evaluation.result_id == PromptResult.result_id
        ).join(
            PromptVersion, PromptResult.prompt_version_id == PromptVersion.prompt_version_id
        ).join(
            Prompt, PromptVersion.prompt_id == Prompt.prompt_id
        ).filter(
            PromptResult.model_name.isnot(None),
            Evaluation.bertscore_f1.isnot(None)
        )
        if description is not None:
            query = query.filter(PromptResult.description == description)
        results = query.all()
        
        df = pd.DataFrame([{
            'model': r.model_name,
            'version': r.version,
            'description': r.description,
            'model_version': f"{r.model_name} ({r.version})" + (f" [{r.description}]" if r.description else ""),
            'strategy': r.strategy_type,
            'bertscore': float(r.bertscore_f1) if r.bertscore_f1 else None,
            'bleu': float(r.bleu) if r.bleu else None,
            'sari': float(r.sari) if r.sari else None,
            'perplexity': float(r.perplexity) if r.perplexity else None,
            'fkgl_delta': float(r.delta_fkgl) if r.delta_fkgl else None,
            'fre_delta': float(r.fre_delta) if r.fre_delta else None,
            'fkgl_output': float(r.fkgl_output) if r.fkgl_output else None,
            'fre_output': float(r.fre_output) if r.fre_output else None,
        } for r in results])
        
        return df
        
    finally:
        db.close()


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compare_models_statistically(df, model1="gpt-4o-mini (v1)", model2="gemini-2.0-flash (v1)"):
    """
    Perform statistical comparisons between two model+version combinations.
    model1/model2 should be 'model_name (version)' strings.
    Returns DataFrame with test results.
    """
    # Filter data for both model+version combinations
    df1 = df[df['model_version'] == model1].copy()
    df2 = df[df['model_version'] == model2].copy()
    
    metrics = ['bertscore', 'bleu', 'sari', 'perplexity', 'fkgl_delta', 'fre_delta', 
               'fkgl_output', 'fre_output']
    
    results = []
    
    for metric in metrics:
        # Get data for both models
        data1 = df1[metric].dropna()
        data2 = df2[metric].dropna()
        
        if len(data1) == 0 or len(data2) == 0:
            continue
        
        # Descriptive statistics
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        
        # Test for normality (Shapiro-Wilk test on sample)
        # Use smaller sample if dataset is large
        sample_size = min(50, len(data1), len(data2))
        _, p_norm1 = stats.shapiro(data1.sample(min(sample_size, len(data1))))
        _, p_norm2 = stats.shapiro(data2.sample(min(sample_size, len(data2))))
        is_normal = p_norm1 > 0.05 and p_norm2 > 0.05
        
        # Choose appropriate test
        if is_normal:
            # Use t-test for normal distributions
            t_stat, p_value = stats.ttest_ind(data1, data2)
            test_name = "t-test"
        else:
            # Use Mann-Whitney U test for non-normal distributions
            u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            test_name = "Mann-Whitney U"
        
        # Calculate effect size (Cohen's d)
        effect_size = cohens_d(data1, data2)
        effect_interpretation = interpret_effect_size(effect_size)
        
        # Determine which model is better
        # For metrics where higher is better: bertscore, bleu, sari, fre_delta, fre_output
        # For metrics where lower is better: perplexity, fkgl_delta (more negative), fkgl_output
        if metric in ['bertscore', 'bleu', 'sari', 'fre_delta', 'fre_output']:
            better_model = model1 if mean1 > mean2 else model2
            difference = mean1 - mean2
        else:  # perplexity, fkgl_delta, fkgl_output (lower is better)
            better_model = model1 if mean1 < mean2 else model2
            difference = mean1 - mean2
        
        results.append({
            'metric': metric,
            f'{model1}_mean': mean1,
            f'{model1}_std': std1,
            f'{model2}_mean': mean2,
            f'{model2}_std': std2,
            'difference': difference,
            'better_model': better_model,
            'test': test_name,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': effect_size,
            'effect_interpretation': effect_interpretation,
            'n1': len(data1),
            'n2': len(data2),
        })
    
    return pd.DataFrame(results)


def compare_by_strategy(df, model1="gpt-4o-mini (v1)", model2="gemini-2.0-flash (v1)"):
    """Compare model+version pairs within each strategy."""
    strategies = df['strategy'].unique()
    all_results = []
    
    for strategy in strategies:
        df_strategy = df[df['strategy'] == strategy]
        comparison = compare_models_statistically(df_strategy, model1, model2)
        comparison['strategy'] = strategy
        all_results.append(comparison)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def print_comparison_summary(df_comparison):
    """Print formatted comparison results."""
    print("\n" + "="*100)
    print("STATISTICAL COMPARISON RESULTS")
    print("="*100)
    
    for _, row in df_comparison.iterrows():
        print(f"\n{row['metric'].upper()}:")
        print(f"  {row['better_model']} is better")
        print(f"  Difference: {row['difference']:.4f}")
        print(f"  Test: {row['test']}")
        print(f"  p-value: {row['p_value']:.6f} {'***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else '(not significant)'}")
        print(f"  Effect size (Cohen's d): {row['effect_size']:.3f} ({row['effect_interpretation']})")
        print(f"  Sample sizes: {int(row['n1'])} vs {int(row['n2'])}")


def export_comparison(df, filename, output_dir=None):
    """Export comparison results to CSV."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs" / "csv"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    print(f"✓ Exported to {filepath}")
    return filepath


def main():
    """Main function to run statistical comparisons."""
    parser = argparse.ArgumentParser(description="Statistical comparison between models")
    parser.add_argument("--description", type=str, default=None, help="Only compare results with this description")
    args = parser.parse_args()
    
    print("Starting statistical comparison...")
    if args.description:
        print(f"Filtering by description: {args.description}")
    
    # Get detailed data
    print("\n1. Loading detailed results from database...")
    df = get_detailed_results(description=args.description)
    print(f"✓ Loaded {len(df)} results")
    
    # Show available model+version combinations
    model_versions = sorted(df['model_version'].unique())
    print(f"\nAvailable model+version combinations:")
    for mv in model_versions:
        print(f"  - {mv} ({len(df[df['model_version'] == mv])} samples)")
    
    # Define comparison pairs
    # 1. Best model overall (gpt-4o-mini v2 vs gemini v1)
    # 2. v1 vs v2 for models that have both
    comparison_pairs = []
    
    # All v1 models: compare against gpt-4o-mini v1 as baseline
    baseline = "gpt-4o-mini (v1)"
    for mv in model_versions:
        if mv != baseline:
            comparison_pairs.append((baseline, mv))
    
    all_comparisons = []
    for model1, model2 in comparison_pairs:
        print(f"\n{'='*100}")
        print(f"COMPARING: {model1}  vs  {model2}")
        print(f"{'='*100}")
        df_comparison = compare_models_statistically(df, model1, model2)
        print_comparison_summary(df_comparison)
        df_comparison['pair'] = f"{model1} vs {model2}"
        all_comparisons.append(df_comparison)
    
    # Export all comparisons
    if all_comparisons:
        df_all = pd.concat(all_comparisons, ignore_index=True)
        export_comparison(df_all, "statistical_comparison_overall.csv")
    
    # Comparison by strategy (gpt-4o-mini v1 vs v2)
    if "gpt-4o-mini (v2)" in model_versions:
        print(f"\n{'='*100}")
        print("STRATEGY BREAKDOWN: gpt-4o-mini v1 vs v2")
        print(f"{'='*100}")
        df_strategy_comparison = compare_by_strategy(df, "gpt-4o-mini (v1)", "gpt-4o-mini (v2)")
        if not df_strategy_comparison.empty:
            export_comparison(df_strategy_comparison, "statistical_comparison_by_strategy.csv")
            print(f"✓ Compared across {df_strategy_comparison['strategy'].nunique()} strategies")
    
    if "llama3.2 (v2)" in model_versions:
        print(f"\n{'='*100}")
        print("STRATEGY BREAKDOWN: llama3.2 v1 vs v2")
        print(f"{'='*100}")
        df_llama_comparison = compare_by_strategy(df, "llama3.2 (v1)", "llama3.2 (v2)")
        if not df_llama_comparison.empty:
            export_comparison(df_llama_comparison, "statistical_comparison_llama_v1_v2_by_strategy.csv")
            print(f"✓ Compared across {df_llama_comparison['strategy'].nunique()} strategies")
    
    print("\n" + "="*100)
    print("✓ Statistical comparison complete!")
    print("="*100)
    print("\nNote: p < 0.05 indicates statistical significance")
    print("Effect size interpretation: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)")


if __name__ == "__main__":
    main()

