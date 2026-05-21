"""
Generate visualizations for model comparison results.
Creates bar charts, box plots, scatter plots, and heatmaps.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.db.session import SessionLocal
from app.models.evaluation import Evaluation
from app.models.prompt import PromptResult, PromptVersion, Prompt

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def get_detailed_results():
    """Get detailed per-item results for visualization."""
    db = SessionLocal()
    
    try:
        results = db.query(
            PromptResult.model_name,
            PromptVersion.version,
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
        ).all()
        
        df = pd.DataFrame([{
            'model': r.model_name,
            'version': r.version,
            'model_version': f"{r.model_name} ({r.version})",
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


def save_figure(fig, filename, output_dir=None):
    """Save figure to file."""
    base_dir = Path(__file__).parent / "outputs" / "visualizations"
    if output_dir is None:
      output_dir = base_dir
    else:
      output_dir = base_dir / output_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved {filename}")


def plot_bar_charts_overall(df):
    """Create bar charts comparing overall metrics."""
    print("\nCreating bar charts for overall comparison...")
    
    # Aggregate by model_version
    df_agg = df.groupby('model_version').agg({
        'bertscore': 'mean',
        'bleu': 'mean',
        'sari': 'mean',
        'perplexity': 'mean',
        'fkgl_delta': 'mean',
        'fre_delta': 'mean',
    }).reset_index()
    
    # Quality metrics: higher is better (bertscore, bleu, sari); perplexity lower is better
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics_quality = ['bertscore', 'bleu', 'sari', 'perplexity']
    titles = [
        'BERTScore (Higher is Better)',
        'BLEU (Higher is Better)',
        'SARI (Higher is Better)',
        'Perplexity (Lower is Better)',
    ]
    for idx, (metric, title) in enumerate(zip(metrics_quality, titles)):
        ax = axes[idx]
        df_agg.plot(x='model_version', y=metric, kind='bar', ax=ax, 
                   legend=False)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score' if metric != 'perplexity' else 'Perplexity')
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'bar_charts_quality_metrics.png', 'bar_charts')
    
    # Readability deltas
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    n_models = len(df_agg)
    colors = plt.cm.Set3(np.linspace(0, 1, max(n_models, 1)))

    # FKGL Delta (more negative is better)
    ax = axes[0]
    df_agg.plot(x='model_version', y='fkgl_delta', kind='bar', ax=ax,
               color=colors[:n_models], legend=False)
    ax.set_title('FKGL Delta (More Negative = Simpler)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FKGL Delta')
    ax.set_xlabel('Model (Version)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # FRE Delta (more positive is better)
    ax = axes[1]
    df_agg.plot(x='model_version', y='fre_delta', kind='bar', ax=ax,
               color=colors[:n_models], legend=False)
    ax.set_title('FRE Delta (More Positive = Easier)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FRE Delta')
    ax.set_xlabel('Model (Version)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    save_figure(fig, 'bar_charts_readability_deltas.png', 'bar_charts')


def plot_box_plots(df):
    """Create box plots showing distribution of metrics."""
    print("\nCreating box plots for distribution comparison...")
    
    # Quality metrics (including perplexity)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics_quality = ['bertscore', 'bleu', 'sari', 'perplexity']
    titles = ['BERTScore', 'BLEU', 'SARI', 'Perplexity (lower is better)']
    for idx, (metric, title) in enumerate(zip(metrics_quality, titles)):
        ax = axes[idx]
        df.boxplot(column=metric, by='model_version', ax=ax,
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Score' if metric != 'perplexity' else 'Perplexity')
        ax.get_figure().suptitle('')  # Remove default title
    
    plt.tight_layout()
    save_figure(fig, 'box_plots_quality_metrics.png', 'box_plots')
    
    # Readability deltas
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # FKGL Delta
    ax = axes[0]
    df.boxplot(column='fkgl_delta', by='model_version', ax=ax,
              patch_artist=True,
              boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax.set_title('FKGL Delta Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model (Version)')
    ax.set_ylabel('FKGL Delta')
    ax.tick_params(axis='x', rotation=30)
    ax.get_figure().suptitle('')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # FRE Delta
    ax = axes[1]
    df.boxplot(column='fre_delta', by='model_version', ax=ax,
              patch_artist=True,
              boxprops=dict(facecolor='lightcoral', alpha=0.7))
    ax.set_title('FRE Delta Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model (Version)')
    ax.set_ylabel('FRE Delta')
    ax.tick_params(axis='x', rotation=30)
    ax.get_figure().suptitle('')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    save_figure(fig, 'box_plots_readability_deltas.png', 'box_plots')


def plot_scatter_plots(df):
    """Create scatter plots showing relationships between metrics."""
    print("\nCreating scatter plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # BERTScore vs SARI
    ax = axes[0, 0]
    for model in df['model_version'].unique():
        df_model = df[df['model_version'] == model]
        ax.scatter(df_model['bertscore'], df_model['sari'], 
                  label=model, alpha=0.6, s=50)
    ax.set_xlabel('BERTScore')
    ax.set_ylabel('SARI')
    ax.set_title('BERTScore vs SARI', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # BLEU vs SARI
    ax = axes[0, 1]
    for model in df['model_version'].unique():
        df_model = df[df['model_version'] == model]
        ax.scatter(df_model['bleu'], df_model['sari'], 
                  label=model, alpha=0.6, s=50)
    ax.set_xlabel('BLEU')
    ax.set_ylabel('SARI')
    ax.set_title('BLEU vs SARI', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # FKGL Delta vs FRE Delta
    ax = axes[1, 0]
    for model in df['model_version'].unique():
        df_model = df[df['model_version'] == model]
        ax.scatter(df_model['fkgl_delta'], df_model['fre_delta'], 
                  label=model, alpha=0.6, s=50)
    ax.set_xlabel('FKGL Delta (more negative = simpler)')
    ax.set_ylabel('FRE Delta (more positive = easier)')
    ax.set_title('Readability Improvement Comparison', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    
    # BERTScore vs FKGL Delta
    ax = axes[1, 1]
    for model in df['model_version'].unique():
        df_model = df[df['model_version'] == model]
        ax.scatter(df_model['bertscore'], df_model['fkgl_delta'], 
                  label=model, alpha=0.6, s=50)
    ax.set_xlabel('BERTScore')
    ax.set_ylabel('FKGL Delta')
    ax.set_title('Quality vs Simplification Depth', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    save_figure(fig, 'scatter_plots_relationships.png', 'scatter_plots')


def plot_heatmap_by_strategy(df):
    """Create heatmap comparing models across strategies."""
    print("\nCreating heatmap by strategy...")
    
    # Aggregate by model_version and strategy
    df_agg = df.groupby(['model_version', 'strategy']).agg({
        'bertscore': 'mean',
        'bleu': 'mean',
        'sari': 'mean',
        'perplexity': 'mean',
        'fkgl_delta': 'mean',
        'fre_delta': 'mean',
    }).reset_index()
    
    # Pivot for heatmap (delta metrics use center=0; perplexity lower is better, no center)
    metrics = ['bertscore', 'bleu', 'sari', 'perplexity', 'fkgl_delta', 'fre_delta']
    
    for metric in metrics:
        df_pivot = df_agg.pivot(index='strategy', columns='model_version', values=metric)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0 if 'delta' in metric else None,
                   ax=ax, cbar_kws={'label': metric})
        ax.set_title(f'{metric.upper()} by Model and Strategy', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, f'heatmap_{metric}_by_strategy.png', 'heatmaps')


def plot_strategy_comparison(df):
    """Create grouped bar charts comparing models across strategies."""
    print("\nCreating strategy comparison charts...")
    
    # Aggregate by model_version and strategy
    df_agg = df.groupby(['model_version', 'strategy']).agg({
        'bertscore': 'mean',
        'bleu': 'mean',
        'sari': 'mean',
        'perplexity': 'mean',
    }).reset_index()
    
    # Quality metrics (including perplexity)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['bertscore', 'bleu', 'sari', 'perplexity']
    titles = ['BERTScore', 'BLEU', 'SARI', 'Perplexity (lower is better)']
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        df_pivot = df_agg.pivot(index='strategy', columns='model_version', values=metric)
        df_pivot.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
        ax.set_title(f'{title} by Strategy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score' if metric != 'perplexity' else 'Perplexity')
        ax.set_xlabel('Strategy')
        ax.legend(title='Model (Version)', fontsize=7)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'strategy_comparison_quality.png', 'bar_charts')


def main():
    """Main function to generate all visualizations."""
    print("Starting visualization generation...")
    
    # Load data
    print("\n1. Loading detailed results from database...")
    df = get_detailed_results()
    print(f"✓ Loaded {len(df)} results")
    
    # Generate visualizations
    print("\n2. Generating visualizations...")
    
    plot_bar_charts_overall(df)
    plot_box_plots(df)
    plot_scatter_plots(df)
    plot_heatmap_by_strategy(df)
    plot_strategy_comparison(df)
    
    print("\n" + "="*100)
    print("✓ All visualizations generated!")
    print("="*100)
    print("\nVisualizations saved to: app/experiments/analysis/outputs/visualizations/")


if __name__ == "__main__":
    main()

