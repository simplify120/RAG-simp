"""
plot_04_metric_distributions.py — Show metric distributions across the 40 texts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from app.experiments.visualization.config import (
    MODEL_COLORS, MODEL_LABELS, STRATEGY_ORDER, METRIC_META,
    apply_publication_style, save_figure
)
from app.experiments.visualization.data_loader import load_all_phases

def plot_violins_with_strips(df_all):
    """
    Violin plot overlaid with individual data points (strip plot)
    to show the distribution of scores across the 40 test items.
    """
    apply_publication_style()
    
    # We'll isolate the best strategy for each model in its latest phase 
    # to show a clean distribution of the "best" results.
    
    # Simple heuristic: for each model, find the phase/strategy with the highest mean SARI
    df_agg = df_all.groupby(["phase", "strategy", "model"]).mean(numeric_only=True).reset_index()
    best_configs = []
    
    for model in df_agg['model'].unique():
        model_data = df_agg[df_agg['model'] == model]
        if not model_data.empty:
            best_row = model_data.loc[model_data['sari'].idxmax()]
            best_configs.append((model, best_row['phase'], best_row['strategy']))
            
    # Filter full dataframe to just these best configs
    df_best_list = []
    for model, phase, strat in best_configs:
        subset = df_all[(df_all['model'] == model) & (df_all['phase'] == phase) & (df_all['strategy'] == strat)]
        df_best_list.append(subset)
        
    if not df_best_list:
        print("⚠ No valid configurations found for distributions.")
        return
        
    df_plot = pd.concat(df_best_list, ignore_index=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics = ['sari', 'bertscore', 'bleu', 'lens']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        meta = METRIC_META[metric]
        
        # Plot violins
        sns.violinplot(
            data=df_plot,
            x='model',
            y=metric,
            palette=MODEL_COLORS,
            inner=None,          # no mini-box inside
            linewidth=0,
            alpha=0.4,
            ax=ax
        )
        
        # Overlay points
        sns.stripplot(
            data=df_plot,
            x='model',
            y=metric,
            color='black',
            alpha=0.6,
            jitter=0.2,
            size=4,
            ax=ax
        )
        
        direction = "(Higher is better)" if meta["higher_is_better"] else "(Lower is better)"
        ax.set_title(f"{meta['label']} Distribution (40 texts)\nBest configuration per model", fontweight='bold', pad=15)
        ax.set_xlabel("")
        ax.set_ylabel(meta['label'])
        
        # X-labels
        ax.set_xticklabels([MODEL_LABELS.get(p.get_text(), p.get_text()) for p in ax.get_xticklabels()], rotation=15, ha='right')

    plt.tight_layout()
    save_figure(fig, '01_metric_distributions_violin', '04_metric_distributions')


def plot_correlation_heatmap(df_all):
    """
    Correlation heatmap of the various metrics.
    """
    apply_publication_style()
    
    metrics = ['sari', 'bertscore', 'bleu', 'perplexity', 'fkgl_delta', 'fre_delta', 'lens']
    # Filter to numeric columns only
    df_metrics = df_all[[m for m in metrics if m in df_all.columns]].dropna()
    
    if df_metrics.empty:
        return
        
    corr = df_metrics.corr()
    
    # Rename for display
    corr.columns = [METRIC_META[c]['label'] for c in corr.columns]
    corr.index = [METRIC_META[c]['label'] for c in corr.index]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )
    
    ax.set_title("Metric Correlation Across All Models & Output", fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_figure(fig, '02_metric_correlation', '04_metric_distributions')


def main():
    print("Generating Metric Distributions plots...")
    df_all = load_all_phases(include_t5=True)
    
    plot_violins_with_strips(df_all)
    plot_correlation_heatmap(df_all)
    print("Done plot_04.")


if __name__ == "__main__":
    main()
