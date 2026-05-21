"""
plot_03_strategy_analysis.py — Analyze Constraint vs Structured vs Zero-shot
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from app.experiments.visualization.config import (
    MODEL_COLORS, MODEL_LABELS, STRATEGY_ORDER, STRATEGY_LABELS, STRATEGY_COLORS,
    METRIC_META, apply_publication_style, save_figure
)
from app.experiments.visualization.data_loader import load_all_phases

def plot_strategy_heatmap(df_agg):
    """
    Heatmap of Strategy (rows) x Model (columns) for SARI.
    Values are averaged across all phases.
    """
    apply_publication_style()
    
    # We focus on the quality metrics, create multiple heatmaps
    metrics_to_plot = ['sari', 'bertscore']
    
    for metric in metrics_to_plot:
        meta = METRIC_META[metric]
        
        # Aggregate
        df_pivot = df_agg.pivot_table(
            index='strategy', 
            columns='model', 
            values=metric, 
            aggfunc='mean'
        )
        
        # Reorder rows
        existing_strats = [s for s in STRATEGY_ORDER if s in df_pivot.index]
        df_pivot = df_pivot.reindex(existing_strats)
        
        # Rename row/col labels for visual appeal
        df_pivot.index = [STRATEGY_LABELS.get(idx, idx) for idx in df_pivot.index]
        df_pivot.columns = [MODEL_LABELS.get(col, col) for col in df_pivot.columns]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            df_pivot, 
            annot=True, 
            fmt='.3f', 
            cmap='YlGnBu' if meta['higher_is_better'] else 'YlOrRd',
            cbar_kws={'label': meta['label']},
            ax=ax
        )
        
        ax.set_title(f"{meta['label']} by Strategy and Model\n(Averaged across phases)", fontweight='bold', pad=15)
        ax.set_ylabel("Prompting Strategy")
        ax.set_xlabel("Model")
        
        plt.tight_layout()
        save_figure(fig, f'01_strategy_heatmap_{metric}', '03_strategy_analysis')


def plot_strategy_boxplots(df_all):
    """
    Box/violin plots showing the distribution of SARI across the 40 texts, split by strategy.
    """
    apply_publication_style()
    
    # Filter to one phase for clarity, e.g. the final RAG phase if available, 
    # or just use all data grouped by strategy. Let's group by Strategy + Model.
    
    # Restrict to the 3 main strategies
    df_plot = df_all[df_all['strategy'].isin(STRATEGY_ORDER)].copy()
    df_plot['strategy'] = pd.Categorical(df_plot['strategy'], categories=STRATEGY_ORDER, ordered=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, metric in enumerate(['sari', 'bertscore']):
        ax = axes[idx]
        meta = METRIC_META[metric]
        
        sns.boxplot(
            data=df_plot,
            x='strategy',
            y=metric,
            hue='model',
            palette=MODEL_COLORS,
            ax=ax,
            boxprops={'alpha': 0.8},
            fliersize=2
        )
        
        ax.set_title(f"{meta['label']} Distribution by Strategy", fontweight='bold', pad=15)
        ax.set_xlabel("Strategy")
        ax.set_ylabel(meta['label'])
        ax.set_xticklabels([STRATEGY_LABELS.get(p.get_text(), p.get_text()) for p in ax.get_xticklabels()])
        
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            friendly_labels = [MODEL_LABELS.get(l, l) for l in labels]
            ax.legend(handles, friendly_labels, title="Model", loc='best', framealpha=0.9)
        else:
            ax.get_legend().remove()
            
    plt.tight_layout()
    save_figure(fig, '02_strategy_boxplots', '03_strategy_analysis')


def main():
    print("Generating Strategy Analysis plots...")
    df_all = load_all_phases()
    
    if df_all.empty:
        print("  ⚠ No evaluation data found in database.")
        return
        
    df_agg = df_all.groupby(["phase", "strategy", "model"]).mean(numeric_only=True).reset_index()
    
    plot_strategy_heatmap(df_agg)
    plot_strategy_boxplots(df_all)
    print("Done plot_03.")


if __name__ == "__main__":
    main()
