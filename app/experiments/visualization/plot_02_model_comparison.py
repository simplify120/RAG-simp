"""
plot_02_model_comparison.py — Compare LLM best configs vs T5 baseline.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

from app.experiments.visualization.config import (
    MODEL_COLORS, MODEL_LABELS, METRIC_META, METRICS_QUALITY, 
    apply_publication_style, save_figure
)
from app.experiments.visualization.data_loader import load_all_phases, load_t5_results

def plot_horizontal_bars(df_best, df_t5):
    """
    Horizontal bar chart for best LLM models vs T5.
    """
    apply_publication_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Sort models nicely
    models = sorted(df_best['model'].unique())
    
    for idx, metric in enumerate(METRICS_QUALITY):
        ax = axes[idx]
        meta = METRIC_META[metric]
        
        # Get T5 mean if available
        t5_val = None
        if not df_t5.empty:
            # Map metric names to CSV columns (case insensitivity or expected mappings)
            col_map = {'bertscore': 'BERTScore', 'bleu': 'BLEU', 'sari': 'SARI', 'lens': 'LENS', 'perplexity': 'Perplexity'}
            t5_col = col_map.get(metric, metric)
            if t5_col in df_t5.columns:
                t5_val = df_t5[t5_col].iloc[0]

        # Prepare data for plotting
        plot_data = df_best[['model', metric]].copy()
        plot_data = plot_data.sort_values(by=metric, ascending=True)
        
        colors = [MODEL_COLORS.get(m, '#333333') for m in plot_data['model']]
        labels = [MODEL_LABELS.get(m, m) for m in plot_data['model']]
        
        # Plot bars
        bars = ax.barh(labels, plot_data[metric], color=colors, alpha=0.85)
        
        # Add labels to the ends of the bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + (0.01 * width if width > 0 else 0)
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                    va='center', ha='left', fontsize=9, fontweight='bold')
        
        # Add T5 reference line
        if t5_val is not None and not math.isnan(t5_val):
            ax.axvline(x=t5_val, color=MODEL_COLORS.get("t5-large-text-simplification", 'red'), 
                       linestyle='--', linewidth=2, zorder=0)
            
            # Label the line (top of the plot)
            ax.text(t5_val, len(labels)-0.2, f" T5 Baseline ({t5_val:.3f})", 
                    color=MODEL_COLORS.get("t5-large-text-simplification", 'red'), 
                    fontsize=10, fontweight='bold', va='bottom', ha='left')
        
        direction = "(Higher is better)" if meta["higher_is_better"] else "(Lower is better)"
        ax.set_title(f"{meta['label']} — Best Config vs T5\n{direction}", fontweight='bold', pad=15)
        ax.set_xlabel(meta['label'])
        
        # Adjust x-limit to make room for text labels
        current_xlim = ax.get_xlim()
        ax.set_xlim(current_xlim[0], current_xlim[1] * 1.15)
        
    plt.tight_layout()
    save_figure(fig, '01_best_models_vs_t5_bar', '02_model_comparison')


def main():
    print("Generating Model Comparison plots...")
    df_all = load_all_phases()
    df_t5 = load_t5_results()
    
    if df_all.empty:
        print("  ⚠ No evaluation data found in database.")
        return
        
    # Find the "best" config for each model.
    # We'll use the phase/strategy combination that yields the highest average SARI
    df_agg = df_all.groupby(["phase", "strategy", "model"]).mean(numeric_only=True).reset_index()
    
    best_rows = []
    for model in df_agg['model'].unique():
        model_data = df_agg[df_agg['model'] == model]
        best_row = model_data.loc[model_data['sari'].idxmax()]
        best_rows.append(best_row)
        
    df_best = pd.DataFrame(best_rows)
    print("Best configurations based on SARI:")
    for _, r in df_best.iterrows():
        print(f"  {r['model']}: Phase={r['phase']}, Strategy={r['strategy']} (SARI={r['sari']:.3f})")

    plot_horizontal_bars(df_best, df_t5)
    print("Done plot_02.")


if __name__ == "__main__":
    main()
