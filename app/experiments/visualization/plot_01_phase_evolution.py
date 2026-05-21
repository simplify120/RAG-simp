"""
plot_01_phase_evolution.py — Show performance evolution across phases.
PE_v1 -> PE_v2 -> RAG_v2 -> RAG_v3
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from app.experiments.visualization.config import (
    PHASE_ORDER, PHASE_LABELS, MODEL_COLORS, MODEL_LABELS, 
    METRIC_META, METRICS_QUALITY, STRATEGY_ORDER, apply_publication_style, save_figure
)
from app.experiments.visualization.data_loader import load_all_phases

def plot_line_evolution(df_agg):
    """
    Line chart showing the evolution of each model across phases.
    Averaged across strategies for clarity, or showing the "best" strategy.
    """
    apply_publication_style()
    
    # We will aggregate across strategies (mean) to show overall model trajectory
    df_model_phase = df_agg.groupby(['phase', 'model']).mean(numeric_only=True).reset_index()
    
    # Ensure categorical ordering
    df_model_phase['phase'] = pd.Categorical(df_model_phase['phase'], categories=PHASE_ORDER, ordered=True)
    df_model_phase = df_model_phase.sort_values('phase')
    
    # Create a 2x2 grid for the 4 quality metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(METRICS_QUALITY):
        ax = axes[idx]
        meta = METRIC_META[metric]
        
        # Plot lines
        sns.pointplot(
            data=df_model_phase, 
            x='phase', 
            y=metric, 
            hue='model',
            palette=MODEL_COLORS,
            markers=['o', 's', 'D', '^', 'v', 'p'],
            linewidth=2,
            ax=ax
        )
        
        # Styling
        direction = "(Higher is better)" if meta["higher_is_better"] else "(Lower is better)"
        ax.set_title(f"{meta['label']} Evolution\n{direction}", fontweight='bold', pad=15)
        ax.set_xlabel("")
        ax.set_ylabel(f"{meta['label']} {meta['unit']}".strip())
        
        # Custom x-tick labels
        ax.set_xticklabels([PHASE_LABELS.get(p.get_text(), p.get_text()) for p in ax.get_xticklabels()])
        
        # Clean up legend (only keep one, or put outside)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            # use friendly labels
            friendly_labels = [MODEL_LABELS.get(l, l) for l in labels]
            ax.legend(handles, friendly_labels, title="", loc='best', frameon=True)
        else:
            ax.get_legend().remove()
            
    plt.tight_layout()
    save_figure(fig, '01_evolution_lines_by_model', '01_phase_evolution')


def plot_grouped_bar_evolution(df_all):
    """
    Grouped bar chart showing the performance jump.
    Comparing the overall mean of Step 1 (PE) vs Step 2 (RAG).
    """
    apply_publication_style()
    
    # Collapse phases into PE vs RAG
    df_all['super_phase'] = df_all['phase'].apply(lambda x: "Prompt Engineering (PE)" if x.startswith("PE") else "Retrieval Augmented Gen (RAG)")
    
    # Filter out UNKNOWN or Baseline if present
    df_plot = df_all[df_all['super_phase'].isin(["Prompt Engineering (PE)", "Retrieval Augmented Gen (RAG)"])]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(METRICS_QUALITY):
        ax = axes[idx]
        meta = METRIC_META[metric]
        
        sns.barplot(
            data=df_plot,
            x='model',
            y=metric,
            hue='super_phase',
            palette=["#95a5a6", "#2ecc71"], # Gray for PE, Green for RAG
            errorbar=('ci', 95),
            capsize=.05,
            ax=ax
        )
        
        direction = "(Higher is better)" if meta["higher_is_better"] else "(Lower is better)"
        ax.set_title(f"{meta['label']} — PE vs RAG Phase\n{direction}", fontweight='bold', pad=15)
        ax.set_xlabel("")
        ax.set_ylabel(meta['label'])
        
        # x-tick labels
        ax.set_xticklabels([MODEL_LABELS.get(p.get_text(), p.get_text()) for p in ax.get_xticklabels()], rotation=15, ha='right')
        
        if idx == 0:
            ax.legend(title="", loc='best', frameon=True)
        else:
            ax.get_legend().remove()
            
    plt.tight_layout()
    save_figure(fig, '02_evolution_bars_pe_vs_rag', '01_phase_evolution')


def main():
    print("Generating Phase Evolution plots...")
    df_all = load_all_phases()
    
    if df_all.empty:
        print("  ⚠ No evaluation data found in database.")
        return
        
    df_agg = df_all.groupby(["phase", "strategy", "model"]).mean(numeric_only=True).reset_index()
    
    plot_line_evolution(df_agg)
    plot_grouped_bar_evolution(df_all)
    print("Done plot_01.")


if __name__ == "__main__":
    main()
