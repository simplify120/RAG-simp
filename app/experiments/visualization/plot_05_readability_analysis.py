"""
plot_05_readability_analysis.py — Show FKGL and FRE input -> output changes
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from app.experiments.visualization.config import (
    MODEL_COLORS, MODEL_LABELS, METRIC_META, apply_publication_style, save_figure
)
from app.experiments.visualization.data_loader import load_all_phases

def plot_fkgl_arrows(df_all):
    """
    Plot representing the shift from Original FKGL to Output FKGL.
    Each model gets a mean arrow.
    """
    apply_publication_style()
    
    # Group by model
    df_agg = df_all.groupby('model').mean(numeric_only=True).reset_index()
    
    if 'fkgl_input' not in df_agg.columns or 'fkgl_output' not in df_agg.columns:
        print("  ⚠ FKGL input/output columns missing, skipping arrow plot.")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot baseline distribution of input texts
    # We take the unique raw inputs from df_all
    inputs = df_all[['item_id', 'fkgl_input']].drop_duplicates()
    sns.kdeplot(data=inputs, x='fkgl_input', color='gray', fill=True, alpha=0.2, ax=ax, label='Original Input Density')
    
    # Draw arrows
    for _, row in df_agg.iterrows():
        model = row['model']
        start_x = row['fkgl_input']
        end_x = row['fkgl_output']
        
        # Calculate arrow drop vertically just to separate models visually
        y_pos = 0.05 + 0.05 * (list(df_agg['model']).index(model))
        
        color = MODEL_COLORS.get(model, '#333')
        label = MODEL_LABELS.get(model, model)
        
        # The line
        ax.annotate(
            "",
            xy=(end_x, y_pos), xycoords='data',
            xytext=(start_x, y_pos), textcoords='data',
            arrowprops=dict(arrowstyle="->", color=color, lw=2.5, shrinkA=0, shrinkB=0)
        )
        
        # Plot start and end points
        ax.plot(start_x, y_pos, 'o', color='black', markersize=6)
        
        # Label above arrow
        ax.text((start_x + end_x) / 2, y_pos + 0.01, f"{label} (Δ {end_x - start_x:.1f})", 
                ha='center', va='bottom', color=color, fontweight='bold', fontsize=10)
        
    ax.set_title("FKGL Readability Shift (Mean)\n(Arrows pointing left indicate simplification)", fontweight='bold', pad=15)
    ax.set_xlabel("Flesch-Kincaid Grade Level")
    ax.set_ylabel("Kernel Density (Inputs)")
    ax.set_yticks([]) # remove y-ticks as they are mostly structural
    
    # Add a vertical line for elementary level target (e.g. 5th grade)
    ax.axvline(5.0, color='red', linestyle='--', alpha=0.5, label='Target (5th Grade)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    save_figure(fig, '01_fkgl_shift_arrows', '05_readability')


def plot_fre_delta_scatter(df_all):
    """
    Scatter plot of Input Readability vs Readability Gain.
    Shows if models simplify harder texts more.
    """
    apply_publication_style()
    
    if 'fre_input' not in df_all.columns or 'fre_delta' not in df_all.columns:
        return
        
    # Pick the best config for each model to avoid overplotting
    df_agg = df_all.groupby(["phase", "strategy", "model"]).mean(numeric_only=True).reset_index()
    best_configs = []
    
    for model in df_agg['model'].unique():
        model_data = df_agg[df_agg['model'] == model]
        if not model_data.empty:
            best_row = model_data.loc[model_data['sari'].idxmax()] # best by sari
            best_configs.append((model, best_row['phase'], best_row['strategy']))
            
    df_best_list = []
    for model, phase, strat in best_configs:
        subset = df_all[(df_all['model'] == model) & (df_all['phase'] == phase) & (df_all['strategy'] == strat)]
        df_best_list.append(subset)
    
    if not df_best_list:
        return
        
    df_plot = pd.concat(df_best_list, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(
        data=df_plot,
        x='fre_input',
        y='fre_delta',
        hue='model',
        palette=MODEL_COLORS,
        alpha=0.7,
        s=60,
        ax=ax
    )
    
    # Add trendlines
    for model in df_plot['model'].unique():
        sns.regplot(
            data=df_plot[df_plot['model'] == model],
            x='fre_input',
            y='fre_delta',
            scatter=False,
            color=MODEL_COLORS.get(model, '#333'),
            ci=None, # no confidence interval shading
            ax=ax
        )
    
    # Close the extra figure created by lmplot
    plt.close()
    
    ax.set_title("Input Difficulty vs. Readability Gain\n(Flesch Reading Ease)", fontweight='bold', pad=15)
    ax.set_xlabel("Input FRE Score (Lower = Harder)")
    ax.set_ylabel("FRE Delta (Positive = Simplified)")
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [MODEL_LABELS.get(l, l) for l in labels], title="", loc='best')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    save_figure(fig, '02_fre_difficulty_vs_gain', '05_readability')


def main():
    print("Generating Readability Analysis plots...")
    df_all = load_all_phases(include_t5=True)
    
    plot_fkgl_arrows(df_all)
    plot_fre_delta_scatter(df_all)
    print("Done plot_05.")


if __name__ == "__main__":
    main()
