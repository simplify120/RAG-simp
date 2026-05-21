"""
run_all.py — Master runner for the visualization suite.
Executes all plot generators and saves outputs.
"""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.experiments.visualization import (
    plot_01_phase_evolution,
    plot_02_model_comparison,
    plot_03_strategy_analysis,
    plot_04_metric_distributions,
    plot_05_readability_analysis
)

def main():
    print("="*80)
    print("Starting LLM Text Simplification Visualization Suite")
    print("="*80)
    
    output_dir = Path(__file__).parent / "outputs" / "figures"
    print(f"Outputs will be saved to: {output_dir}\n")
    
    # Run all modules
    plot_01_phase_evolution.main()
    print("-" * 40)
    plot_02_model_comparison.main()
    print("-" * 40)
    plot_03_strategy_analysis.main()
    print("-" * 40)
    plot_04_metric_distributions.main()
    print("-" * 40)
    plot_05_readability_analysis.main()
    
    print("\n" + "="*80)
    print("  [OK] Visualization Suite Complete!")
    print(f"All figures available in: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
