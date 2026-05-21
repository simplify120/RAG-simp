"""
Generate evaluation results tables grouped by prompt_version_id.

Each table shows:
- Rows: Models (OpenAI, Ollama, etc.)
- Columns: Metrics (BERTScore, BLEU, SARI, etc.)
- One table per prompt_version_id
- Exports to outputs/csv/prompt_versions/ with filenames: step1_zeroshot_v1.csv, step2_zeroshot_v2.csv, etc.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.experiments.analysis.aggregate_metrics import (
    aggregate_by_prompt_version,
    print_tables_by_prompt_version,
    export_tables_by_prompt_version
)

# Default descriptions for step 1 and step 2
STEP1_DESCRIPTION = "step 1 - simple prompt engineering"
STEP2_DESCRIPTION = "step 2 - RAG top k=3"  # Best versions: zeroshot v2, structured v2, constraint v1
STEP2_UPGR_DESCRIPTION = "step 2 - RAG top k=3 with upgrated prompt"  # vv3 upgraded-prompt variants


def main():
    """Generate and display/export results tables by prompt_version_id."""
    parser = argparse.ArgumentParser(
        description="Export evaluation results by prompt version to CSV"
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help=f"Filter to a single description. Default: run both step 1 and step 2",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Skip printing tables to console (only export CSV)",
    )
    args = parser.parse_args()

    if args.description:
        descriptions = [args.description]
    else:
        descriptions = [STEP1_DESCRIPTION, STEP2_DESCRIPTION, STEP2_UPGR_DESCRIPTION]

    print("="*120)
    print("EVALUATION RESULTS BY PROMPT VERSION")
    print("="*120)

    all_exported = []
    for desc in descriptions:
        print(f"\n--- Processing: {desc} ---")
        tables_dict = aggregate_by_prompt_version(description=desc)

        if not tables_dict:
            print(f"  ⚠ No data for description: {desc}")
            continue

        print(f"  ✓ Found {len(tables_dict)} prompt version(s)")
        if not args.no_print:
            print_tables_by_prompt_version(tables_dict)

        exported = export_tables_by_prompt_version(tables_dict)
        all_exported.extend(exported)

    if not all_exported:
        print("\n⚠ No evaluation data found.")
        print("Make sure you have:")
        print("  1. Run LLM comparisons to generate prompt results")
        print("  2. Calculated evaluation metrics (BERTScore, BLEU, SARI, etc.)")
        return

    print(f"\n✓ Successfully exported {len(all_exported)} table(s) to outputs/csv/prompt_versions/")
    print("="*120)


if __name__ == "__main__":
    main()
