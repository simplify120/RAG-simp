"""
Run all evaluation metrics for a given description.

By default each metric only processes PromptResults that do not yet have that
metric stored on the linked Evaluation row (incremental / resume-friendly).

Usage:
    python -m app.experiments.evaluation.evaluate_run --description "step 2 - RAG top k=3"
    python -m app.experiments.evaluation.evaluate_run  # all PromptResults (per-metric skip still applies)
    python -m app.experiments.evaluation.evaluate_run --description "..." --force-recalculate
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure app is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

METRICS = [
    "app.experiments.evaluation.sari.calculate_sari",
    "app.experiments.evaluation.bleu.calculate_bleu",
    "app.experiments.evaluation.bertscore.calculate_bert",
    "app.experiments.evaluation.fkgl.calculate_fkgl",
    "app.experiments.evaluation.fre.calculate_fre",
    "app.experiments.evaluation.perplexity.calculate_perplexity",
    "app.experiments.evaluation.lens.calculate_lens",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run all evaluation metrics (SARI, BLEU, BERTScore, FKGL, FRE, Perplexity, LENS)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help='Only evaluate results with this description (e.g. "step 2 - RAG top k=3")',
    )
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Recompute every metric even when already stored (passed through to each calculator)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Only evaluate PromptResults with this model_name (e.g. claude-haiku-4-5 vs sonar)",
    )
    args = parser.parse_args()

    desc_arg = ["--description", args.description] if args.description else []
    force_arg = ["--force-recalculate"] if args.force_recalculate else []
    model_arg = ["--model-name", args.model_name] if args.model_name else []
    extra = desc_arg + force_arg + model_arg
    print(
        f"Running evaluation{' for description: ' + args.description if args.description else ' (all results)'}..."
        f"{' [force recalculate]' if args.force_recalculate else ' [skip rows that already have each metric]'}"
        f"{' [model=' + args.model_name + ']' if args.model_name else ''}\n"
    )

    for i, module in enumerate(METRICS, 1):
        name = module.split(".")[-1].replace("calculate_", "")
        print(f"[{i}/{len(METRICS)}] {name}...")
        result = subprocess.run(
            [sys.executable, "-m", module] + extra,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        if result.returncode != 0:
            print(f"  Failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print("\n✓ All evaluations complete.")


if __name__ == "__main__":
    main()
