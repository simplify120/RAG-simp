"""
LENS (Learnable Evaluation Metric for Text Simplification) is a reference-based
metric that correlates better with human judgment than SARI or BERTScore.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Suppress PyTorch Lightning verbose output BEFORE importing lens
os.environ.setdefault("PYTORCH_LIGHTNING_LOG_LEVEL", "ERROR")
os.environ.setdefault("PL_ENABLE_PROGRESS_BAR", "0")  # Disable "Predicting DataLoader" progress bars
for logger_name in (
    "lightning.pytorch.utilities.rank_zero",
    "pytorch_lightning.utilities.rank_zero",
    "lightning.pytorch",
    "pytorch_lightning",
):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import torch
from sqlalchemy import or_

# The LENS checkpoint was saved on CUDA. Patch torch.load to force CPU
# loading since lens-metric doesn't expose a map_location parameter.
_orig_torch_load = torch.load
def _cpu_torch_load(f, *args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")  # force CPU — pytorch_lightning passes map_location=None explicitly
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _cpu_torch_load

from lens import LENS, download_model
from app.db.session import SessionLocal
from app.models.prompt import PromptResult
from app.models.dataset import DatasetItem
from app.models.evaluation import Evaluation


LENS_MODEL_ID = "davidheineman/lens"
LENS_BATCH_SIZE = 32  # Process in batches to reduce PyTorch Lightning prediction cycles


def _fetch_results(description=None, force_recalculate=False, model_name=None, max_retries=3):
    """Fetch results with retry for transient Supabase connection timeouts."""
    for attempt in range(max_retries):
        try:
            db = SessionLocal()
            try:
                query = db.query(
                    PromptResult.result_id,
                    PromptResult.prompt_version_id,
                    PromptResult.input_text,
                    PromptResult.output_text,
                    DatasetItem.text_ele
                ).join(
                    DatasetItem, PromptResult.item_id == DatasetItem.item_id
                ).outerjoin(
                    Evaluation, Evaluation.result_id == PromptResult.result_id
                ).filter(
                    PromptResult.output_text.isnot(None),
                    DatasetItem.text_ele.isnot(None),
                )
                if not force_recalculate:
                    query = query.filter(
                        or_(
                            Evaluation.result_id.is_(None),
                            Evaluation.lens.is_(None),
                        )
                    )
                if description is not None:
                    query = query.filter(PromptResult.description == description)
                if model_name is not None:
                    query = query.filter(PromptResult.model_name == model_name)
                return query.all()
            finally:
                db.close()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"DB connection failed (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def calculate_lens_scores(description=None, force_recalculate=False, model_name=None):
    # Fetch all results first (with retry for transient Supabase timeouts)
    results = _fetch_results(
        description, force_recalculate=force_recalculate, model_name=model_name
    )

    if not results:
        print("\n✓ LENS: no rows to process (all matching results already have lens, or no PromptResults matched).")
        return

    # Download/load the LENS model checkpoint from HuggingFace (cached after first run)
    # rescale=True gives scores in 0-100 range for better interpretability
    model_path = download_model(LENS_MODEL_ID)
    lens_metric = LENS(model_path, rescale=True)

    # Process in batches - use a fresh DB session per batch to avoid connection
    # timeout during long LENS score() calls (4-8 min per batch)
    for batch_start in range(0, len(results), LENS_BATCH_SIZE):
        batch = results[batch_start : batch_start + LENS_BATCH_SIZE]
        batch_end = batch_start + len(batch)
        print(f"Processing {batch_start + 1}-{batch_end}/{len(results)}...")

        complex_texts = [r.input_text for r in batch]
        simplified_texts = [r.output_text for r in batch]
        refs = [[r.text_ele] for r in batch]

        scores = None
        try:
            scores = lens_metric.score(
                complex=complex_texts,
                simplified=simplified_texts,
                references=refs,
                batch_size=LENS_BATCH_SIZE,
                devices=[]
            )
        except Exception:
            import traceback
            traceback.print_exc()

        if scores is not None:
            for write_attempt in range(3):
                try:
                    db = SessionLocal()
                    try:
                        for i, (result_id, prompt_version_id, _, _, _) in enumerate(batch):
                            lens_score = float(scores[i]) if i < len(scores) else None

                            existing_eval = db.query(Evaluation).filter(
                                Evaluation.result_id == result_id
                            ).first()

                            if existing_eval:
                                existing_eval.lens = lens_score
                            else:
                                evaluation = Evaluation(
                                    prompt_version_id=prompt_version_id,
                                    result_id=result_id,
                                    lens=lens_score
                                )
                                db.add(evaluation)
                        db.commit()
                        break
                    finally:
                        db.close()
                except Exception as e:
                    if write_attempt < 2:
                        wait = (write_attempt + 1) * 5
                        print(f"DB write failed (attempt {write_attempt + 1}/3), retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"Error writing batch {batch_start + 1}-{batch_end}: {e}")
                        import traceback
                        traceback.print_exc()
                        raise

    print(f"\n✓ Successfully calculated and stored LENS scores:")
    print(f"  - Processed: {len(results)} results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate LENS scores for PromptResults")
    parser.add_argument("--description", type=str, default=None, help="Only process results with this description")
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Recompute LENS even when already stored on Evaluation",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Filter by PromptResult.model_name")
    args = parser.parse_args()
    calculate_lens_scores(
        description=args.description,
        force_recalculate=args.force_recalculate,
        model_name=args.model_name,
    )
