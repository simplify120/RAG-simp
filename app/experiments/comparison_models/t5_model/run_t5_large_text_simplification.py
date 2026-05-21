"""
Run T5-large text simplification on dataset_items and store results with evaluation metrics.

Usage:
    python -m app.experiments.comparison_models.t5_model.run_t5_large_text_simplification
    python -m app.experiments.comparison_models.t5_model.run_t5_large_text_simplification --limit 10
"""

import argparse
import logging
import sys
from pathlib import Path
from uuid import UUID
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# Ensure app is on path when run as module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.db.session import SessionLocal
from app.models.dataset import DatasetItem
from app.models.t5_evaluation import T5LargeTextSimplificationEvaluation
from app.experiments.comparison_models.t5_model.metrics import (
    compute_bertscore,
    compute_bleu,
    compute_fkgl,
    compute_fre,
    compute_perplexity,
    compute_sari,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "eilamc14/t5-large-text-simplification"
PREFIX = "Simplify: "
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 128
NUM_BEAMS = 4


def fetch_dataset_items(
    db, limit: Optional[int] = None
) -> List[Tuple[UUID, str, Optional[str]]]:
    """Fetch item_id, text_adv, text_ele from dataset_items where text_adv is not null."""
    query = db.query(
        DatasetItem.item_id,
        DatasetItem.text_adv,
        DatasetItem.text_ele,
    ).filter(DatasetItem.text_adv.isnot(None))
    rows = query.all()
    if limit:
        rows = rows[:limit]
    return [(r.item_id, r.text_adv, r.text_ele) for r in rows]


def exists_for_item(db, item_id: UUID) -> bool:
    """Check if a record already exists for the given item_id."""
    return (
        db.query(T5LargeTextSimplificationEvaluation)
        .filter(T5LargeTextSimplificationEvaluation.item_id == item_id)
        .first()
        is not None
    )


def run_inference(
    text_adv: str,
    tokenizer,
    model,
    device,
) -> str:
    """Run T5 simplification on input text."""
    input_text = PREFIX + text_adv
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_all_metrics(
    input_text: str,
    output_text: str,
    reference_text: Optional[str],
    perplexity_model,
    perplexity_tokenizer,
    device,
) -> dict:
    """Compute all evaluation metrics."""
    sari = compute_sari(input_text, output_text, reference_text)
    bertscore_f1 = compute_bertscore(output_text, reference_text)
    bleu = compute_bleu(output_text, reference_text)
    fkgl_input, fkgl_output, delta_fkgl = compute_fkgl(input_text, output_text)
    fre_input, fre_output, fre_delta = compute_fre(input_text, output_text)
    perplexity = compute_perplexity(
        output_text, perplexity_model, perplexity_tokenizer, device
    )
    return {
        "sari": sari,
        "bertscore_f1": bertscore_f1,
        "fkgl_input": fkgl_input,
        "fkgl_output": fkgl_output,
        "delta_fkgl": delta_fkgl,
        "fre_input": fre_input,
        "fre_output": fre_output,
        "fre_delta": fre_delta,
        "bleu": bleu,
        "perplexity": perplexity,
    }


def insert_result(db, item_id: UUID, input_text: str, output_text: str, metrics: dict):
    """Insert a result row into t5_large_text_simplification_evaluation."""
    record = T5LargeTextSimplificationEvaluation(
        item_id=item_id,
        input_text=input_text,
        output_text=output_text,
        sari=metrics.get("sari"),
        bertscore_f1=metrics.get("bertscore_f1"),
        fkgl_input=metrics.get("fkgl_input"),
        fkgl_output=metrics.get("fkgl_output"),
        delta_fkgl=metrics.get("delta_fkgl"),
        fre_input=metrics.get("fre_input"),
        fre_output=metrics.get("fre_output"),
        fre_delta=metrics.get("fre_delta"),
        bleu=metrics.get("bleu"),
        perplexity=metrics.get("perplexity"),
    )
    db.add(record)
    db.commit()


def main():
    parser = argparse.ArgumentParser(
        description="Run T5-large text simplification on dataset_items and store results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N items (for testing)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load T5 model
    logger.info(f"Loading T5 model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # Load perplexity model (distilgpt2)
    logger.info("Loading perplexity model (distilgpt2)")
    perplexity_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    perplexity_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    if perplexity_tokenizer.pad_token is None:
        perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token
    perplexity_model.eval()

    db = SessionLocal()
    try:
        items = fetch_dataset_items(db, limit=args.limit)
        logger.info(f"Found {len(items)} dataset items to process")

        processed = 0
        skipped = 0
        failed = 0

        for idx, (item_id, text_adv, text_ele) in enumerate(items, 1):
            if idx % 10 == 0 or idx == 1:
                logger.info(f"Processing {idx}/{len(items)}...")

            if exists_for_item(db, item_id):
                skipped += 1
                continue

            try:
                output_text = run_inference(text_adv, tokenizer, model, device)
                metrics = compute_all_metrics(
                    text_adv,
                    output_text,
                    text_ele,
                    perplexity_model,
                    perplexity_tokenizer,
                    device,
                )
                insert_result(db, item_id, text_adv, output_text, metrics)
                processed += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Failed item {item_id}: {e}")

        logger.info(
            f"\nDone. Processed: {processed}, Skipped: {skipped}, Failed: {failed}"
        )
    except Exception as e:
        db.rollback()
        logger.exception(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
