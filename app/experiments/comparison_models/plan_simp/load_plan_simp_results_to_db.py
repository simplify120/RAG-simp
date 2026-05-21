"""
Load plan_simp_raw_results.jsonl (from run_plan_simp_pipeline.py) into the database
with metrics. No GPU inference — safe to run with a flaky connection if rows commit
individually.

Usage:
    python -m app.experiments.comparison_models.plan_simp.load_plan_simp_results_to_db
    python -m app.experiments.comparison_models.plan_simp.load_plan_simp_results_to_db \\
        --jsonl app/experiments/comparison_models/plan_simp/outputs/plan_simp_raw_results.jsonl
    python -m app.experiments.comparison_models.plan_simp.load_plan_simp_results_to_db --allow-duplicate
    python -m app.experiments.comparison_models.plan_simp.load_plan_simp_results_to_db \\
        --jsonl app/experiments/comparison_models/plan_simp/outputs_complement/plan_simp_raw_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

import torch
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from app.db.session import SessionLocal
from app.experiments.comparison_models.t5_model.metrics import (
    compute_bertscore,
    compute_bleu,
    compute_fkgl,
    compute_fre,
    compute_perplexity,
    compute_sari,
)
from app.models.plan_simp_evaluation import PlanSimpTextSimplificationEvaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PERPLEXITY_MODEL_ID = "distilgpt2"


def compute_all_metrics(
    input_text: str,
    output_text: str,
    reference_text: Optional[str],
    perplexity_model,
    perplexity_tokenizer,
    device: str,
) -> Dict[str, Any]:
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


def exists_for_item(db, item_id: UUID) -> bool:
    return (
        db.query(PlanSimpTextSimplificationEvaluation)
        .filter(PlanSimpTextSimplificationEvaluation.item_id == item_id)
        .first()
        is not None
    )


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Plan-Simp JSONL into PostgreSQL")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "plan_simp_raw_results.jsonl",
        help="Path to plan_simp_raw_results.jsonl",
    )
    parser.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Insert even if this item_id already has a row (default: skip existing)",
    )
    args = parser.parse_args()

    if not args.jsonl.is_file():
        raise SystemExit(f"JSONL not found: {args.jsonl}")

    records = load_jsonl(args.jsonl)
    if not records:
        raise SystemExit("JSONL is empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading perplexity model on %s", device)
    perplexity_tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL_ID)
    perplexity_model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL_ID).to(device)
    if perplexity_tokenizer.pad_token is None:
        perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token
    perplexity_model.eval()

    db = SessionLocal()
    inserted = 0
    skipped = 0
    failed = 0

    try:
        for rec in records:
            item_id = UUID(rec["item_id"])
            input_text = rec["input_text"]
            output_text = rec.get("output_text") or ""
            reference_text = rec.get("reference_text")

            if not args.allow_duplicate and exists_for_item(db, item_id):
                skipped += 1
                continue

            try:
                metrics = compute_all_metrics(
                    input_text,
                    output_text,
                    reference_text,
                    perplexity_model,
                    perplexity_tokenizer,
                    device,
                )
            except Exception as e:
                failed += 1
                logger.warning("Metrics failed for item %s: %s", item_id, e)
                continue

            row = PlanSimpTextSimplificationEvaluation(
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

            try:
                db.add(row)
                db.commit()
                inserted += 1
            except (OperationalError, SQLAlchemyError) as e:
                db.rollback()
                failed += 1
                logger.error(
                    "DB error for item %s (rolled back, continuing): %s",
                    item_id,
                    e,
                )

        logger.info("Done. Inserted: %s, Skipped: %s, Failed: %s", inserted, skipped, failed)
    finally:
        db.close()


if __name__ == "__main__":
    main()
