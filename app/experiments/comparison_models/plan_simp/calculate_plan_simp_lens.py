"""
Calculate LENS scores for Plan-Simp rows in plan_simp_text_simplification_evaluation.

Usage:
    python -m app.experiments.comparison_models.plan_simp.calculate_plan_simp_lens
    python -m app.experiments.comparison_models.plan_simp.calculate_plan_simp_lens \\
        --split complement --only-missing
    python -m app.experiments.comparison_models.plan_simp.calculate_plan_simp_lens --split all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal

os.environ.setdefault("PYTORCH_LIGHTNING_LOG_LEVEL", "ERROR")
os.environ.setdefault("PL_ENABLE_PROGRESS_BAR", "0")
for logger_name in (
    "lightning.pytorch.utilities.rank_zero",
    "pytorch_lightning.utilities.rank_zero",
    "lightning.pytorch",
    "pytorch_lightning",
):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

_orig_torch_load = torch.load


def _cpu_torch_load(f, *args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return _orig_torch_load(f, *args, **kwargs)


torch.load = _cpu_torch_load

from lens import LENS, download_model

from app.db.session import SessionLocal
from app.experiments.comparison_models.plan_simp.run_plan_simp_pipeline import (
    get_complement_items_bge_split,
    get_test_items_bge_split,
)
from app.models.dataset import DatasetItem
from app.models.plan_simp_evaluation import PlanSimpTextSimplificationEvaluation

LENS_MODEL_ID = "davidheineman/lens"
LENS_BATCH_SIZE = 32


def _load_valid_lens_specs(
    split: Literal["test", "complement", "all"],
    only_missing: bool,
) -> List[Dict[str, Any]]:
    """Fetch rows from DB and return plain dicts (no open session after return)."""
    db = SessionLocal()
    try:
        if split == "test":
            split_item_ids = {item[0] for item in get_test_items_bge_split(db)}
            query = db.query(PlanSimpTextSimplificationEvaluation).filter(
                PlanSimpTextSimplificationEvaluation.item_id.in_(split_item_ids)
            )
        elif split == "complement":
            split_item_ids = {item[0] for item in get_complement_items_bge_split(db)}
            query = db.query(PlanSimpTextSimplificationEvaluation).filter(
                PlanSimpTextSimplificationEvaluation.item_id.in_(split_item_ids)
            )
        elif split == "all":
            query = db.query(PlanSimpTextSimplificationEvaluation)
        else:
            raise ValueError(f"Unknown split: {split!r}")

        if only_missing:
            query = query.filter(PlanSimpTextSimplificationEvaluation.lens.is_(None))

        rows = query.all()

        item_ids = {r.item_id for r in rows}
        dataset_items = (
            db.query(DatasetItem).filter(DatasetItem.item_id.in_(item_ids)).all()
        )
        reference_map = {str(d.item_id): d.text_ele for d in dataset_items}

        valid_specs: List[Dict[str, Any]] = []
        for row in rows:
            ref = reference_map.get(str(row.item_id))
            if ref and row.output_text and row.input_text:
                valid_specs.append(
                    {
                        "id": row.id,
                        "input_text": row.input_text,
                        "output_text": row.output_text,
                        "reference": ref,
                    }
                )
        return valid_specs
    finally:
        db.close()


def calculate_plan_simp_lens_scores(
    split: Literal["test", "complement", "all"] = "test",
    only_missing: bool = False,
) -> None:
    # Short DB read only — hosted Postgres (e.g. Supabase) often kills idle connections;
    # holding a session open during 20+ minutes of LENS inference caused commit() to fail.
    valid_specs = _load_valid_lens_specs(split, only_missing)
    n = len(valid_specs)
    print(f"Split={split}, only_missing={only_missing}, scoring {n} rows.")

    model_path = download_model(LENS_MODEL_ID)
    lens_metric = LENS(model_path, rescale=True)

    committed = 0
    for batch_start in range(0, n, LENS_BATCH_SIZE):
        batch = valid_specs[batch_start : batch_start + LENS_BATCH_SIZE]
        batch_end = batch_start + len(batch)
        print(f"Processing {batch_start + 1}-{batch_end}/{n}...")

        complex_texts = [s["input_text"] for s in batch]
        simplified_texts = [s["output_text"] for s in batch]
        refs = [[s["reference"]] for s in batch]

        try:
            scores = lens_metric.score(
                complex=complex_texts,
                simplified=simplified_texts,
                references=refs,
                batch_size=LENS_BATCH_SIZE,
                devices=[],
            )
        except Exception:
            traceback.print_exc()
            raise

        db = SessionLocal()
        try:
            batch_commits = 0
            for i, spec in enumerate(batch):
                if i < len(scores):
                    val = scores[i]
                    if hasattr(val, "item"):
                        val = val.item()
                    db.query(PlanSimpTextSimplificationEvaluation).filter(
                        PlanSimpTextSimplificationEvaluation.id == spec["id"]
                    ).update({"lens": float(val)}, synchronize_session=False)
                    batch_commits += 1
            db.commit()
            committed += batch_commits
            print(f"  Committed LENS for {batch_commits} rows ({batch_start + 1}-{batch_end}).")
        except Exception as e:
            db.rollback()
            print(f"Error committing batch {batch_start + 1}-{batch_end}: {e}")
            traceback.print_exc()
            raise
        finally:
            db.close()

    print(f"LENS scores stored for {committed} Plan-Simp rows (committed per batch).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill LENS into plan_simp_text_simplification_evaluation")
    parser.add_argument(
        "--split",
        choices=("test", "complement", "all"),
        default="test",
        help="test: 40-item BGE sample; complement: remaining text_adv items; all: every evaluation row",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only update rows where lens IS NULL",
    )
    args = parser.parse_args()
    calculate_plan_simp_lens_scores(split=args.split, only_missing=args.only_missing)


if __name__ == "__main__":
    main()
