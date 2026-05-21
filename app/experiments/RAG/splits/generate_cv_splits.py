"""
Generate rag_cv_v1.json: fixed 40-item test split (seed 42) + 4 folds on the complement (149).

Mirrors app.experiments.RAG.openai.build_embedding_index_test_set.get_test_items logic exactly:
  1. All dataset_items with non-null text_adv, .all() (no ORDER BY).
  2. random.seed(42), random.sample(..., 40) -> test_ids.
  3. Complement sorted by str(item_id); chunks [0:37], [37:74], [74:111], [111:149].

Usage (from repo root):
    python -m app.experiments.RAG.splits.generate_cv_splits
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

from app.db.session import SessionLocal
from app.models.dataset import DatasetItem

RANDOM_SEED = 42
TEST_SAMPLE_SIZE = 40
FOLD_BOUNDARIES = [(0, 37), (37, 74), (74, 111), (111, 149)]


def main() -> None:
    out_path = Path(__file__).resolve().parent / "rag_cv_v1.json"

    db = SessionLocal()
    try:
        all_items: list[tuple[uuid.UUID, str]] = (
            db.query(DatasetItem.item_id, DatasetItem.text_adv)
            .filter(DatasetItem.text_adv.isnot(None))
            .all()
        )

        random.seed(RANDOM_SEED)
        k = min(TEST_SAMPLE_SIZE, len(all_items))
        test_item_ids = {item[0] for item in random.sample(all_items, k)}

        complement_ids = sorted(
            (item[0] for item in all_items if item[0] not in test_item_ids),
            key=str,
        )

        if len(test_item_ids) != k:
            raise RuntimeError("internal: test set size mismatch")
        if len(complement_ids) != len(all_items) - k:
            raise RuntimeError("internal: complement size mismatch")

        if len(all_items) != 189 or len(complement_ids) != 149:
            raise ValueError(
                "This manifest expects exactly 189 items with text_adv "
                f"(40 test + 149 complement). Got total={len(all_items)}, "
                f"complement={len(complement_ids)}."
            )

        complement_folds: dict[str, list[str]] = {}
        for fold_idx, (lo, hi) in enumerate(FOLD_BOUNDARIES):
            chunk = complement_ids[lo:hi]
            complement_folds[str(fold_idx)] = [str(uid) for uid in chunk]

        expected_sizes = [37, 37, 37, 38]
        for fold_idx, expected in enumerate(expected_sizes):
            actual = len(complement_folds[str(fold_idx)])
            if actual != expected:
                raise ValueError(
                    f"Fold {fold_idx}: expected {expected} items, got {actual}."
                )

        payload = {
            "version": "v1",
            "random_seed": RANDOM_SEED,
            "test_sample_size": k,
            "total_items": len(all_items),
            "test_ids": sorted((str(uid) for uid in test_item_ids), key=str),
            "complement_folds": complement_folds,
        }

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        print(f"  total_items={len(all_items)}, test={len(test_item_ids)}, complement={len(complement_ids)}")
        for i in range(4):
            print(f"  fold {i}: {len(complement_folds[str(i)])} items")

    finally:
        db.close()


if __name__ == "__main__":
    main()
