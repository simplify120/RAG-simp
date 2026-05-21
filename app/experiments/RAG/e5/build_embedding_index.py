"""
build_embedding_index.py (E5 version)
------------------------------------
Backfills public.dataset_item_embeddings_e5_768 with E5 embeddings
for the TRAIN split of dataset_items (all items except the 40 test items).

Usage:
    python -m app.experiments.RAG.e5.build_embedding_index          # skip already-embedded items
    python -m app.experiments.RAG.e5.build_embedding_index --force  # recompute all embeddings
"""

import argparse
import random
import time
import uuid
from typing import List, Tuple

from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from app.db.session import SessionLocal
from app.models.dataset import DatasetItem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
EMBEDDING_DIM = 768
BATCH_SIZE = 32
TEST_SAMPLE_SIZE = 40
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# E5 model
# ---------------------------------------------------------------------------
print(f"Loading E5 model: {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_train_items(db) -> List[Tuple[uuid.UUID, str]]:
    """
    Returns the TRAIN split as a list of (item_id, text_adv) tuples.
    Mirrors the exact logic used in the OpenAI version.
    """
    all_items = (
        db.query(DatasetItem.item_id, DatasetItem.text_adv)
        .filter(DatasetItem.text_adv.isnot(None))
        .all()
    )

    random.seed(RANDOM_SEED)
    test_item_ids = set(
        item[0] for item in random.sample(
            all_items, min(TEST_SAMPLE_SIZE, len(all_items))
        )
    )
    return [item for item in all_items if item[0] not in test_item_ids]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generates E5 embeddings. Prepend 'passage: ' as these are stored embeddings.
    """
    prefixed_texts = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True)
    return embeddings.tolist()


def upsert_embeddings(
    db,
    batch: List[Tuple[uuid.UUID, str, List[float]]],
) -> None:
    """
    Upserts a batch of embeddings into public.dataset_item_embeddings_e5_768.
    """
    sql = text("""
        INSERT INTO public.dataset_item_embeddings_e5_768 (item_id, text_adv, embedding)
        VALUES (:item_id, :text_adv, :embedding)
        ON CONFLICT (item_id) DO UPDATE SET
            text_adv  = EXCLUDED.text_adv,
            embedding = EXCLUDED.embedding
    """)

    for item_id, text_adv, embedding in batch:
        embedding_str = "[" + ",".join(str(f) for f in embedding) + "]"
        db.execute(sql, {
            "item_id":   str(item_id),
            "text_adv":  text_adv,
            "embedding": embedding_str,
        })


def _already_embedded_ids(db) -> set:
    rows = db.execute(
        text("SELECT item_id FROM public.dataset_item_embeddings_e5_768")
    ).fetchall()
    return {str(row[0]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill public.dataset_item_embeddings_e5_768 with E5 embeddings."
    )
    parser.add_argument("--force", action="store_true", help="Re-embed all")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        train_items = get_train_items(db)
        total = len(train_items)
        print(f"Total train items: {total}")

        if args.force:
            items_to_process = train_items
        else:
            existing_ids = _already_embedded_ids(db)
            items_to_process = [
                item for item in train_items if str(item[0]) not in existing_ids
            ]
            print(f"Skipped: {total - len(items_to_process)}")

        print(f"Processing {len(items_to_process)} items...")

        for i in range(0, len(items_to_process), BATCH_SIZE):
            batch_items = items_to_process[i : i + BATCH_SIZE]
            texts = [item[1] for item in batch_items]
            embeddings = embed_texts(texts)

            upsert_data = []
            for (item_id, text_adv), emb in zip(batch_items, embeddings):
                upsert_data.append((item_id, text_adv, emb))

            upsert_embeddings(db, upsert_data)
            db.commit()
            print(f"Batch {i//BATCH_SIZE + 1} done.")

        # Validation
        row_count = db.execute(
            text("SELECT COUNT(*) FROM public.dataset_item_embeddings_e5_768")
        ).scalar()
        print(f"Total rows in DB: {row_count}")

    except Exception as exc:
        db.rollback()
        print(f"Error: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
