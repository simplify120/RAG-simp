"""
build_embedding_index.py
------------------------
Backfills public.dataset_item_embeddings_openai_1536 with OpenAI embeddings
for the TRAIN split of dataset_items (all items except the 40 test items
selected with random.seed(42)).

Usage:
    python -m app.experiments.RAG.openai.build_embedding_index          # skip already-embedded items
    python -m app.experiments.RAG.openai.build_embedding_index --force  # recompute all embeddings
"""

import argparse
import random
import time
import uuid
from typing import List, Tuple

from sqlalchemy import text
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.dataset import DatasetItem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 20
MAX_RETRIES = 3
BACKOFF_BASE = 2.0   # seconds; retry waits 2, 4, 8 s
TEST_SAMPLE_SIZE = 40
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
_api_key = settings.OPENAI_API_KEY
if not _api_key:
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Add it to your .env file or export it as an environment variable."
    )
openai_client = OpenAI(api_key=_api_key)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_train_items(db) -> List[Tuple[uuid.UUID, str]]:
    """
    Returns the TRAIN split as a list of (item_id, text_adv) tuples.

    Mirrors the exact logic used in the existing runners:
      1. Query all items with non-null text_adv using .all() (no ORDER BY).
      2. Set random.seed(42).
      3. Select 40 test items via random.sample.
      4. Train items are everything else.
    """
    all_items: List[Tuple[uuid.UUID, str]] = (
        db.query(DatasetItem.item_id, DatasetItem.text_adv)
        .filter(DatasetItem.text_adv.isnot(None))
        .all()
    )

    random.seed(RANDOM_SEED)
    test_items = set(
        item[0] for item in random.sample(all_items, min(TEST_SAMPLE_SIZE, len(all_items)))
    )

    train_items = [item for item in all_items if item[0] not in test_items]
    return train_items


def embed_text_adv(text_adv: str) -> List[float]:
    """
    Calls the OpenAI Embeddings API and returns exactly 1536 floats.
    Retries up to MAX_RETRIES times with exponential backoff on transient errors.
    """
    transient_errors = (RateLimitError, APITimeoutError, APIConnectionError)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text_adv,
            )
            embedding = response.data[0].embedding
            if len(embedding) != EMBEDDING_DIM:
                raise ValueError(
                    f"Expected {EMBEDDING_DIM} dimensions, got {len(embedding)}"
                )
            return embedding

        except transient_errors as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE ** attempt
            print(
                f"    [retry {attempt}/{MAX_RETRIES}] Transient error: {exc}. "
                f"Waiting {wait:.0f}s..."
            )
            time.sleep(wait)


def upsert_embedding(
    db,
    item_id: uuid.UUID,
    text_adv: str,
    embedding: List[float],
) -> None:
    """
    Upserts a single embedding row into public.dataset_item_embeddings_openai_1536.
    The embedding list is converted to the '[f1,f2,...]' string that pgvector accepts.
    Does NOT commit — caller is responsible for batched commits.
    """
    # pgvector accepts a string representation of the vector
    embedding_str = "[" + ",".join(str(f) for f in embedding) + "]"

    sql = text("""
        INSERT INTO public.dataset_item_embeddings_openai_1536 (item_id, text_adv, embedding)
        VALUES (:item_id, :text_adv, :embedding)
        ON CONFLICT (item_id) DO UPDATE SET
            text_adv  = EXCLUDED.text_adv,
            embedding = EXCLUDED.embedding
    """)

    db.execute(sql, {
        "item_id":   str(item_id),
        "text_adv":  text_adv,
        "embedding": embedding_str,
    })


def _already_embedded_ids(db) -> set:
    """Returns the set of item_ids already present in the embeddings table."""
    rows = db.execute(
        text("SELECT item_id FROM public.dataset_item_embeddings_openai_1536")
    ).fetchall()
    return {str(row[0]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill public.dataset_item_embeddings_openai_1536 with OpenAI embeddings."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed items that are already in the embeddings table.",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        # ------------------------------------------------------------------ #
        # 1. Determine train items
        # ------------------------------------------------------------------ #
        train_items = get_train_items(db)
        total = len(train_items)
        print(f"\n{'='*60}")
        print(f"Total train items: {total}")

        # ------------------------------------------------------------------ #
        # 2. Optionally skip already-embedded items
        # ------------------------------------------------------------------ #
        if args.force:
            items_to_process = train_items
            print("--force flag set: recomputing all embeddings.")
        else:
            existing_ids = _already_embedded_ids(db)
            items_to_process = [
                item for item in train_items if str(item[0]) not in existing_ids
            ]
            skipped_count = total - len(items_to_process)
            print(f"Already embedded: {skipped_count} items (skipping).")
            print(f"Items to embed:   {len(items_to_process)}")

        # ------------------------------------------------------------------ #
        # 3. Embed & upsert in batches
        # ------------------------------------------------------------------ #
        inserted = 0
        failed = 0
        skipped = total - len(items_to_process) if not args.force else 0

        print(f"{'='*60}\n")

        for batch_start in range(0, len(items_to_process), BATCH_SIZE):
            batch = items_to_process[batch_start: batch_start + BATCH_SIZE]

            for item_id, text_adv in batch:
                try:
                    embedding = embed_text_adv(text_adv)
                    upsert_embedding(db, item_id, text_adv, embedding)
                    inserted += 1
                except Exception as exc:
                    print(f"  [FAILED] item_id={item_id}: {exc}")
                    failed += 1

            db.commit()

            batch_end = min(batch_start + BATCH_SIZE, len(items_to_process))
            print(
                f"Batch done: items {batch_start + 1}–{batch_end} "
                f"| inserted/updated so far: {inserted} | failed: {failed}"
            )

        # ------------------------------------------------------------------ #
        # 4. Validation summary
        # ------------------------------------------------------------------ #
        row_count = db.execute(
            text("SELECT COUNT(*) FROM public.dataset_item_embeddings_openai_1536")
        ).scalar()

        # Check dimension of one stored embedding
        sample_dim = None
        sample_row = db.execute(
            text(
                "SELECT array_length(embedding::real[], 1) "
                "FROM public.dataset_item_embeddings_openai_1536 LIMIT 1"
            )
        ).scalar()
        sample_dim = sample_row

        print("VALIDATION SUMMARY")
        print(f"  Total train items:              {total}")
        print(f"  Inserted / updated:             {inserted}")
        print(f"  Skipped (already embedded):     {skipped}")
        print(f"  Failed:                         {failed}")
        print(f"  Rows in embeddings table:       {row_count}")
        print(f"  Embedding dimension (sample):   {sample_dim}")

        if sample_dim and sample_dim != EMBEDDING_DIM:
            print(
                f"WARNING: Expected dimension {EMBEDDING_DIM}, "
                f"but stored dimension is {sample_dim}!"
            )
        elif sample_dim == EMBEDDING_DIM:
            print("✓ Embedding dimension check PASSED (1536).")

    except Exception as exc:
        db.rollback()
        print(f"\nFatal error — rolled back: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
