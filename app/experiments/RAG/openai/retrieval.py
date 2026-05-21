"""
RAG retrieval module.

Retrieves top-k similar items from the train embeddings (dataset_item_embeddings_openai_1536)
using cosine similarity. Uses precomputed embeddings from dataset_item_embeddings_openai_1536_test_set
for the query (no API calls).
"""

from __future__ import annotations

import uuid
from typing import List, Set, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.embedding import DatasetItemEmbeddingOpenAI, DatasetItemEmbeddingOpenAITestSet
from app.models.dataset import DatasetItem


def _cosine_distance_vec(query: np.ndarray, candidate: np.ndarray) -> float:
    """
    Cosine distance = 1 - cos_sim, matching pgvector's cosine_distance operator
    used by retrieve_top_k (ascending order = most similar first).
    """
    na = np.linalg.norm(query)
    nb = np.linalg.norm(candidate)
    if na == 0.0 or nb == 0.0:
        return 2.0
    cos_sim = float(np.dot(query, candidate) / (na * nb))
    return 1.0 - cos_sim


def _get_query_embedding(item_id: uuid.UUID, db: Session) -> List[float]:
    """
    Fetch the precomputed embedding for a test item from dataset_item_embeddings_openai_1536_test_set.
    """
    row = (
        db.query(DatasetItemEmbeddingOpenAITestSet.embedding)
        .filter(DatasetItemEmbeddingOpenAITestSet.item_id == item_id)
        .first()
    )
    if not row:
        raise ValueError(
            f"No embedding found for item_id={item_id} in dataset_item_embeddings_openai_1536_test_set. "
            "Run build_embedding_index_test_set first."
        )
    emb = row[0]
    if hasattr(emb, "tolist"):
        return emb.tolist()
    return list(emb)


def retrieve_top_k(
    item_id: uuid.UUID,
    k: int,
    db: Session,
) -> List[Tuple[str, str]]:
    """
    Retrieve top-k most similar items from the train embeddings corpus.

    Uses the precomputed embedding from dataset_item_embeddings_openai_1536_test_set (no API call).
    Searches dataset_item_embeddings_openai_1536 via cosine similarity.

    Args:
        item_id: Test item ID (embedding looked up from test set table).
        k: Number of similar items to retrieve.
        db: SQLAlchemy session.

    Returns:
        List of (text_adv, text_ele) tuples. Items without text_ele are excluded.
    """
    query_embedding = _get_query_embedding(item_id, db)

    stmt = (
        select(DatasetItemEmbeddingOpenAI.text_adv, DatasetItem.text_ele)
        .join(
            DatasetItem,
            DatasetItemEmbeddingOpenAI.item_id == DatasetItem.item_id,
        )
        .where(DatasetItem.text_ele.isnot(None))
        .order_by(
            DatasetItemEmbeddingOpenAI.embedding.cosine_distance(query_embedding)
        )
        .limit(k)
    )

    rows = db.execute(stmt).fetchall()
    return [(row[0], row[1]) for row in rows if row[1]]


def retrieve_top_k_cv(
    item_id: uuid.UUID,
    excluded_ids: Set[uuid.UUID],
    k: int,
    db: Session,
) -> List[Tuple[str, str]]:
    """
    CV retrieval: query embedding from the main (train) table; corpus = all rows in the
    main table NOT in ``excluded_ids`` (the held-out fold) plus all rows in the test_set
    table, merged and ranked by cosine distance (ascending), same metric as ``retrieve_top_k``.

    ``excluded_ids`` must contain every complement item_id in the current CV fold (typically
    37 or 38 ids). The query ``item_id`` should be one of those; it is also excluded from
    the corpus to avoid self-retrieval.
    """
    if k <= 0:
        return []

    q_row = (
        db.query(DatasetItemEmbeddingOpenAI.embedding)
        .filter(DatasetItemEmbeddingOpenAI.item_id == item_id)
        .first()
    )
    if not q_row:
        raise ValueError(
            f"No embedding found for item_id={item_id} in dataset_item_embeddings_openai_1536. "
            "CV queries use the complement (main) table — run build_embedding_index first."
        )
    q_emb = q_row[0]
    if hasattr(q_emb, "tolist"):
        q_emb = q_emb.tolist()
    query_vec = np.asarray(q_emb, dtype=np.float64)

    corpus_ids = excluded_ids | {item_id}

    main_stmt = (
        select(
            DatasetItemEmbeddingOpenAI.item_id,
            DatasetItemEmbeddingOpenAI.text_adv,
            DatasetItemEmbeddingOpenAI.embedding,
            DatasetItem.text_ele,
        )
        .join(
            DatasetItem,
            DatasetItemEmbeddingOpenAI.item_id == DatasetItem.item_id,
        )
        .where(
            DatasetItem.text_ele.isnot(None),
            DatasetItemEmbeddingOpenAI.item_id.notin_(corpus_ids),
        )
    )
    test_stmt = (
        select(
            DatasetItemEmbeddingOpenAITestSet.item_id,
            DatasetItemEmbeddingOpenAITestSet.text_adv,
            DatasetItemEmbeddingOpenAITestSet.embedding,
            DatasetItem.text_ele,
        )
        .join(
            DatasetItem,
            DatasetItemEmbeddingOpenAITestSet.item_id == DatasetItem.item_id,
        )
        .where(
            DatasetItem.text_ele.isnot(None),
            DatasetItemEmbeddingOpenAITestSet.item_id.notin_(corpus_ids),
        )
    )

    rows = list(db.execute(main_stmt).fetchall()) + list(db.execute(test_stmt).fetchall())

    scored: List[Tuple[float, str, Tuple[str, str]]] = []
    for cid, text_adv, emb, text_ele in rows:
        if not text_ele:
            continue
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        cand = np.asarray(emb, dtype=np.float64)
        dist = _cosine_distance_vec(query_vec, cand)
        scored.append((dist, str(cid), (text_adv, text_ele)))

    scored.sort(key=lambda t: (t[0], t[1]))
    return [pair for _, _, pair in scored[:k]]
