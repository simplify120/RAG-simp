"""
retrieval.py (E5 version)
--------------------------
Retrieves top-k similar items from the train embeddings (dataset_item_embeddings_e5_768)
using cosine similarity. Uses precomputed embeddings from dataset_item_embeddings_e5_768_test_set.
"""

from __future__ import annotations

import uuid
from typing import List, Set, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.embedding import DatasetItemEmbeddingE5, DatasetItemEmbeddingE5TestSet
from app.models.dataset import DatasetItem


def _cosine_distance_vec(query: np.ndarray, candidate: np.ndarray) -> float:
    na = np.linalg.norm(query)
    nb = np.linalg.norm(candidate)
    if na == 0.0 or nb == 0.0:
        return 2.0
    cos_sim = float(np.dot(query, candidate) / (na * nb))
    return 1.0 - cos_sim


def _get_query_embedding(item_id: uuid.UUID, db: Session) -> List[float]:
    """
    Fetch the precomputed embedding for a test item from dataset_item_embeddings_e5_768_test_set.
    """
    row = (
        db.query(DatasetItemEmbeddingE5TestSet.embedding)
        .filter(DatasetItemEmbeddingE5TestSet.item_id == item_id)
        .first()
    )
    if not row:
        raise ValueError(
            f"No embedding found for item_id={item_id} in dataset_item_embeddings_e5_768_test_set. "
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
    Retrieve top-k most similar items from the E5 train embeddings corpus.
    Uses the precomputed embedding from the test set table.
    """
    query_embedding = _get_query_embedding(item_id, db)

    stmt = (
        select(DatasetItemEmbeddingE5.text_adv, DatasetItem.text_ele)
        .join(
            DatasetItem,
            DatasetItemEmbeddingE5.item_id == DatasetItem.item_id,
        )
        .where(DatasetItem.text_ele.isnot(None))
        .order_by(
            DatasetItemEmbeddingE5.embedding.cosine_distance(query_embedding)
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
    if k <= 0:
        return []

    q_row = (
        db.query(DatasetItemEmbeddingE5.embedding)
        .filter(DatasetItemEmbeddingE5.item_id == item_id)
        .first()
    )
    if not q_row:
        raise ValueError(
            f"No embedding found for item_id={item_id} in dataset_item_embeddings_e5_768. "
            "Run build_embedding_index first."
        )
    q_emb = q_row[0]
    if hasattr(q_emb, "tolist"):
        q_emb = q_emb.tolist()
    query_vec = np.asarray(q_emb, dtype=np.float64)

    corpus_ids = excluded_ids | {item_id}

    main_stmt = (
        select(
            DatasetItemEmbeddingE5.item_id,
            DatasetItemEmbeddingE5.text_adv,
            DatasetItemEmbeddingE5.embedding,
            DatasetItem.text_ele,
        )
        .join(
            DatasetItem,
            DatasetItemEmbeddingE5.item_id == DatasetItem.item_id,
        )
        .where(
            DatasetItem.text_ele.isnot(None),
            DatasetItemEmbeddingE5.item_id.notin_(corpus_ids),
        )
    )
    test_stmt = (
        select(
            DatasetItemEmbeddingE5TestSet.item_id,
            DatasetItemEmbeddingE5TestSet.text_adv,
            DatasetItemEmbeddingE5TestSet.embedding,
            DatasetItem.text_ele,
        )
        .join(
            DatasetItem,
            DatasetItemEmbeddingE5TestSet.item_id == DatasetItem.item_id,
        )
        .where(
            DatasetItem.text_ele.isnot(None),
            DatasetItemEmbeddingE5TestSet.item_id.notin_(corpus_ids),
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
