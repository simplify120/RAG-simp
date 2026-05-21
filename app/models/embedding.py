from sqlalchemy import Column, String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from pgvector.sqlalchemy import Vector

from app.db.base import Base


class DatasetItemEmbeddingOpenAI(Base):
    """
    ORM model for public.dataset_item_embeddings_openai_1536.

    Stores 1536-dimensional OpenAI embeddings (text-embedding-3-small)
    for each dataset item, used by the RAG pipeline.
    """

    __tablename__ = "dataset_item_embeddings_openai_1536"
    __table_args__ = {"schema": "public"}

    embedding_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    item_id = Column(
        UUID(as_uuid=True),
        ForeignKey("dataset_items.item_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    text_adv = Column(String, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    # Relationship back to the source item
    dataset_item = relationship("DatasetItem", backref="embedding_openai")


class DatasetItemEmbeddingOpenAITestSet(Base):
    """
    ORM model for public.dataset_item_embeddings_openai_1536_test_set.

    Stores 1536-dimensional OpenAI embeddings for the 40 test-set items.
    """

    __tablename__ = "dataset_item_embeddings_openai_1536_test_set"
    __table_args__ = {"schema": "public"}

    embedding_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    item_id = Column(
        UUID(as_uuid=True),
        ForeignKey("dataset_items.item_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    text_adv = Column(String, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    dataset_item = relationship("DatasetItem", backref="embedding_openai_test_set")


class DatasetItemEmbeddingE5(Base):
    """
    ORM model for public.dataset_item_embeddings_e5_768.

    Stores 768-dimensional E5 embeddings for each dataset item.
    """

    __tablename__ = "dataset_item_embeddings_e5_768"
    __table_args__ = {"schema": "public"}

    embedding_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    item_id = Column(
        UUID(as_uuid=True),
        ForeignKey("dataset_items.item_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    text_adv = Column(String, nullable=False)
    embedding = Column(Vector(768), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    dataset_item = relationship("DatasetItem", backref="embedding_e5")


class DatasetItemEmbeddingE5TestSet(Base):
    """
    ORM model for public.dataset_item_embeddings_e5_768_test_set.

    Stores 768-dimensional E5 embeddings for the test-set items.
    """

    __tablename__ = "dataset_item_embeddings_e5_768_test_set"
    __table_args__ = {"schema": "public"}

    embedding_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    item_id = Column(
        UUID(as_uuid=True),
        ForeignKey("dataset_items.item_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    text_adv = Column(String, nullable=False)
    embedding = Column(Vector(768), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    dataset_item = relationship("DatasetItem", backref="embedding_e5_test_set")


class DatasetItemEmbeddingBGE(Base):
    """
    ORM model for public.dataset_item_embeddings_bge_768.

    Stores 768-dimensional BGE embeddings (BAAI/bge-base-en-v1.5) for train items.
    """

    __tablename__ = "dataset_item_embeddings_bge_768"
    __table_args__ = {"schema": "public"}

    embedding_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    item_id = Column(
        UUID(as_uuid=True),
        ForeignKey("dataset_items.item_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    text_adv = Column(String, nullable=False)
    embedding = Column(Vector(768), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    dataset_item = relationship("DatasetItem", backref="embedding_bge")


class DatasetItemEmbeddingBGETestSet(Base):
    """
    ORM model for public.dataset_item_embeddings_bge_768_test_set.

    Stores 768-dimensional BGE embeddings (BAAI/bge-base-en-v1.5) for test-set items.
    """

    __tablename__ = "dataset_item_embeddings_bge_768_test_set"
    __table_args__ = {"schema": "public"}

    embedding_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    item_id = Column(
        UUID(as_uuid=True),
        ForeignKey("dataset_items.item_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    text_adv = Column(String, nullable=False)
    embedding = Column(Vector(768), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    dataset_item = relationship("DatasetItem", backref="embedding_bge_test_set")
