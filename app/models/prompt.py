from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base import Base


class Prompt(Base):
    __tablename__ = "prompts"

    prompt_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_type = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # Relationships
    versions = relationship("PromptVersion", back_populates="prompt", cascade="all, delete-orphan")


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    prompt_version_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_id = Column(UUID(as_uuid=True), ForeignKey("prompts.prompt_id", ondelete="CASCADE"), nullable=False)
    version = Column(String, nullable=False)
    template_text = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # Relationships
    prompt = relationship("Prompt", back_populates="versions")
    results = relationship("PromptResult", back_populates="prompt_version", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="prompt_version", cascade="all, delete-orphan")


class PromptResult(Base):
    __tablename__ = "prompt_results"

    result_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_id = Column(UUID(as_uuid=True), ForeignKey("dataset_items.item_id", ondelete="CASCADE"), nullable=False)
    prompt_version_id = Column(UUID(as_uuid=True), ForeignKey("prompt_versions.prompt_version_id", ondelete="CASCADE"), nullable=False)
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    model_name = Column(Text,nullable=True)
    description = Column(Text, nullable=True)

    # Relationships
    dataset_item = relationship("DatasetItem", back_populates="results")
    prompt_version = relationship("PromptVersion", back_populates="results")
    evaluation = relationship("Evaluation", back_populates="prompt_result", uselist=False)

