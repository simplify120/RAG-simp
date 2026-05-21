from sqlalchemy import Column, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.db.base import Base


class Evaluation(Base):
    __tablename__ = "evaluation"

    evaluation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_version_id = Column(UUID(as_uuid=True), ForeignKey("prompt_versions.prompt_version_id", ondelete="CASCADE"), nullable=False)
    result_id = Column(UUID(as_uuid=True), ForeignKey("prompt_results.result_id", ondelete="CASCADE"), nullable=False)
    sari = Column(Float, nullable=True)
    bertscore_f1 = Column(Float, nullable=True)
    bleu = Column(Float, nullable=True)
    perplexity = Column(Float, nullable=True)
    fkgl_input = Column(Float, nullable=True)
    fkgl_output = Column(Float, nullable=True)
    delta_fkgl = Column(Float, nullable=True)
    fre_input = Column(Float, nullable=True)
    fre_output = Column(Float, nullable=True)
    fre_delta = Column(Float, nullable=True)
    entity_additions_rate = Column(Float, nullable=True)
    number_mismatch_rate = Column(Float, nullable=True)
    lens = Column(Float, nullable=True)

    # Relationships
    prompt_version = relationship("PromptVersion", back_populates="evaluations")
    prompt_result = relationship("PromptResult", back_populates="evaluation")

