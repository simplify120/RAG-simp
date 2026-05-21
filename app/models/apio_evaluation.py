from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.base import Base


class APioTextSimplificationEvaluation(Base):
    """APIO text simplification results per technique (zero-shot, few-shot, instruction_induction, optimized)."""

    __tablename__ = "apio_text_simplification_evaluation"
    __table_args__ = (UniqueConstraint("item_id", "technique", name="uq_apio_evaluation_item_technique"),)

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    item_id = Column(UUID(as_uuid=True), ForeignKey("dataset_items.item_id", ondelete="CASCADE"), nullable=False, index=True)
    technique = Column(String(32), nullable=False)  # zero_shot, few_shot, instruction_induction, optimized
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=True)
    sari = Column(Float, nullable=True)
    bertscore_f1 = Column(Float, nullable=True)
    fkgl_input = Column(Float, nullable=True)
    fkgl_output = Column(Float, nullable=True)
    delta_fkgl = Column(Float, nullable=True)
    fre_input = Column(Float, nullable=True)
    fre_output = Column(Float, nullable=True)
    fre_delta = Column(Float, nullable=True)
    bleu = Column(Float, nullable=True)
    perplexity = Column(Float, nullable=True)
    lens = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True, server_default=func.now())
