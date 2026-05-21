from sqlalchemy import Column, DateTime, Float, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.base import Base


class T5LargeTextSimplificationEvaluation(Base):
    __tablename__ = "t5_large_text_simplification_evaluation"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    item_id = Column(UUID(as_uuid=True), ForeignKey("dataset_items.item_id", ondelete="CASCADE"), nullable=False, index=True)
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
