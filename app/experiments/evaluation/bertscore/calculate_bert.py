"""
Calculate BERTScore for PromptResults and store in Evaluation table.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from sqlalchemy import or_
from transformers.utils import logging as hf_logging
from bert_score import score
from app.db.session import SessionLocal
from app.models.prompt import PromptResult
from app.models.dataset import DatasetItem
from app.models.evaluation import Evaluation

# Suppress warnings about unused pooler layer weights
hf_logging.set_verbosity_error()


def calculate_bertscore(description=None, force_recalculate=False, model_name=None):
    db = SessionLocal()
    
    try:
        query = db.query(
            PromptResult.result_id,
            PromptResult.prompt_version_id,
            PromptResult.output_text,
            DatasetItem.text_ele
        ).join(
            DatasetItem, PromptResult.item_id == DatasetItem.item_id
        ).outerjoin(
            Evaluation, Evaluation.result_id == PromptResult.result_id
        ).filter(
            PromptResult.output_text.isnot(None),
            DatasetItem.text_ele.isnot(None),
        )
        if not force_recalculate:
            query = query.filter(
                or_(
                    Evaluation.result_id.is_(None),
                    Evaluation.bertscore_f1.is_(None),
                )
            )
        if description is not None:
            query = query.filter(PromptResult.description == description)
        if model_name is not None:
            query = query.filter(PromptResult.model_name == model_name)
        results = query.all()
        
        # Calculate BERTScore for each individual PromptResult
        processed = 0
        
        for result_id, prompt_version_id, output_text, text_ele in results:
            processed += 1
            
            if processed % 10 == 0:
                print(f"Processing {processed}/{len(results)}...")
            
            bertscore_f1 = None
            
            # Calculate BERTScore
            try:
                P, R, F1 = score(
                    cands=[output_text],
                    refs=[text_ele],
                    lang='en',
                    verbose=False
                )
                bertscore_f1 = F1.item()
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            # Check if Evaluation already exists for this result_id
            existing_eval = db.query(Evaluation).filter(
                Evaluation.result_id == result_id
            ).first()
            
            if existing_eval:
                existing_eval.bertscore_f1 = bertscore_f1
            else:
                evaluation = Evaluation(
                    prompt_version_id=prompt_version_id,
                    result_id=result_id,
                    bertscore_f1=bertscore_f1
                )
                db.add(evaluation)
        
        db.commit()
        print(f"\n✓ Successfully calculated and stored BERTScore:")
        print(f"  - Processed: {processed} results")
        
    except Exception as e:
        db.rollback()
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BERTScore for PromptResults")
    parser.add_argument("--description", type=str, default=None, help="Only process results with this description")
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Recompute BERTScore even when already stored on Evaluation",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Filter by PromptResult.model_name")
    args = parser.parse_args()
    calculate_bertscore(
        description=args.description,
        force_recalculate=args.force_recalculate,
        model_name=args.model_name,
    )
