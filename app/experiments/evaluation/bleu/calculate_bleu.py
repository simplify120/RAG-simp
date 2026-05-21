"""
Calculate BLEU score for PromptResults and store in Evaluation table.
BLEU compares the model's output_text to the reference text_ele.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from sqlalchemy import or_
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from app.db.session import SessionLocal
from app.models.prompt import PromptResult
from app.models.dataset import DatasetItem
from app.models.evaluation import Evaluation


def calculate_bleu(description=None, force_recalculate=False, model_name=None):
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
                    Evaluation.bleu.is_(None),
                )
            )
        if description is not None:
            query = query.filter(PromptResult.description == description)
        if model_name is not None:
            query = query.filter(PromptResult.model_name == model_name)
        results = query.all()
        
        # Calculate BLEU for each individual PromptResult
        processed = 0
        smoothing = SmoothingFunction().method1  # Use smoothing to handle edge cases
        
        for result_id, prompt_version_id, output_text, text_ele in results:
            processed += 1
            
            if processed % 10 == 0:
                print(f"Processing {processed}/{len(results)}...")
            
            bleu_score = None
            
            # Calculate BLEU score
            # BLEU compares candidate (output_text) to reference (text_ele)
            try:
                # Tokenize the texts (split into words)
                candidate = output_text.split()
                reference = text_ele.split()
                
                # Calculate BLEU score with smoothing
                # sentence_bleu expects reference as a list of lists (multiple references possible)
                bleu_score = sentence_bleu(
                    [reference],  # Reference as list of lists
                    candidate,    # Candidate as list of tokens
                    smoothing_function=smoothing
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            # Check if Evaluation already exists for this result_id
            existing_eval = db.query(Evaluation).filter(
                Evaluation.result_id == result_id
            ).first()
            
            if existing_eval:
                existing_eval.bleu = bleu_score
            else:
                evaluation = Evaluation(
                    prompt_version_id=prompt_version_id,
                    result_id=result_id,
                    bleu=bleu_score
                )
                db.add(evaluation)
        
        db.commit()
        print(f"\n✓ Successfully calculated and stored BLEU scores:")
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
    parser = argparse.ArgumentParser(description="Calculate BLEU scores for PromptResults")
    parser.add_argument("--description", type=str, default=None, help="Only process results with this description")
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Recompute BLEU even when already stored on Evaluation",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Filter by PromptResult.model_name")
    args = parser.parse_args()
    calculate_bleu(
        description=args.description,
        force_recalculate=args.force_recalculate,
        model_name=args.model_name,
    )

