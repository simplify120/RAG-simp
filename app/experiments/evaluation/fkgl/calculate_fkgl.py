"""
Calculate FKGL (Flesch-Kincaid Grade Level) for PromptResults and store in Evaluation table.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import textstat
from sqlalchemy import or_
from app.db.session import SessionLocal
from app.models.prompt import PromptResult
from app.models.dataset import DatasetItem
from app.models.evaluation import Evaluation


def calculate_fkgl(description=None, force_recalculate=False, model_name=None):
    db = SessionLocal()
    
    try:
        # Get all PromptResults with their corresponding DatasetItems
        query = db.query(
            PromptResult.result_id,
            PromptResult.prompt_version_id,
            PromptResult.input_text,
            PromptResult.output_text,
            DatasetItem.text_ele
        ).join(
            DatasetItem, PromptResult.item_id == DatasetItem.item_id
        ).outerjoin(
            Evaluation, Evaluation.result_id == PromptResult.result_id
        ).filter(
            PromptResult.output_text.isnot(None),
            PromptResult.input_text.isnot(None),
        )
        if not force_recalculate:
            query = query.filter(
                or_(
                    Evaluation.result_id.is_(None),
                    Evaluation.fkgl_output.is_(None),
                )
            )
        if description is not None:
            query = query.filter(PromptResult.description == description)
        if model_name is not None:
            query = query.filter(PromptResult.model_name == model_name)
        results = query.all()
        
        # Calculate FKGL for each individual PromptResult
        processed = 0
        
        for result_id, prompt_version_id, input_text, output_text, text_ele in results:
            processed += 1
            
            if processed % 10 == 0:
                print(f"Processing {processed}/{len(results)}...")
            
            fkgl_input = None
            fkgl_output = None
            delta_fkgl = None
            
            # Calculate FKGL for input text
            try:
                fkgl_input = textstat.flesch_kincaid_grade(input_text)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            # Calculate FKGL for output text
            try:
                fkgl_output = textstat.flesch_kincaid_grade(output_text)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            # Calculate FKGL delta (output FKGL - input FKGL)
            # Negative delta means the output is simpler (lower grade level) than input
            if fkgl_input is not None and fkgl_output is not None:
                delta_fkgl = fkgl_output - fkgl_input
            
            # Check if Evaluation already exists for this result_id
            existing_eval = db.query(Evaluation).filter(
                Evaluation.result_id == result_id
            ).first()
            
            if existing_eval:
                # Update existing evaluation
                existing_eval.fkgl_input = fkgl_input
                existing_eval.fkgl_output = fkgl_output
                existing_eval.delta_fkgl = delta_fkgl
            else:
                # Create new evaluation
                evaluation = Evaluation(
                    prompt_version_id=prompt_version_id,
                    result_id=result_id,
                    fkgl_input=fkgl_input,
                    fkgl_output=fkgl_output,
                    delta_fkgl=delta_fkgl
                )
                db.add(evaluation)
        
        # Commit all changes
        db.commit()
        print(f"\n✓ Successfully calculated and stored FKGL scores:")
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
    parser = argparse.ArgumentParser(description="Calculate FKGL for PromptResults")
    parser.add_argument("--description", type=str, default=None, help="Only process results with this description")
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Recompute FKGL even when already stored on Evaluation",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Filter by PromptResult.model_name")
    args = parser.parse_args()
    calculate_fkgl(
        description=args.description,
        force_recalculate=args.force_recalculate,
        model_name=args.model_name,
    )

