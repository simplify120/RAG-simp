"""
Calculate Perplexity scores for PromptResults and store in Evaluation table.


Perplexity measures fluency/grammatical correctness by computing how "surprised"
a language model is by the text. Lower perplexity = more fluent/natural text.

We compute perplexity ONLY on the generated output_text (not input) because:
- We want to evaluate the quality of the LLM's simplified output
- Input text (advanced) is already known to be fluent
- Perplexity measures if the generated text is grammatically natural

We use a fixed pretrained model (distilgpt2) as evaluator:
- No training, no fine-tuning, no LLM API calls
- True perplexity based on token likelihood under the model
- Processes texts individually with proper error handling
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import torch
from sqlalchemy import or_
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.db.session import SessionLocal
from app.models.prompt import PromptResult
from app.models.evaluation import Evaluation


def calculate_perplexity_single(text, model, tokenizer, device, max_length=1024):
    """
    Calculate perplexity for a single text.
    
    Args:
        text: Input text string
        model: Pretrained language model
        tokenizer: Corresponding tokenizer
        device: torch device (cpu/cuda)
        max_length: Maximum sequence length (default 1024 for distilgpt2)
    
    Returns:
        Perplexity score (float) or None if calculation fails
    """
    try:
        # Tokenize with truncation
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        input_ids = encodings.input_ids.to(device)
        
        # Calculate perplexity
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    except Exception as e:
        print(f"  Error calculating perplexity: {str(e)[:100]}")
        return None


def calculate_perplexity(description=None, force_recalculate=False, model_name=None):
    db = SessionLocal()
    
    try:
        # PromptResults needing perplexity (skip rows that already have it stored)
        query = db.query(
            PromptResult.result_id,
            PromptResult.prompt_version_id,
            PromptResult.output_text
        ).outerjoin(
            Evaluation, Evaluation.result_id == PromptResult.result_id
        ).filter(
            PromptResult.output_text.isnot(None),
            PromptResult.output_text != "",
        )
        if not force_recalculate:
            query = query.filter(
                or_(
                    Evaluation.result_id.is_(None),
                    Evaluation.perplexity.is_(None),
                )
            )
        if description is not None:
            query = query.filter(PromptResult.description == description)
        if model_name is not None:
            query = query.filter(PromptResult.model_name == model_name)
        results = query.all()
        
        if not results:
            print("No PromptResults need perplexity (none matched filters, or all already have perplexity).")
            return

        print("Loading distilgpt2 model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        
        # Set padding token (GPT2 doesn't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        print(f"Model loaded on {device}")
        
        print(f"Found {len(results)} PromptResult rows to process")
        print("Computing perplexities (this may take a few minutes)...\n")
        
        # Process each text individually
        processed = 0
        successful = 0
        failed = 0
        perplexity_scores = []
        
        for result_id, prompt_version_id, output_text in results:
            processed += 1
            
            if processed % 10 == 0:
                print(f"Processing {processed}/{len(results)}...")
            
            # Calculate perplexity for this text
            perplexity_score = calculate_perplexity_single(
                output_text,
                model,
                tokenizer,
                device,
                max_length=1024
            )
            
            if perplexity_score is not None:
                perplexity_scores.append(perplexity_score)
                successful += 1
            else:
                failed += 1
            
            # Check if Evaluation already exists for this result_id
            existing_eval = db.query(Evaluation).filter(
                Evaluation.result_id == result_id
            ).first()
            
            if existing_eval:
                existing_eval.perplexity = perplexity_score
            else:
                evaluation = Evaluation(
                    prompt_version_id=prompt_version_id,
                    result_id=result_id,
                    perplexity=perplexity_score
                )
                db.add(evaluation)
        
        db.commit()
        
        # Calculate mean perplexity from successful scores
        mean_perplexity = sum(perplexity_scores) / len(perplexity_scores) if perplexity_scores else 0
        
        print(f"\n✓ Successfully calculated and stored Perplexity scores:")
        print(f"  - Total processed: {processed} results")
        print(f"  - Successful: {successful} results")
        print(f"  - Failed: {failed} results")
        print(f"  - Mean perplexity: {mean_perplexity:.2f}")
        
    except Exception as e:
        db.rollback()
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Perplexity for PromptResults")
    parser.add_argument("--description", type=str, default=None, help="Only process results with this description")
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Recompute perplexity even when already stored on Evaluation",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Filter by PromptResult.model_name")
    args = parser.parse_args()
    calculate_perplexity(
        description=args.description,
        force_recalculate=args.force_recalculate,
        model_name=args.model_name,
    )
