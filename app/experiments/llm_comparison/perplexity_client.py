import argparse
import random
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agents import Agent, Runner
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.prompt import Prompt, PromptVersion, PromptResult
from app.models.dataset import DatasetItem

# Stored on PromptResult rows; use the same string when running evaluations, e.g.:
#   python -m app.experiments.evaluation.evaluate_run --description "step 1 - simple prompt engineering"
RESULT_DESCRIPTION = "step 1 - simple prompt engineering"

PILOT_N = 40

# Set random seed for reproducibility (must match historical pilot run)
random.seed(42)

# Set Perplexity API key for agents library
if settings.PERPLEXITYAI_API_KEY:
    os.environ["PERPLEXITYAI_API_KEY"] = settings.PERPLEXITYAI_API_KEY
else:
    raise ValueError("PERPLEXITYAI_API_KEY not set in environment")

# Query: prompt_id connects Prompt and PromptVersion
db = SessionLocal()

try:
    # Prompt Instructions - Get both template_text and prompt_version_id (active versions only)
    zero_shot_prompt = db.query(PromptVersion).join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id).filter(
        Prompt.strategy_type == "zeroshot",
        PromptVersion.is_active == True,
    ).first()

    structured_prompt = db.query(PromptVersion).join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id).filter(
        Prompt.strategy_type == "structured",
        PromptVersion.is_active == True,
    ).first()

    constraint_prompt = db.query(PromptVersion).join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id).filter(
        Prompt.strategy_type == "constraint",
        PromptVersion.is_active == True,
    ).first()

    # Validate that all prompts exist
    if not zero_shot_prompt:
        raise ValueError("No active zeroshot prompt found in database")
    if not structured_prompt:
        raise ValueError("No active structured prompt found in database")
    if not constraint_prompt:
        raise ValueError("No active constraint prompt found in database")

    InstructionZeroShotPrompt = zero_shot_prompt.template_text
    InstructionStructuredPrompt = structured_prompt.template_text
    InstructionConstraintPrompt = constraint_prompt.template_text

    zero_shot_pv_id = zero_shot_prompt.prompt_version_id
    structured_pv_id = structured_prompt.prompt_version_id
    constraint_pv_id = constraint_prompt.prompt_version_id

    all_items = db.query(DatasetItem.item_id, DatasetItem.text_adv).filter(DatasetItem.text_adv.isnot(None)).all()
    pilot_items = random.sample(all_items, min(PILOT_N, len(all_items)))
    pilot_set = set(pilot_items)
    remaining_items = [row for row in all_items if row not in pilot_set]
finally:
    db.close()

# Create agents with instructions from database
# Using sonar (basic) - faster/cheaper than sonar-pro
# The agents library uses litellm under the hood with litellm/ prefix format
ZeroAgent = Agent(
    name="ZeroAgent",
    instructions=InstructionZeroShotPrompt.replace("{INPUT_TEXT}", ""),
    model="litellm/perplexity/sonar"
)

StructuredAgent = Agent(
    name="StructuredAgent",
    instructions=InstructionStructuredPrompt.replace("{INPUT_TEXT}", ""),
    model="litellm/perplexity/sonar"
)

ConstraintAgent = Agent(
    name="ConstraintAgent",
    instructions=InstructionConstraintPrompt.replace("{INPUT_TEXT}", ""),
    model="litellm/perplexity/sonar",
)

# Iterate and call LLM 3 times for each item (zero-shot, structured, constraint)
async def run_comparison(dataset_items):
    db_session = SessionLocal()

    try:
        print(f"Starting comparison with {len(dataset_items)} items (description={RESULT_DESCRIPTION!r})...")

        for idx, (item_id, text_adv) in enumerate(dataset_items, 1):
            print(f"\nProcessing item {idx}/{len(dataset_items)} (ID: {item_id})...")
            
            # Add delay between requests to avoid rate limits
            # Perplexity may need slightly longer delays due to web search capabilities
            if idx > 1:
                await asyncio.sleep(2)  # 2 second delay between items for Perplexity
            
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # asyncio.gather guarantees results are returned in the same order as coroutines
                    result_zeroshot, result_structured, result_constraint = await asyncio.gather(
                        Runner.run(ZeroAgent, InstructionZeroShotPrompt.replace("{INPUT_TEXT}", text_adv)),
                        Runner.run(StructuredAgent, InstructionStructuredPrompt.replace("{INPUT_TEXT}", text_adv)),
                        Runner.run(ConstraintAgent, InstructionConstraintPrompt.replace("{INPUT_TEXT}", text_adv))
                    )
                    break  # Success, exit retry loop
                except (asyncio.CancelledError, asyncio.TimeoutError) as e:
                    # Handle timeout/cancellation errors
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Request cancelled/timed out. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"Request cancelled/timed out after {max_retries} attempts. Skipping item {item_id}.")
                        raise
                except Exception as e:
                    error_str = str(e).lower()
                    # Check if it's a rate limit error
                    if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            print(f"Rate limit error after {max_retries} attempts. Skipping item {item_id}.")
                            raise
                    else:
                        # For other errors, log and retry
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"Error occurred: {str(e)[:100]}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            print(f"Error after {max_retries} attempts: {str(e)[:200]}")
                            raise
            
            # Extract text from Runner results (after successful retry loop)
            output_zeroshot_text = result_zeroshot.final_output if hasattr(result_zeroshot, 'final_output') else str(result_zeroshot)
            output_structured_text = result_structured.final_output if hasattr(result_structured, 'final_output') else str(result_structured)
            output_constraint_text = result_constraint.final_output if hasattr(result_constraint, 'final_output') else str(result_constraint)
            
            results = {
                "zero-shot": (output_zeroshot_text, zero_shot_pv_id),
                "structured": (output_structured_text, structured_pv_id),
                "constraint": (output_constraint_text, constraint_pv_id),
            }

            for key, (output_text, prompt_version_id) in results.items():
                prompt_result = PromptResult(
                    item_id=item_id,
                    prompt_version_id=prompt_version_id,
                    input_text=text_adv,
                    output_text=output_text,
                    model_name="sonar",
                    description=RESULT_DESCRIPTION,
                )
                
                db_session.add(prompt_result)
                db_session.commit()     
    except Exception as e:
        db_session.rollback()
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db_session.close()

def _select_items(subset: str):
    if subset == "pilot":
        return pilot_items
    if subset == "remaining":
        return remaining_items
    if subset == "all":
        return all_items
    raise ValueError(f"Unknown subset: {subset}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Perplexity Sonar on dataset items and store PromptResults."
    )
    parser.add_argument(
        "--subset",
        choices=("pilot", "remaining", "all"),
        default="remaining",
        help=(
            "pilot: same 40 items as the original seed-42 run; "
            "remaining: all other items with text_adv (the ~149 left after pilot); "
            "all: full dataset."
        ),
    )
    args = parser.parse_args()
    items = _select_items(args.subset)
    print(
        f"Subset={args.subset!r}: {len(items)} items "
        f"(total with text_adv={len(all_items)}, pilot={len(pilot_items)}, remaining={len(remaining_items)})"
    )
    asyncio.run(run_comparison(items))


if __name__ == "__main__":
    main()
