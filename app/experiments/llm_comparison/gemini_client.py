import random
import asyncio
import os
import time
from agents import Agent, Runner, trace
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.prompt import Prompt, PromptVersion, PromptResult
from app.models.dataset import DatasetItem

# Set random seed for reproducibility
random.seed(42)

# Set Gemini API key for agents library
if settings.GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
else:
    raise ValueError("GEMINI_API_KEY not set in environment")

# Query: prompt_id connects Prompt and PromptVersion
db = SessionLocal()

try:
    #Prompt Instructions - Get both template_text and prompt_version_id
    zero_shot_prompt = db.query(PromptVersion).join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id).filter(
        Prompt.strategy_type == "zeroshot", 
        PromptVersion.is_active == True
    ).first()

    structured_prompt = db.query(PromptVersion).join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id).filter(
        Prompt.strategy_type == "structured", 
        PromptVersion.is_active == True
    ).first()

    constraint_prompt = db.query(PromptVersion).join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id).filter(
        Prompt.strategy_type == "constraint", 
        PromptVersion.is_active == True
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

    # Get all items, then randomly select 40 with seed 42
    all_items = db.query(DatasetItem.item_id, DatasetItem.text_adv).filter(DatasetItem.text_adv.isnot(None)).all()
    DatasetItems = random.sample(all_items, min(40, len(all_items)))
finally:
    db.close()

# Create agents with instructions from database
# Using gemini-2.0-flash as it was referenced in earlier API calls
# The agents library uses litellm under the hood with litellm/ prefix format
ZeroAgent = Agent(
    name="ZeroAgent",
    instructions=InstructionZeroShotPrompt.replace("{INPUT_TEXT}", ""),
    model="litellm/gemini/gemini-2.0-flash"
)

StructuredAgent = Agent(
    name="StructuredAgent",
    instructions=InstructionStructuredPrompt.replace("{INPUT_TEXT}", ""),
    model="litellm/gemini/gemini-2.0-flash"
)

ConstraintAgent = Agent(
    name="ConstraintAgent",
    instructions=InstructionConstraintPrompt.replace("{INPUT_TEXT}", ""),
    model="litellm/gemini/gemini-2.0-flash",
)

# Iterate and call LLM 3 times for each item (zero-shot, structured, constraint)
async def run_comparison():
    db_session = SessionLocal()
    
    try:
        print(f"Starting comparison with {len(DatasetItems)} items...")
        
        for idx, (item_id, text_adv) in enumerate(DatasetItems, 1):
            print(f"\nProcessing item {idx}/{len(DatasetItems)} (ID: {item_id})...")
            
            # Add delay between requests to avoid rate limits
            if idx > 1:
                await asyncio.sleep(1)  # 1 second delay between items
            
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    with trace(f"LLM Comparison - Item {item_id}"):
                        # asyncio.gather guarantees results are returned in the same order as coroutines
                        result_zeroshot, result_structured, result_constraint = await asyncio.gather(
                            Runner.run(ZeroAgent, InstructionZeroShotPrompt.replace("{INPUT_TEXT}", text_adv)),
                            Runner.run(StructuredAgent, InstructionStructuredPrompt.replace("{INPUT_TEXT}", text_adv)),
                            Runner.run(ConstraintAgent, InstructionConstraintPrompt.replace("{INPUT_TEXT}", text_adv))
                        )
                    break  # Success, exit retry loop
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
                        # Not a rate limit error, re-raise immediately
                        raise
            
            # Extract text from Runner results (after successful retry loop)
            output_zeroshot_text = result_zeroshot.final_output if hasattr(result_zeroshot, 'final_output') else str(result_zeroshot)
            output_structured_text = result_structured.final_output if hasattr(result_structured, 'final_output') else str(result_structured)
            output_constraint_text = result_constraint.final_output if hasattr(result_constraint, 'final_output') else str(result_constraint)
            
            results = {
                "zero-shot": (output_zeroshot_text, zero_shot_prompt.prompt_version_id),
                "structured": (output_structured_text, structured_prompt.prompt_version_id),
                "constraint": (output_constraint_text, constraint_prompt.prompt_version_id)
            }
            
            for key, (output_text, prompt_version_id) in results.items():
                prompt_result = PromptResult(
                    item_id=item_id,
                    prompt_version_id=prompt_version_id,
                    input_text=text_adv,
                    output_text=output_text,
                    model_name="gemini-2.0-flash"
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

if __name__ == "__main__":
    asyncio.run(run_comparison())

