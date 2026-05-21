import random
import asyncio
import subprocess
from openai import OpenAI
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.prompt import Prompt, PromptVersion, PromptResult
from app.models.dataset import DatasetItem

# Set random seed for reproducibility
random.seed(42)

# Pull llama3.2 model if not already available
MODEL_NAME = "llama3.2"
print(f"Checking if {MODEL_NAME} is available...")
try:
    result = subprocess.run(
        ["ollama", "pull", MODEL_NAME],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"✓ {MODEL_NAME} is ready")
except subprocess.CalledProcessError as e:
    print(f"Error pulling model: {e}")
    print(f"Make sure Ollama is running and try: ollama pull {MODEL_NAME}")
    raise
except FileNotFoundError:
    print("Error: Ollama not found. Please install Ollama first.")
    raise

# Initialize Ollama client with OpenAI-compatible API
ollama = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # Ollama doesn't require a real API key
)

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

# Helper function to call Ollama
async def call_ollama(prompt_text):
    """Call Ollama API using OpenAI client"""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: ollama.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}]
        )
    )
    return response.choices[0].message.content

# Iterate and call LLM 3 times for each item (zero-shot, structured, constraint)
async def run_comparison():
    db_session = SessionLocal()
    
    try:
        print(f"Starting comparison with {len(DatasetItems)} items...")
        
        for idx, (item_id, text_adv) in enumerate(DatasetItems, 1):
            print(f"\nProcessing item {idx}/{len(DatasetItems)} (ID: {item_id})...")
            
            # Add small delay between requests to be safe
            if idx > 1:
                await asyncio.sleep(0.5)
            
            try:
                # Create the full prompts by replacing placeholder
                zeroshot_full_prompt = InstructionZeroShotPrompt.replace("{INPUT_TEXT}", text_adv)
                structured_full_prompt = InstructionStructuredPrompt.replace("{INPUT_TEXT}", text_adv)
                constraint_full_prompt = InstructionConstraintPrompt.replace("{INPUT_TEXT}", text_adv)
                
                # Run all three requests in parallel
                output_zeroshot_text, output_structured_text, output_constraint_text = await asyncio.gather(
                    call_ollama(zeroshot_full_prompt),
                    call_ollama(structured_full_prompt),
                    call_ollama(constraint_full_prompt)
                )
                
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
                        model_name=MODEL_NAME
                    )
                    
                    db_session.add(prompt_result)
                    db_session.commit()
                    print(f"  ✓ Saved {key} result")
                    
            except Exception as e:
                print(f"Error processing item {item_id}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next item instead of failing completely
                continue
                    
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
