"""
RAG pipeline for text simplification.

Retrieves top-k similar examples from train embeddings, prepends them to
the active prompt templates (unchanged), and runs the LLM for each test item.
Stores results in prompt_results with user-provided description.

Usage (from repo root; test set is 40 items from get_test_items — embeddings must exist):

    python -m app.experiments.RAG.openai.run_rag_pipeline --model claude-haiku-4-5 --description "step 2 - RAG top k=3" --top-k 3
    python -m app.experiments.RAG.openai.run_rag_pipeline --model openai --description "step 2 - RAG top k=3" --strategy zeroshot

Then evaluate:

    python -m app.experiments.evaluation.evaluate_run --description "step 2 - RAG top k=3" --model-name claude-haiku-4-5
"""

import argparse
import asyncio
from typing import List, Optional, Tuple

from app.db.session import SessionLocal
from app.models.prompt import Prompt, PromptVersion, PromptResult
from app.experiments.RAG.openai.build_embedding_index_test_set import get_test_items
from app.experiments.RAG.openai.retrieval import retrieve_top_k
from app.experiments.llm_comparison.model_registry import call_llm, get_model_display_name, MODEL_CONFIG

RAG_EXAMPLES_HEADER = "Here are some similar examples to the following text:"


def build_rag_prompt(
    original_template: str,
    retrieved_examples: List[Tuple[str, str]],
    input_text: str,
) -> str:
    """
    Build RAG-augmented prompt. Template + target text first, then examples below.

    Does NOT modify the original template. RAG acts as external augmentation only.
    """
    template_with_input = original_template.replace("{INPUT_TEXT}", input_text)
    if not retrieved_examples:
        return template_with_input

    lines = [template_with_input, "", RAG_EXAMPLES_HEADER]
    for i, (text_adv, text_ele) in enumerate(retrieved_examples, 1):
        lines.append(f"Example {i} - Advanced: {text_adv} -> Simplified: {text_ele}")

    return "\n".join(lines)


def load_active_prompts(db, strategies: List[str]) -> List[Tuple[str, object]]:
    """
    Load active prompt versions for the given strategies.
    Returns list of (strategy_name, PromptVersion).
    """
    result = []
    for strategy in strategies:
        pv = (
            db.query(PromptVersion)
            .join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id)
            .filter(
                Prompt.strategy_type == strategy,
                PromptVersion.is_active == True,
            )
            .first()
        )
        if not pv:
            raise ValueError(f"No active {strategy} prompt found in database")
        result.append((strategy, pv))
    return result


async def run_pipeline(
    model: str,
    description: str,
    top_k: int,
    strategies: List[str],
    limit: Optional[int],
) -> None:
    db = SessionLocal()
    try:
        prompts = load_active_prompts(db, strategies)
        test_items = get_test_items(db)
        if limit:
            test_items = test_items[:limit]

        model_display = get_model_display_name(model)
        total = len(test_items) * len(prompts)
        done = 0

        print(f"RAG pipeline: model={model}, description={description}, top_k={top_k}")
        print(f"Strategies: {strategies}, Test items: {len(test_items)}")
        print(f"Total runs: {total}\n")

        for idx, (item_id, text_adv) in enumerate(test_items, 1):
            print(f"Processing item {idx}/{len(test_items)} (ID: {item_id})...")

            if idx > 1:
                await asyncio.sleep(1)

            retrieved = retrieve_top_k(item_id, top_k, db)
            if not retrieved:
                print(f"  Warning: No retrieved examples for item {item_id}, proceeding without RAG context")

            max_retries = 3
            retry_delay = 2

            for strategy_name, prompt_version in prompts:
                template = prompt_version.template_text
                full_prompt = build_rag_prompt(template, retrieved, text_adv)

                for attempt in range(max_retries):
                    try:
                        output_text = await call_llm(model, full_prompt)
                        break
                    except Exception as e:
                        error_str = str(e).lower()
                        if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                            if attempt < max_retries - 1:
                                wait = retry_delay * (2**attempt)
                                print(f"  Rate limit. Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                                await asyncio.sleep(wait)
                                continue
                        if attempt < max_retries - 1:
                            wait = retry_delay * (2**attempt)
                            print(f"  Error: {str(e)[:80]}. Retry {attempt + 1}/{max_retries} in {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        raise

                pr = PromptResult(
                    item_id=item_id,
                    prompt_version_id=prompt_version.prompt_version_id,
                    input_text=text_adv,
                    output_text=output_text,
                    model_name=model_display,
                    description=description,
                )
                db.add(pr)
                db.commit()
                done += 1
                print(f"  ✓ {strategy_name}")

        print(f"\nDone. Stored {done} results.")

    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG pipeline for text simplification.",
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--description", required=True, help="Description for stored results")
    parser.add_argument("--top-k", type=int, default=3, help="Number of similar chunks to retrieve")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["zeroshot", "structured", "constraint", "all"],
        help="Prompt strategy (or 'all' for all three)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test items (for testing)")

    args = parser.parse_args()

    strategies = ["zeroshot", "structured", "constraint"] if args.strategy == "all" else [args.strategy]

    asyncio.run(
        run_pipeline(
            model=args.model,
            description=args.description,
            top_k=args.top_k,
            strategies=strategies,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
