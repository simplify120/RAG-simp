"""
run_rag_pipeline.py (E5 version)
---------------------------------
RAG pipeline for text simplification using E5 embeddings.
Retrieves top-k similar examples from train embeddings (E5), prepends them to
the active prompt templates, and runs the LLM for each test item.
"""

import argparse
import asyncio
from typing import List, Optional, Tuple

from app.db.session import SessionLocal
from app.models.prompt import Prompt, PromptVersion, PromptResult
from app.experiments.RAG.e5.build_embedding_index_test_set import get_test_items
from app.experiments.RAG.e5.retrieval import retrieve_top_k
from app.experiments.llm_comparison.model_registry import call_llm, get_model_display_name, MODEL_CONFIG

RAG_EXAMPLES_HEADER = "Here are some similar examples to the following text:"


def build_rag_prompt(
    original_template: str,
    retrieved_examples: List[Tuple[str, str]],
    input_text: str,
) -> str:
    """Build RAG-augmented prompt."""
    template_with_input = original_template.replace("{INPUT_TEXT}", input_text)
    if not retrieved_examples:
        return template_with_input

    lines = [template_with_input, "", RAG_EXAMPLES_HEADER]
    for i, (text_adv, text_ele) in enumerate(retrieved_examples, 1):
        lines.append(f"Example {i} - Advanced: {text_adv} -> Simplified: {text_ele}")

    return "\n".join(lines)


def load_active_prompts(db, strategies: List[str]) -> List[Tuple[str, object]]:
    """Load active prompt versions."""
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

        print(f"E5 RAG pipeline: model={model}, description={description}, top_k={top_k}")
        print(f"Strategies: {strategies}, Test items: {len(test_items)}")
        print(f"Total runs: {total}\n")

        for idx, (item_id, text_adv) in enumerate(test_items, 1):
            print(f"Processing item {idx}/{len(test_items)} (ID: {item_id})...")

            if idx > 1:
                await asyncio.sleep(0.5)  # Shorter sleep than OpenAI as we don't have local rate limits usually

            retrieved = retrieve_top_k(item_id, top_k, db)
            if not retrieved:
                print(f"  Warning: No retrieved examples for item {item_id}")

            for strategy_name, prompt_version in prompts:
                template = prompt_version.template_text
                full_prompt = build_rag_prompt(template, retrieved, text_adv)

                # Call LLM with retry
                max_retries = 3
                output_text = None
                for attempt in range(max_retries):
                    try:
                        output_text = await call_llm(model, full_prompt)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait = 2 * (2**attempt)
                            print(f"  Error: {e}. Retry {attempt+1} in {wait}s...")
                            await asyncio.sleep(wait)
                        else:
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
        print(f"Pipeline error: {e}")
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E5 RAG pipeline.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--description", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--strategy", default="all", choices=["zeroshot", "structured", "constraint", "all"])
    parser.add_argument("--limit", type=int, default=None)

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
