"""
RAG pipeline (CV, BGE embeddings): evaluate one complement fold at a time.

Usage (from repo root):

    python -m app.experiments.RAG.bge.run_rag_pipeline_cv \\
        --model claude-haiku-4-5 --description "step 3 - RAG cv top k=3" --top-k 3 --fold 0

    python -m app.experiments.RAG.bge.run_rag_pipeline_cv \\
        --model sonar --description "step 2 - RAG top k=3" --top-k 3 --all-folds
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from app.db.session import SessionLocal
from app.models.dataset import DatasetItem
from app.models.prompt import Prompt, PromptVersion, PromptResult
from app.experiments.RAG.cv_pacing import (
    CV_DEFAULT_SLEEP_SECONDS,
    CV_FOLD_GAP_MULTIPLIER,
    generic_retry_backoff_seconds,
    rate_limit_backoff_seconds,
)
from app.experiments.RAG.bge.retrieval import retrieve_top_k_cv
from app.experiments.llm_comparison.model_registry import call_llm, get_model_display_name, MODEL_CONFIG

RAG_EXAMPLES_HEADER = "Here are some similar examples to the following text:"
_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "splits" / "rag_cv_v1.json"


def _load_cv_manifest() -> dict:
    if not _MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"CV manifest not found: {_MANIFEST_PATH}. "
            "Run: python -m app.experiments.RAG.splits.generate_cv_splits"
        )
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def build_rag_prompt(
    original_template: str,
    retrieved_examples: List[Tuple[str, str]],
    input_text: str,
) -> str:
    template_with_input = original_template.replace("{INPUT_TEXT}", input_text)
    if not retrieved_examples:
        return template_with_input

    lines = [template_with_input, "", RAG_EXAMPLES_HEADER]
    for i, (text_adv, text_ele) in enumerate(retrieved_examples, 1):
        lines.append(f"Example {i} - Advanced: {text_adv} -> Simplified: {text_ele}")

    return "\n".join(lines)


def load_active_prompts(db, strategies: List[str]) -> List[Tuple[str, object]]:
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


def _fold_query_items(
    db,
    manifest: dict,
    fold: int,
) -> Tuple[List[Tuple[UUID, str]], Set[UUID]]:
    fold_key = str(fold)
    id_strings: List[str] = manifest["complement_folds"][fold_key]
    fold_ids_ordered = [UUID(s) for s in id_strings]
    excluded_ids: Set[UUID] = set(fold_ids_ordered)

    rows = (
        db.query(DatasetItem.item_id, DatasetItem.text_adv)
        .filter(DatasetItem.item_id.in_(fold_ids_ordered))
        .all()
    )
    by_id: Dict[UUID, str] = {r[0]: r[1] for r in rows}
    missing = [i for i in fold_ids_ordered if i not in by_id]
    if missing:
        raise ValueError(f"Missing dataset_items for fold {fold}: {missing[:5]}...")

    ordered = [(i, by_id[i]) for i in fold_ids_ordered]
    return ordered, excluded_ids


async def run_pipeline(
    model: str,
    description: str,
    top_k: int,
    strategies: List[str],
    fold: int,
    limit: Optional[int],
    sleep_seconds: float = CV_DEFAULT_SLEEP_SECONDS,
) -> None:
    manifest = _load_cv_manifest()
    db = SessionLocal()
    try:
        prompts = load_active_prompts(db, strategies)
        query_items, excluded_ids = _fold_query_items(db, manifest, fold)
        if limit:
            query_items = query_items[:limit]

        description_out = f"{description}"
        model_display = get_model_display_name(model)
        total = len(query_items) * len(prompts)
        done = 0

        print(
            f"BGE RAG CV pipeline: model={model}, description={description_out}, "
            f"top_k={top_k}, fold={fold}"
        )
        print(f"Strategies: {strategies}, Query items: {len(query_items)}")
        strategy_pause = max(4.0, sleep_seconds * 0.5)
        print(f"Sleep: {sleep_seconds}s between items, ~{strategy_pause:.1f}s between strategies")
        print(f"Total runs: {total}\n")
        for idx, (item_id, text_adv) in enumerate(query_items, 1):
            print(f"Processing item {idx}/{len(query_items)} (ID: {item_id})...")

            if idx > 1:
                await asyncio.sleep(sleep_seconds)

            retrieved = retrieve_top_k_cv(item_id, excluded_ids, top_k, db)
            if not retrieved:
                print(f"  Warning: No retrieved examples for item {item_id}")

            max_retries = 12

            for si, (strategy_name, prompt_version) in enumerate(prompts):
                template = prompt_version.template_text
                full_prompt = build_rag_prompt(template, retrieved, text_adv)

                output_text = None
                for attempt in range(max_retries):
                    try:
                        output_text = await call_llm(model, full_prompt)
                        break
                    except Exception as e:
                        error_str = str(e).lower()
                        if (
                            "429" in error_str
                            or "rate limit" in error_str
                            or "quota" in error_str
                            or "resource exhausted" in error_str
                            or "resource_exhausted" in error_str
                            or "ratelimiterror" in error_str
                            or "too many requests" in error_str
                        ):
                            if attempt < max_retries - 1:
                                wait = rate_limit_backoff_seconds(attempt)
                                print(
                                    f"  Rate limit. Waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}..."
                                )
                                await asyncio.sleep(wait)
                                continue
                        if attempt < max_retries - 1:
                            wait = generic_retry_backoff_seconds(attempt)
                            print(
                                f"  Error: {str(e)[:120]}. Retry {attempt + 1}/{max_retries} in {wait:.0f}s..."
                            )
                            await asyncio.sleep(wait)
                            continue
                        raise

                pr = PromptResult(
                    item_id=item_id,
                    prompt_version_id=prompt_version.prompt_version_id,
                    input_text=text_adv,
                    output_text=output_text,
                    model_name=model_display,
                    description=description_out,
                )
                db.add(pr)
                db.commit()
                done += 1
                print(f"  ✓ {strategy_name}")
                if si < len(prompts) - 1:
                    await asyncio.sleep(strategy_pause)

        print(f"\nDone. Stored {done} results.")

    except Exception as e:
        db.rollback()
        print(f"Pipeline error: {e}")
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BGE RAG pipeline (CV fold).")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--description", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    fold_group = parser.add_mutually_exclusive_group(required=True)
    fold_group.add_argument("--fold", type=int, choices=[0, 1, 2, 3])
    fold_group.add_argument(
        "--all-folds",
        action="store_true",
        help="Run folds 0, 1, 2, and 3 sequentially.",
    )
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["zeroshot", "structured", "constraint", "all"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=CV_DEFAULT_SLEEP_SECONDS,
        metavar="SEC",
        help="Seconds between query items after the first; also scales pause between strategies (default: 12).",
    )

    args = parser.parse_args()
    strategies = ["zeroshot", "structured", "constraint"] if args.strategy == "all" else [args.strategy]
    folds = [0, 1, 2, 3] if args.all_folds else [args.fold]

    async def _run_folds() -> None:
        for i, f in enumerate(folds):
            if i > 0:
                gap = args.sleep_seconds * CV_FOLD_GAP_MULTIPLIER
                print(f"\n--- Pausing {gap:.0f}s before fold {f} ---\n")
                await asyncio.sleep(gap)
            await run_pipeline(
                model=args.model,
                description=args.description,
                top_k=args.top_k,
                strategies=strategies,
                fold=f,
                limit=args.limit,
                sleep_seconds=args.sleep_seconds,
            )

    asyncio.run(_run_folds())


if __name__ == "__main__":
    main()
