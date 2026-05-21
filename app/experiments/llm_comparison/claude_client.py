"""
Run Claude Haiku 4.5 on dataset items (zeroshot / structured / constraint) and store PromptResults.

Default --subset all runs every item with text_adv (~189). Active prompt versions only (is_active=True).

Evaluation (Claude rows only; avoids mixing with sonar under the same description):
  python -m app.experiments.evaluation.evaluate_run --description "step 1 - simple prompt engineering" --model-name claude-haiku-4-5

LiteLLM expects ANTHROPIC_API_KEY; this module sets it from CLAUDE_API_KEY in .env.
"""

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agents import Agent, Runner

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.dataset import DatasetItem
from app.models.prompt import Prompt, PromptResult, PromptVersion

RESULT_DESCRIPTION = "step 1 - simple prompt engineering"
PILOT_N = 40
# LiteLLM + Agents; see https://docs.litellm.ai/docs/providers/anthropic
CLAUDE_MODEL_LITELLM = "litellm/anthropic/claude-haiku-4-5-20251001"
MODEL_NAME_STORED = "claude-haiku-4-5"

# Throttle Anthropic: sequential strategies + delay between items (parallel gather caused 529 overload).
DEFAULT_INTER_ITEM_SLEEP_SEC = 3.0
DEFAULT_INTER_STRATEGY_SLEEP_SEC = 0.75
MAX_RUN_ATTEMPTS = 8

random.seed(42)

_api = settings.CLAUDE_API_KEY or os.getenv("CLAUDE_API_KEY")
if not _api:
    raise ValueError("CLAUDE_API_KEY not set in environment (.env)")
os.environ["ANTHROPIC_API_KEY"] = _api
os.environ.setdefault("CLAUDE_API_KEY", _api)

db = SessionLocal()
try:
    zero_shot_prompt = (
        db.query(PromptVersion)
        .join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id)
        .filter(Prompt.strategy_type == "zeroshot", PromptVersion.is_active == True)
        .first()
    )
    structured_prompt = (
        db.query(PromptVersion)
        .join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id)
        .filter(Prompt.strategy_type == "structured", PromptVersion.is_active == True)
        .first()
    )
    constraint_prompt = (
        db.query(PromptVersion)
        .join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id)
        .filter(Prompt.strategy_type == "constraint", PromptVersion.is_active == True)
        .first()
    )

    if not zero_shot_prompt or not structured_prompt or not constraint_prompt:
        raise ValueError("Missing active prompt version for zeroshot, structured, or constraint")

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

ZeroAgent = Agent(
    name="ZeroAgent",
    instructions=InstructionZeroShotPrompt.replace("{INPUT_TEXT}", ""),
    model=CLAUDE_MODEL_LITELLM,
)
StructuredAgent = Agent(
    name="StructuredAgent",
    instructions=InstructionStructuredPrompt.replace("{INPUT_TEXT}", ""),
    model=CLAUDE_MODEL_LITELLM,
)
ConstraintAgent = Agent(
    name="ConstraintAgent",
    instructions=InstructionConstraintPrompt.replace("{INPUT_TEXT}", ""),
    model=CLAUDE_MODEL_LITELLM,
)


def _backoff_seconds(attempt_index: int, error: BaseException) -> float:
    """Longer waits for Anthropic 529/overloaded; standard exponential otherwise."""
    s = str(error).lower()
    overloaded = "529" in s or "overloaded" in s
    if overloaded:
        return min(120.0, 5.0 * (2**attempt_index))
    if "429" in s or "rate limit" in s or "quota" in s:
        return min(90.0, 4.0 * (2**attempt_index))
    return min(60.0, 2.0 * (2**attempt_index))


async def _run_agent_with_retry(agent: Agent, user_message: str) -> object:
    for attempt in range(MAX_RUN_ATTEMPTS):
        try:
            return await Runner.run(agent, user_message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if attempt >= MAX_RUN_ATTEMPTS - 1:
                raise
            wait = _backoff_seconds(attempt, e)
            print(f"  Retry in {wait:.1f}s ({attempt + 2}/{MAX_RUN_ATTEMPTS}): {str(e)[:140]}")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


async def run_comparison(
    dataset_items,
    inter_item_sleep_sec: float = DEFAULT_INTER_ITEM_SLEEP_SEC,
    inter_strategy_sleep_sec: float = DEFAULT_INTER_STRATEGY_SLEEP_SEC,
):
    db_session = SessionLocal()
    try:
        print(
            f"Starting Claude Haiku 4.5 run with {len(dataset_items)} items "
            f"(description={RESULT_DESCRIPTION!r})..."
        )
        print(
            f"Throttle: sequential strategies, {inter_item_sleep_sec}s between items, "
            f"{inter_strategy_sleep_sec}s between strategies, up to {MAX_RUN_ATTEMPTS} attempts per call."
        )

        for idx, (item_id, text_adv) in enumerate(dataset_items, 1):
            print(f"\nProcessing item {idx}/{len(dataset_items)} (ID: {item_id})...")
            if idx > 1:
                await asyncio.sleep(inter_item_sleep_sec)

            user_z = InstructionZeroShotPrompt.replace("{INPUT_TEXT}", text_adv)
            user_s = InstructionStructuredPrompt.replace("{INPUT_TEXT}", text_adv)
            user_c = InstructionConstraintPrompt.replace("{INPUT_TEXT}", text_adv)

            result_zeroshot = await _run_agent_with_retry(ZeroAgent, user_z)
            await asyncio.sleep(inter_strategy_sleep_sec)
            result_structured = await _run_agent_with_retry(StructuredAgent, user_s)
            await asyncio.sleep(inter_strategy_sleep_sec)
            result_constraint = await _run_agent_with_retry(ConstraintAgent, user_c)

            output_zeroshot_text = (
                result_zeroshot.final_output if hasattr(result_zeroshot, "final_output") else str(result_zeroshot)
            )
            output_structured_text = (
                result_structured.final_output
                if hasattr(result_structured, "final_output")
                else str(result_structured)
            )
            output_constraint_text = (
                result_constraint.final_output
                if hasattr(result_constraint, "final_output")
                else str(result_constraint)
            )

            results = {
                "zero-shot": (output_zeroshot_text, zero_shot_pv_id),
                "structured": (output_structured_text, structured_pv_id),
                "constraint": (output_constraint_text, constraint_pv_id),
            }

            for _, (output_text, prompt_version_id) in results.items():
                db_session.add(
                    PromptResult(
                        item_id=item_id,
                        prompt_version_id=prompt_version_id,
                        input_text=text_adv,
                        output_text=output_text,
                        model_name=MODEL_NAME_STORED,
                        description=RESULT_DESCRIPTION,
                    )
                )
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
    parser = argparse.ArgumentParser(description="Run Claude Haiku 4.5 on dataset items.")
    parser.add_argument(
        "--subset",
        choices=("pilot", "remaining", "all"),
        default="all",
        help="all: every item with text_adv (~189); pilot: 40-item seed-42 sample; remaining: complement of pilot.",
    )
    parser.add_argument(
        "--inter-item-sleep",
        type=float,
        default=DEFAULT_INTER_ITEM_SLEEP_SEC,
        metavar="SEC",
        help=f"Seconds to wait between dataset items (default: {DEFAULT_INTER_ITEM_SLEEP_SEC}).",
    )
    parser.add_argument(
        "--inter-strategy-sleep",
        type=float,
        default=DEFAULT_INTER_STRATEGY_SLEEP_SEC,
        metavar="SEC",
        help=f"Seconds between sequential strategy calls within one item (default: {DEFAULT_INTER_STRATEGY_SLEEP_SEC}).",
    )
    args = parser.parse_args()
    items = _select_items(args.subset)
    print(
        f"Subset={args.subset!r}: {len(items)} items "
        f"(total with text_adv={len(all_items)}, pilot={len(pilot_items)}, remaining={len(remaining_items)})"
    )
    asyncio.run(
        run_comparison(
            items,
            inter_item_sleep_sec=args.inter_item_sleep,
            inter_strategy_sleep_sec=args.inter_strategy_sleep,
        )
    )


if __name__ == "__main__":
    main()
