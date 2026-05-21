"""
Model registry and unified LLM caller for RAG pipeline.

Maps model names (openai, gemini, llama, sonar, sonar-pro, claude-haiku-4-5) to their
configuration and provides a single async call_llm() that dispatches
to the appropriate client.
"""

import asyncio
import os
from typing import Any, Dict, Optional

from agents import Agent, Runner
from openai import OpenAI

from app.core.config import settings


MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "openai": {
        "model_id": "gpt-4o-mini",
        "display_name": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "client_type": "agents",
    },
    "gemini": {
        "model_id": "litellm/gemini/gemini-2.0-flash",
        "display_name": "gemini-2.0-flash",
        "env_key": "GEMINI_API_KEY",
        "client_type": "agents",
    },
    "llama": {
        "model_id": "ollama/llama3.2",
        "display_name": "llama3.2",
        "env_key": None,
        "client_type": "ollama",
    },
    "sonar": {
        "model_id": "litellm/perplexity/sonar",
        "display_name": "sonar",
        "env_key": "PERPLEXITYAI_API_KEY",
        "client_type": "agents",
    },
    "sonar-pro": {
        "model_id": "litellm/perplexity/sonar-pro",
        "display_name": "sonar-pro",
        "env_key": "PERPLEXITYAI_API_KEY",
        "client_type": "agents",
    },
    "claude-haiku-4-5": {
        "model_id": "litellm/anthropic/claude-haiku-4-5-20251001",
        "display_name": "claude-haiku-4-5",
        "env_key": "CLAUDE_API_KEY",
        "client_type": "agents",
    },
}


def get_model_display_name(model_name: str) -> str:
    """Return the display name for prompt_results.model_name."""
    config = MODEL_CONFIG.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")
    return config.get("display_name", model_name)


def _ensure_env(model_name: str) -> None:
    """Set required env var for the model if applicable."""
    config = MODEL_CONFIG.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_CONFIG.keys())}")
    env_key = config.get("env_key")
    if env_key:
        api_key = getattr(settings, env_key, None) or os.environ.get(env_key)
        if not api_key:
            raise ValueError(f"{env_key} is not set. Add it to .env or environment.")
        os.environ[env_key] = api_key
        if env_key == "CLAUDE_API_KEY":
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)


async def call_llm(model_name: str, full_prompt: str) -> str:
    """
    Call the LLM with the given prompt. Dispatches to the correct client
    based on model_name.

    Args:
        model_name: One of openai, gemini, llama, sonar, sonar-pro.
        full_prompt: The complete prompt text to send.

    Returns:
        The model's response text.
    """
    _ensure_env(model_name)
    config = MODEL_CONFIG[model_name]
    model_id = config["model_id"]
    client_type = config["client_type"]

    if client_type == "ollama":
        return await _call_ollama(model_id, full_prompt)
    else:
        return await _call_agents(model_id, full_prompt)


async def _call_agents(model_id: str, full_prompt: str) -> str:
    """Use agents library (Agent + Runner) for OpenAI, Gemini, Perplexity."""
    agent = Agent(
        name="RAGAgent",
        instructions="",
        model=model_id,
    )
    result = await Runner.run(agent, full_prompt)
    return result.final_output if hasattr(result, "final_output") else str(result)


async def _call_ollama(model_id: str, full_prompt: str) -> str:
    """Use OpenAI-compatible client for local Ollama."""
    # model_id is e.g. "ollama/llama3.2"; Ollama uses just "llama3.2"
    ollama_model = model_id.replace("ollama/", "") if model_id.startswith("ollama/") else model_id
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=ollama_model,
            messages=[{"role": "user", "content": full_prompt}],
        ),
    )
    return response.choices[0].message.content or ""
