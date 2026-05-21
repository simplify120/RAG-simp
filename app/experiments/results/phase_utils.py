"""Shared phase / experiment labels for results pipeline (DB description + version → phase)."""

from __future__ import annotations

import pandas as pd

from app.experiments.visualization.config import DESCRIPTION_TO_PHASE

EXCLUDED_MODELS = {"sonar-pro"}

# Older exports used human-readable phase names; charts/tables key off PHASE_ORDER (PE_v1 / PE_v2).
_LEGACY_PHASE_TO_CANONICAL = {
    "Simple Prompt v1": "PE_v1",
    "Simple Prompt v2": "PE_v2",
}

PHASE_TO_META = {
    # Keys must match strings returned by assign_phase / DESCRIPTION_TO_PHASE (see PHASE_ORDER).
    "PE_v1": ("Simple Prompt v1", "none"),
    "PE_v2": ("Simple Prompt v2", "none"),
    "Open AI RAG v2": ("OpenAI RAG best of v1/v2", "openai"),
    "Open AI RAG v3": ("OpenAI RAG v3", "openai"),
    "E5 RAG v3": ("E5 v3", "e5"),
    "BGE RAG v3": ("BGE v3", "bge"),
}


def assign_phase(row: pd.Series) -> str:
    """Match visualization/data_loader._assign_phase; PE labels must match PHASE_ORDER (PE_v1 / PE_v2)."""
    desc = row.get("description")
    version = row.get("version")

    phase = DESCRIPTION_TO_PHASE.get(desc)
    if phase is not None:
        return phase

    if desc == "step 1 - simple prompt engineering":
        if version == "v1":
            return "PE_v1"
        if version == "v2":
            return "PE_v2"

    return "UNKNOWN"


def canonicalize_phase_column(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy `phase` labels to canonical keys (matches PHASE_ORDER)."""
    if df.empty or "phase" not in df.columns:
        return df
    out = df.copy()
    out["phase"] = out["phase"].replace(_LEGACY_PHASE_TO_CANONICAL)
    return out


def add_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["phase"] = out.apply(assign_phase, axis=1)
    out["experiment_group"] = out["phase"].map(
        lambda p: PHASE_TO_META.get(p, (None, None))[0]
    )
    out["retrieval"] = out["phase"].map(lambda p: PHASE_TO_META.get(p, (None, None))[1])
    return out


def filter_models(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model" not in df.columns:
        return df
    return df[~df["model"].isin(EXCLUDED_MODELS)].copy()
