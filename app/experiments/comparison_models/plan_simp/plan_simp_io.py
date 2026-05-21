"""Shared helpers for Plan-Simp export / DB load (no heavy imports at module load)."""

from __future__ import annotations

import re


def clean_plan_simp_output(text: str) -> str:
    """Strip plan/control tokens (see plan_simp eval_simp.clean_sequences)."""
    if not text:
        return text
    t = re.sub(r"\[PLAN\].*\[SIMPLIFICATION\]", "", text)
    t = re.sub(r"(\<COPY\>|\<REPHRASE\>|\<SPLIT\>|\<DELETE\>)", "", t)
    t = re.sub(r"\<\\?/?s\>", "", t)
    t = re.sub(r"\<\SEP\>", "", t)
    t = re.sub(r"\<pad\>", "", t)
    t = re.sub(r" +", " ", t)
    return t.strip()
