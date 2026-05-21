"""
Shared evaluation metrics for text simplification.

Pure functions that compute SARI, BERTScore, BLEU, FKGL, FRE, and Perplexity.
Used by the T5 experiment.
"""

from typing import Any, Optional, Tuple

import textstat
from easse.sari import corpus_sari
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# BERTScore and perplexity are imported lazily to avoid loading heavy models at import time


def compute_sari(
    input_text: str, output_text: str, reference_text: Optional[str]
) -> Optional[float]:
    """Compute SARI score. Requires reference_text. Returns None if reference is missing."""
    if not reference_text or not output_text:
        return None
    try:
        return corpus_sari(
            orig_sents=[input_text],
            sys_sents=[output_text],
            refs_sents=[[reference_text]],
        )
    except Exception:
        return None


def compute_bertscore(
    output_text: str, reference_text: Optional[str]
) -> Optional[float]:
    """Compute BERTScore F1. Requires reference_text. Returns None if reference is missing."""
    if not reference_text or not output_text:
        return None
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        from bert_score import score

        _, _, F1 = score(
            cands=[output_text],
            refs=[reference_text],
            lang="en",
            verbose=False,
        )
        return float(F1.item())
    except Exception:
        return None


def compute_bleu(
    output_text: str, reference_text: Optional[str]
) -> Optional[float]:
    """Compute BLEU score. Requires reference_text. Returns None if reference is missing."""
    if not reference_text or not output_text:
        return None
    try:
        smoothing = SmoothingFunction().method1
        candidate = output_text.split()
        reference = reference_text.split()
        return sentence_bleu(
            [reference],
            candidate,
            smoothing_function=smoothing,
        )
    except Exception:
        return None


def compute_fkgl(
    input_text: str, output_text: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute FKGL for input, output, and delta. Returns (fkgl_input, fkgl_output, delta_fkgl)."""
    fkgl_input = None
    fkgl_output = None
    try:
        if input_text:
            fkgl_input = textstat.flesch_kincaid_grade(input_text)
    except Exception:
        pass
    try:
        if output_text:
            fkgl_output = textstat.flesch_kincaid_grade(output_text)
    except Exception:
        pass
    delta_fkgl = None
    if fkgl_input is not None and fkgl_output is not None:
        delta_fkgl = fkgl_output - fkgl_input
    return (fkgl_input, fkgl_output, delta_fkgl)


def compute_fre(
    input_text: str, output_text: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute FRE for input, output, and delta. Returns (fre_input, fre_output, fre_delta)."""
    fre_input = None
    fre_output = None
    try:
        if input_text:
            fre_input = textstat.flesch_reading_ease(input_text)
    except Exception:
        pass
    try:
        if output_text:
            fre_output = textstat.flesch_reading_ease(output_text)
    except Exception:
        pass
    fre_delta = None
    if fre_input is not None and fre_output is not None:
        fre_delta = fre_output - fre_input
    return (fre_input, fre_output, fre_delta)


def compute_perplexity(
    text: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    max_length: int = 1024,
) -> Optional[float]:
    """Compute perplexity for a single text using distilgpt2 (or similar causal LM)."""
    if not text:
        return None
    try:
        import torch

        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = encodings.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        return perplexity
    except Exception:
        return None
