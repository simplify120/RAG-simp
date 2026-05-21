"""
config.py — Global style constants and metric metadata for the visualization suite.

All plotting scripts import from here to ensure a consistent look-and-feel.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Model color palette  (consistent across all figures)
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "gpt-4o-mini":         "#4C72B0",   # steel blue
    "gemini-2.0-flash":    "#DD8452",   # warm orange
    "llama3.2":            "#55A868",   # muted green
    "sonar":               "#C44E52",   # rose red
    "sonar-pro":           "#8172B2",   # violet
    "claude-haiku-4-5":    "#B07AA1",   # muted purple
    "t5-large-text-simplification": "#937860",  # tan / brown (baseline)
    "plan-simp-pgdyn": "#2E86AB",  # teal (Plan-Simp baseline)
}

# Short display labels (for legends, axes)
MODEL_LABELS = {
    "gpt-4o-mini":         "GPT-4o-mini",
    "gemini-2.0-flash":    "Gemini 2.0 Flash",
    "llama3.2":            "Llama 3.2",
    "sonar":               "Sonar",
    "sonar-pro":           "Sonar Pro",
    "claude-haiku-4-5":    "Claude Haiku 4.5",
    "t5-large-text-simplification": "T5-Large",
    "plan-simp-pgdyn": "Plan-Simp",
}

# ---------------------------------------------------------------------------
# Phase definitions (ordered for evolution charts)
# ---------------------------------------------------------------------------
PHASE_ORDER = ["PE_v1", "PE_v2", "Open AI RAG v2", "Open AI RAG v3", "E5 RAG v3", "BGE RAG v3"]

PHASE_LABELS = {
    "PE_v1":  "SE v1\n(Initial)",
    "PE_v2":  "SE v2\n(Improved)",
    "Open AI RAG v2": "Open AI RAG v2\n(Top-K=3)",
    "Open AI RAG v3": "Open AI RAG v3\n(Clean Output)",
    "E5 RAG v3": "E5 RAG v3\n(Clean Output)",
    "BGE RAG v3": "BGE RAG v3\n(Clean Output)",
}

# Compact labels for crowded x-axes (same order as PHASE_ORDER)
PHASE_AXIS_SHORT = [
    "Simple Prompt\nv1",
    "Simple Prompt\nv2",
    "OpenAI RAG\nbest of v1/v2",
    "OpenAI RAG\nv3",
    "E5\nv3",
    "BGE\nv3",
]

# Display names for use in figure labels (raw phase key → human-readable name)
PHASE_DISPLAY_NAMES = {
    "PE_v1":          "Simple Prompt v1",
    "PE_v2":          "Simple Prompt v2",
    "Open AI RAG v2": "OpenAI RAG best of v1/v2",
    "Open AI RAG v3": "OpenAI RAG v3",
    "E5 RAG v3":      "E5 v3",
    "BGE RAG v3":     "BGE v3",
}

# DB description → phase key (must match PHASE_ORDER exactly)
DESCRIPTION_TO_PHASE = {
    "step 1 - simple prompt engineering": None,       # split by version field
    "step 2 - RAG top k=3":               "Open AI RAG v2",
    "step 2 - RAG top k=3 with upgrated prompt": "Open AI RAG v3",
    "step 2 - RAG top k=3 with upgraded prompt": "Open AI RAG v3",
    "E5-RAG-full":                        "E5 RAG v3",
    "BGE-RAG-full":                       "BGE RAG v3",
}

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
STRATEGY_ORDER = ["constraint", "structured", "zeroshot"]

STRATEGY_LABELS = {
    "constraint":  "Constraint",
    "structured":  "Structured",
    "zeroshot":   "Zero-shot",
}

STRATEGY_COLORS = {
    "constraint":  "#264653",
    "structured":  "#2A9D8F",
    "zeroshot":   "#E9C46A",
}

# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------
METRICS_QUALITY = ["sari", "bleu", "bertscore", "lens"]
METRICS_READABILITY = ["fkgl_delta", "fre_delta"]
METRICS_OUTPUT_READABILITY = ["fkgl_output", "fre_output"]
METRICS_ALL = METRICS_QUALITY + METRICS_READABILITY + ["perplexity"]

# Add missing metric definitions directly toMETRIC_META
METRIC_META = {
    "sari": {
        "label": "SARI",
        "higher_is_better": True,
        "unit": "",
        "description": "SARI (Similarity-based Metric for Simplification)",
    },
    "bleu": {
        "label": "BLEU",
        "higher_is_better": True,
        "unit": "",
        "description": "BLEU Score",
    },
    "bertscore": {
        "label": "BERTScore",
        "higher_is_better": True,
        "unit": "",
        "description": "BERTScore F1",
    },
    "lens": {
        "label": "LENS",
        "higher_is_better": True,
        "unit": "",
        "description": "LENS (Learned Evaluation for Natural-language Simplification)",
    },
    "perplexity": {
        "label": "Perplexity",
        "higher_is_better": False,
        "unit": "",
        "description": "Perplexity (lower = more fluent)",
    },
    "fkgl_delta": {
        "label": "FKGL Δ",
        "higher_is_better": False,  # more negative = simpler
        "unit": "grade levels",
        "description": "FKGL Delta (negative = grade level reduction)",
    },
    "fre_delta": {
        "label": "FRE Δ",
        "higher_is_better": True,
        "unit": "points",
        "description": "Flesch Reading Ease Delta (positive = easier)",
    },
    "fkgl_output": {
        "label": "FKGL (Output)",
        "higher_is_better": False,
        "unit": "grade level",
        "description": "Flesch-Kincaid Grade Level of simplified text",
    },
    "fre_output": {
        "label": "FRE (Output)",
        "higher_is_better": True,
        "unit": "",
        "description": "Flesch Reading Ease of simplified text",
    },
}

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

VIZ_DIR = Path(__file__).parent
OUTPUT_DIR = VIZ_DIR / "outputs" / "figures"


def get_output_dir(subfolder: str) -> Path:
    """Return (and create) the output directory for a plot module."""
    d = OUTPUT_DIR / subfolder
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
FIGURE_DPI = 300
FONT_FAMILY = "DejaVu Sans"

def apply_publication_style():
    """Apply a clean, publication-ready matplotlib style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    mpl.rcParams.update({
        "font.family":           FONT_FAMILY,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.grid":             True,
        "axes.grid.axis":        "y",
        "grid.alpha":            0.35,
        "grid.linestyle":        "--",
        "figure.dpi":            100,       # screen; saved at FIGURE_DPI
        "savefig.dpi":           FIGURE_DPI,
        "savefig.bbox":          "tight",
        "legend.framealpha":     0.85,
        "legend.edgecolor":      "0.8",
    })


def save_figure(fig: plt.Figure, filename: str, subfolder: str, formats=("png",)):
    """Save a figure to the output directory in the requested formats."""
    out_dir = get_output_dir(subfolder)
    for fmt in formats:
        filepath = out_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved  {subfolder}/{filename}  ({', '.join(formats)})")
