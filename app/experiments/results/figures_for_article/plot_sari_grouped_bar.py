"""
plot_sari_grouped_bar.py
========================
Reads a LaTeX results table, parses metric scores, and produces grouped
bar charts (basic / refined / RAG) for every Model × Strategy combination.

Usage:
    python plot_sari_grouped_bar.py

Outputs are written next to this script. See PLOT_CONFIGS for filenames.

Requirements: matplotlib, pandas  (pip install matplotlib pandas)
"""

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")

ENHANCEMENT_LABELS = {
    "none":    "basic",
    "refined": "refined",
    "RAG":     "RAG",
}

ENHANCEMENT_ORDER = ["basic", "refined", "RAG"]

BAR_COLORS = {
    "basic":   "#5B9BD5",
    "refined": "#ED7D31",
    "RAG":     "#70AD47",
}

ALL_STRATEGIES = ["Zero-shot", "Structured", "Constraint"]
NO_STRUCTURED = ["Zero-shot", "Constraint"]

# Column index in the LaTeX table (after splitting on '&').
METRIC_COLUMNS = {
    "sari":  3,
    "lens":  9,
    "fre":   8,
}

METRIC_LABELS = {
    "sari": "SARI Score",
    "lens": "LENS Score",
    "fre":  r"FRE$_{\Delta}$",
}

# Each entry: metric, strategies, figure splits (models + output + title suffix).
PLOT_CONFIGS = [
    # Original: SARI, all strategies
    {
        "metric": "sari",
        "strategies": ALL_STRATEGIES,
        "splits": [
            {
                "models": ["Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini"],
                "output": "sari_llama_gemini_gpt.png",
            },
            {
                "models": ["Sonar", "Claude Haiku 4.5"],
                "output": "sari_sonar_calude.png",
            },
            {
                "models": [
                    "Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini",
                    "Sonar", "Claude Haiku 4.5",
                ],
                "output": "sari_all.png",
            },
        ],
    },
    # 1. SARI without Structured
    {
        "metric": "sari",
        "strategies": NO_STRUCTURED,
        "splits": [
            {
                "models": ["Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini"],
                "output": "sari_no_structured_llama_gemini_gpt.png",
            },
            {
                "models": ["Sonar", "Claude Haiku 4.5"],
                "output": "sari_no_structured_sonar_claude.png",
            },
            {
                "models": [
                    "Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini",
                    "Sonar", "Claude Haiku 4.5",
                ],
                "output": "sari_no_structured_all.png",
            },
        ],
    },
    # 2. LENS with Structured (all strategies)
    {
        "metric": "lens",
        "strategies": ALL_STRATEGIES,
        "splits": [
            {
                "models": ["Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini"],
                "output": "lens_llama_gemini_gpt.png",
            },
            {
                "models": ["Sonar", "Claude Haiku 4.5"],
                "output": "lens_sonar_claude.png",
            },
            {
                "models": [
                    "Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini",
                    "Sonar", "Claude Haiku 4.5",
                ],
                "output": "lens_all.png",
            },
        ],
    },
    # 3. LENS without Structured
    {
        "metric": "lens",
        "strategies": NO_STRUCTURED,
        "splits": [
            {
                "models": ["Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini"],
                "output": "lens_no_structured_llama_gemini_gpt.png",
            },
            {
                "models": ["Sonar", "Claude Haiku 4.5"],
                "output": "lens_no_structured_sonar_claude.png",
            },
            {
                "models": [
                    "Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini",
                    "Sonar", "Claude Haiku 4.5",
                ],
                "output": "lens_no_structured_all.png",
            },
        ],
    },
    # 4. FRE with Structured (all strategies)
    {
        "metric": "fre",
        "strategies": ALL_STRATEGIES,
        "splits": [
            {
                "models": ["Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini"],
                "output": "fre_llama_gemini_gpt.png",
            },
            {
                "models": ["Sonar", "Claude Haiku 4.5"],
                "output": "fre_sonar_claude.png",
            },
            {
                "models": [
                    "Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini",
                    "Sonar", "Claude Haiku 4.5",
                ],
                "output": "fre_all.png",
            },
        ],
    },
    # 5. FRE without Structured
    {
        "metric": "fre",
        "strategies": NO_STRUCTURED,
        "splits": [
            {
                "models": ["Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini"],
                "output": "fre_no_structured_llama_gemini_gpt.png",
            },
            {
                "models": ["Sonar", "Claude Haiku 4.5"],
                "output": "fre_no_structured_sonar_claude.png",
            },
            {
                "models": [
                    "Llama 3.2", "Gemini 2.0 Flash", "GPT-4o-mini",
                    "Sonar", "Claude Haiku 4.5",
                ],
                "output": "fre_no_structured_all.png",
            },
        ],
    },
]


# ── LaTeX helpers ──────────────────────────────────────────────────────────────

def strip_latex(text: str) -> str:
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('{', '').replace('}', '').strip()
    return text


def parse_multirow(cell: str) -> str:
    m = re.match(r'\\multirow\s*\{[^}]+\}\s*\{[^}]+\}\s*\{([^}]*)\}', cell.strip())
    if m:
        return strip_latex(m.group(1))
    return strip_latex(cell)


def parse_float(cell: str) -> float | None:
    cleaned = strip_latex(cell)
    try:
        return float(cleaned)
    except ValueError:
        return None


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_table(filepath: str) -> pd.DataFrame:
    """Parse the LaTeX results table.

    Columns: model, strategy, enhancement, sari, bertscore, bleu,
             perplexity, fkgl, fre, lens
    """
    records = []
    current_model = None
    current_strategy = None

    with open(filepath, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith('%'):
                continue
            if not ('&' in line and '\\\\' in line):
                continue

            line = line.rstrip('\\').rstrip()
            cells = [c.strip() for c in line.split('&')]
            if len(cells) < 10:
                continue

            model_clean = parse_multirow(cells[0])
            if model_clean:
                current_model = model_clean

            strategy_clean = parse_multirow(cells[1])
            if strategy_clean:
                current_strategy = strategy_clean

            enhancement_clean = strip_latex(cells[2])
            display_enhancement = ENHANCEMENT_LABELS.get(
                enhancement_clean, enhancement_clean)

            if current_model is None or current_strategy is None:
                continue

            values = {
                "sari":  parse_float(cells[3]),
                "bertscore": parse_float(cells[4]),
                "bleu": parse_float(cells[5]),
                "perplexity": parse_float(cells[6]),
                "fkgl": parse_float(cells[7]),
                "fre": parse_float(cells[8]),
                "lens": parse_float(cells[9]),
            }
            if values["sari"] is None:
                continue

            records.append({
                "model": current_model,
                "strategy": current_strategy,
                "enhancement": display_enhancement,
                **values,
            })

    return pd.DataFrame(records)


# ── Plot ─────────────────────────────────────────────────────────────────────────

def _strategy_suffix(strategies: list[str]) -> str:
    if strategies == ALL_STRATEGIES:
        return ""
    return " (Zero-shot & Constraint only)"


def plot_grouped_bars(
    df: pd.DataFrame,
    output_path: str,
    metric: str,
    strategies: list[str],
    model_subset: list | None = None,
    show: bool = False,
) -> None:
    value_col = metric
    ylabel = METRIC_LABELS[metric]
    metric_title = metric.upper() if metric == "sari" else metric.upper()

    if model_subset:
        df = df[df["model"].isin(model_subset)].copy()
        model_order = model_subset
    else:
        model_order = list(dict.fromkeys(df["model"]))

    groups = []
    for model in model_order:
        for strategy in strategies:
            subset = df[(df["model"] == model) & (df["strategy"] == strategy)]
            if not subset.empty:
                groups.append((model, strategy))

    n_groups = len(groups)
    n_bars = len(ENHANCEMENT_ORDER)
    bar_width = 0.22
    group_gap = 0.15

    positions = []
    x = 0.0
    prev_model = None
    for model, strategy in groups:
        if prev_model is not None and model != prev_model:
            x += group_gap
        positions.append(x)
        x += n_bars * bar_width + 0.18
        prev_model = model

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.6), 7))

    values_in_chart = []
    for (model, strategy), group_x in zip(groups, positions):
        subset = df[(df["model"] == model) & (df["strategy"] == strategy)]

        for bi, enh in enumerate(ENHANCEMENT_ORDER):
            row = subset[subset["enhancement"] == enh]
            if row.empty:
                continue
            val = row[value_col].values[0]
            values_in_chart.append(val)
            bar_x = group_x + bi * bar_width

            ax.bar(
                bar_x, val,
                width=bar_width * 0.92,
                color=BAR_COLORS[enh],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

            label_offset = abs(val) * 0.03 + 0.5
            ax.text(
                bar_x + bar_width * 0.92 / 2,
                val + (label_offset if val >= 0 else -label_offset),
                f"{val:.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=6.5,
                rotation=90,
                color="#333333",
            )

    tick_positions = [p + (n_bars - 1) * bar_width / 2 for p in positions]
    tick_labels = [f"{model}\n{strategy}" for model, strategy in groups]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    prev_model = None
    for gi, (model, strategy) in enumerate(groups):
        if prev_model is not None and model != prev_model:
            sep_x = (positions[gi] + positions[gi - 1] + n_bars * bar_width) / 2
            ax.axvline(sep_x, color="#cccccc", linewidth=1.0, linestyle="--", zorder=1)
        prev_model = model

    legend_handles = [
        mpatches.Patch(color=BAR_COLORS[enh], label=enh)
        for enh in ENHANCEMENT_ORDER
    ]
    ax.legend(handles=legend_handles, title="Enhancement",
              loc="upper left", framealpha=0.9, fontsize=10)

    ax.set_xlabel("Model and Strategy", fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12)

    models_label = " · ".join(model_order)
    title = (
        f"{metric_title} by Model, Strategy, and Enhancement\n"
        f"({models_label}){_strategy_suffix(strategies)}"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    if values_in_chart:
        vmin, vmax = min(values_in_chart), max(values_in_chart)
        span = vmax - vmin if vmax != vmin else abs(vmax) or 1.0
        pad = span * 0.20
        if vmin >= 0:
            ax.set_ylim(0, vmax + pad)
        else:
            ax.set_ylim(vmin - pad, vmax + pad)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Reading: {INPUT_FILE}")
    df = parse_table(INPUT_FILE)
    print(f"\nParsed {len(df)} rows")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    for config in PLOT_CONFIGS:
        metric = config["metric"]
        strategies = config["strategies"]
        print(f"\n── {metric.upper()} | strategies: {', '.join(strategies)} ──")
        for split in config["splits"]:
            out = os.path.join(base_dir, split["output"])
            print(f"  Generating: {out}")
            plot_grouped_bars(
                df, out,
                metric=metric,
                strategies=strategies,
                model_subset=split["models"],
            )
