"""
Publication-quality figure suite for results/figures/.

Outputs are grouped into subfolders (similar to app/experiments/visualization/outputs/figures):
  01_phase_evolution/   — one metric per figure (no inset clutter)
  02_baselines/         — LLM best configs vs T5 & Plan-Simp (full-dataset CSVs)
  03_strategy/          — heatmaps + distributions
  04_rag/               — retriever comparison (split by strategy)
  05_readability/       — FKGL Δ and FRE Δ as separate figures
  06_rankings/          — top configurations

Baselines are read from:
  app/experiments/comparison_models/t5_model/outputs/t5_full_dataset_analysis.csv
  app/experiments/comparison_models/plan_simp/outputs/plan_simp_full_dataset_analysis.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from app.experiments.results.phase_utils import filter_models
from app.experiments.visualization.config import (
    FIGURE_DPI,
    METRIC_META,
    MODEL_COLORS,
    MODEL_LABELS,
    PHASE_AXIS_SHORT,
    PHASE_DISPLAY_NAMES,
    PHASE_LABELS,
    PHASE_ORDER,
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    apply_publication_style,
)

# ---------------------------------------------------------------------------
# Paths & model ordering
# ---------------------------------------------------------------------------

RESULTS_PACKAGE = Path(__file__).resolve().parent

MODEL_ORDER = [
    "gpt-4o-mini",
    "gemini-2.0-flash",
    "llama3.2",
    "sonar",
    "claude-haiku-4-5",
]

RAG_PHASES_V3 = {"Open AI RAG v3", "E5 RAG v3", "BGE RAG v3"}
PE_PHASES = {"PE_v1", "PE_v2"}
ALL_RAG_PHASES = {
    "Open AI RAG v2",
    "Open AI RAG v3",
    "E5 RAG v3",
    "BGE RAG v3",
}

def _repo_root() -> Path:
    return RESULTS_PACKAGE.parent.parent.parent


def baseline_csv_paths() -> dict[str, Path]:
    root = _repo_root()
    return {
        "t5-large-text-simplification": root
        / "app/experiments/comparison_models/t5_model/outputs/t5_full_dataset_analysis.csv",
        "plan-simp-pgdyn": root
        / "app/experiments/comparison_models/plan_simp/outputs/plan_simp_full_dataset_analysis.csv",
    }


def load_comparison_baselines() -> pd.DataFrame:
    """One row per available baseline model (full-dataset aggregate CSVs)."""
    rows = []
    for model_id, path in baseline_csv_paths().items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        r = df.iloc[0].to_dict()
        r["_baseline_id"] = model_id
        rows.append(r)
    return pd.DataFrame(rows)


def _ordered_models(series_or_df) -> list[str]:
    if hasattr(series_or_df, "columns") and "model" in series_or_df.columns:
        present = set(series_or_df["model"].unique())
    else:
        present = set(series_or_df.unique())
    out = [m for m in MODEL_ORDER if m in present]
    out += sorted(present - set(out))
    return out


def _sem(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) <= 1:
        return 0.0
    return float(s.std() / np.sqrt(len(s)))


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _phase_metric_table(det: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for (ph, m), grp in det.groupby(["phase", "model"], observed=False):
        rows.append(
            {
                "phase": ph,
                "model": m,
                "mean": grp[value_col].mean(),
                "sem": _sem(grp[value_col]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 01 Phase evolution — one file per metric
# ---------------------------------------------------------------------------


def plot_phase_lines_single_metric(
    det: pd.DataFrame,
    value_col: str,
    metric_key: str,
    baselines: pd.DataFrame,
    baseline_csv_col: str,
    out_dir: Path,
    filename: str,
    title_suffix: str,
) -> None:
    apply_publication_style()
    d = det[det["phase"].isin(PHASE_ORDER)].copy()
    g = _phase_metric_table(d, value_col)
    if g.empty:
        print(f"  Skip {filename} (no data)")
        return
    g["phase"] = pd.Categorical(g["phase"], categories=PHASE_ORDER, ordered=True)
    models = _ordered_models(g)
    x = np.arange(len(PHASE_ORDER))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for m in models:
        sub = g[g["model"] == m].set_index("phase").reindex(PHASE_ORDER)
        y = sub["mean"].values.astype(float)
        ylo = y - sub["sem"].values.astype(float)
        yhi = y + sub["sem"].values.astype(float)
        color = MODEL_COLORS.get(m, "#555555")
        ax.plot(
            x,
            y,
            "o-",
            color=color,
            label=MODEL_LABELS.get(m, m),
            linewidth=2.0,
            markersize=5,
            zorder=3,
        )
        ax.fill_between(x, ylo, yhi, color=color, alpha=0.14, zorder=1)

    # Baselines: horizontal reference (full-dataset supervised / specialized models)
    if not baselines.empty and baseline_csv_col in baselines.columns:
        for _, br in baselines.iterrows():
            bid = br.get("_baseline_id", "")
            val = br.get(baseline_csv_col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            color = MODEL_COLORS.get(str(bid), "#666666")
            lbl = MODEL_LABELS.get(str(bid), str(bid))
            ax.axhline(
                y=float(val),
                color=color,
                linestyle="--",
                linewidth=1.8,
                zorder=2,
                label=f"{lbl}",
            )

    ax.set_xticks(x)
    if len(PHASE_AXIS_SHORT) == len(PHASE_ORDER):
        ax.set_xticklabels(PHASE_AXIS_SHORT, fontsize=9)
    else:
        ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in PHASE_ORDER], fontsize=8)
    meta = METRIC_META.get(metric_key, {"label": metric_key})
    ax.set_ylabel(meta.get("label", metric_key))
    ax.set_xlabel("Experimental phase")
    ax.set_title(f"{meta.get('label', metric_key)} across phases {title_suffix}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    _save(fig, out_dir, filename)
    print(f"  Wrote {out_dir.name}/{filename}")


# ---------------------------------------------------------------------------
# 02 Baselines — best LLM vs T5 & Plan-Simp
# ---------------------------------------------------------------------------


def _best_row_per_model(agg: pd.DataFrame) -> pd.DataFrame:
    """Best (highest SARI) configuration per LLM."""
    rows = []
    for m, g in agg.groupby("model"):
        g2 = g[g["sari_mean"].notna()]
        if g2.empty:
            continue
        i = g2["sari_mean"].idxmax()
        rows.append(g2.loc[i].to_dict())
    return pd.DataFrame(rows)


def _build_baseline_metric_rows(
    best: pd.DataFrame,
    baselines: pd.DataFrame,
    mean_col: str,
    std_col: str,
    meta_key: str,
) -> pd.DataFrame:
    rows = []
    if not baselines.empty:
        csv_map = {
            "sari": "SARI",
            "bertscore": "BERTScore",
            "lens": "LENS",
            "bleu": "BLEU",
        }
        c_mean = csv_map.get(meta_key)
        c_std = f"{c_mean}_std" if c_mean else None
        for _, br in baselines.iterrows():
            bid = str(br.get("_baseline_id", ""))
            if not c_mean or c_mean not in br:
                continue
            sd = 0.0
            if c_std and c_std in br and pd.notna(br.get(c_std)):
                sd = float(br[c_std])
            rows.append(
                {
                    "label": MODEL_LABELS.get(bid, bid),
                    "mean": float(br[c_mean]),
                    "std": sd,
                    "color": MODEL_COLORS.get(bid, "#888888"),
                }
            )
    for _, r in best.iterrows():
        if mean_col not in r or pd.isna(r.get(mean_col)):
            continue
        m = r["model"]
        sd = float(r[std_col]) if std_col in r and pd.notna(r.get(std_col)) else 0.0
        rows.append(
            {
                "label": MODEL_LABELS.get(m, m) + " (best run)",
                "mean": float(r[mean_col]),
                "std": sd,
                "color": MODEL_COLORS.get(m, "#4C72B0"),
            }
        )
    return pd.DataFrame(rows)


def _format_metric_value(meta_key: str, mu: float) -> str:
    if meta_key == "bertscore":
        return f"{mu:.4f}"
    if meta_key == "bleu":
        return f"{mu:.4f}"
    if meta_key == "sari":
        return f"{mu:.2f}"
    if meta_key == "lens":
        return f"{mu:.2f}"
    return f"{mu:.3f}"


def plot_best_llm_vs_baselines_grid(agg: pd.DataFrame, baselines: pd.DataFrame, out_dir: Path) -> None:
    """
    One figure per metric (SARI, BERTScore, LENS, BLEU): horizontal bars, baselines + best LLM runs.
    Labels sit to the right of mean+SD with extra x-axis margin so they do not overlap error caps.
    """
    best = _best_row_per_model(agg)
    if best.empty:
        print("  Skip baseline charts (no aggregate rows)")
        return

    metrics = [
        ("sari_mean", "sari_std", "SARI", "sari", "01_sari_vs_baselines"),
        ("bertscore_mean", "bertscore_std", "BERTScore", "bertscore", "02_bertscore_vs_baselines"),
        ("lens_mean", "lens_std", "LENS", "lens", "03_lens_vs_baselines"),
        ("bleu_mean", "bleu_std", "BLEU", "bleu", "04_bleu_vs_baselines"),
    ]

    for mean_col, std_col, title, meta_key, fname in metrics:
        apply_publication_style()
        plot_df = _build_baseline_metric_rows(best, baselines, mean_col, std_col, meta_key)
        if plot_df.empty:
            continue
        plot_df = plot_df.sort_values("mean", ascending=False).reset_index(drop=True)
        n = len(plot_df)
        y = np.arange(n)

        fig, ax = plt.subplots(figsize=(9.2, max(4.0, 0.42 * n + 2.2)))
        err = plot_df["std"].fillna(0.0).values
        ax.barh(
            y,
            plot_df["mean"].values,
            xerr=err,
            color=plot_df["color"].values,
            alpha=0.9,
            capsize=3,
            height=0.62,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["label"].values, fontsize=9)
        ax.invert_yaxis()

        meta = METRIC_META[meta_key]
        dir_hint = "" if meta.get("higher_is_better", True) else "(lower is better)"
        ax.set_xlabel(f'{meta["label"]} — mean {dir_hint}')
        ax.set_title(
            f"Best run per LLM vs full-dataset baselines (T5-Large, Plan-Simp)\n{title}",
            fontsize=11,
            pad=12,
        )

        # Room for value labels past error bars (avoids overlap with caps)
        means = plot_df["mean"].values.astype(float)
        stds = plot_df["std"].fillna(0.0).values.astype(float)
        right_edges = means + stds
        data_max = float(np.nanmax(right_edges)) if len(right_edges) else 1.0
        data_max = max(data_max, float(np.nanmax(means)) if len(means) else 1.0)
        span = max(data_max, 1e-12)
        pad_frac = 0.22 if meta_key == "bleu" else 0.16
        ax.set_xlim(0, span * (1.0 + pad_frac))

        label_offset = 0.018 * span * (1.0 + pad_frac)
        for i, row in plot_df.iterrows():
            mu = float(row["mean"])
            sd = float(row["std"]) if pd.notna(row["std"]) else 0.0
            x_end = mu + sd
            tx = x_end + label_offset
            txt = _format_metric_value(meta_key, mu)
            ax.text(
                tx,
                i,
                txt,
                va="center",
                ha="left",
                fontsize=9,
                color="0.15",
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "linewidth": 0.6,
                    "alpha": 0.96,
                },
                clip_on=False,
            )

        ax.grid(axis="x", alpha=0.35)
        fig.tight_layout()
        _save(fig, out_dir, fname)
        print(f"  Wrote {out_dir.name}/{fname}")


# ---------------------------------------------------------------------------
# 03 Strategy analysis
# ---------------------------------------------------------------------------


def plot_strategy_heatmaps_pe_v2(det: pd.DataFrame, out_dir: Path) -> None:
    for metric, csv_name in (("sari", "sari"), ("bertscore", "bertscore")):
        apply_publication_style()
        d = det[det["phase"] == "PE_v2"].copy()
        if d.empty:
            continue
        pv = d.pivot_table(index="model", columns="strategy", values=metric, aggfunc="mean")
        for s in STRATEGY_ORDER:
            if s not in pv.columns:
                pv[s] = np.nan
        pv = pv[[c for c in STRATEGY_ORDER if c in pv.columns]]
        mo = _ordered_models(d)
        pv = pv.reindex([m for m in mo if m in pv.index])
        if pv.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        sns.heatmap(
            pv,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            linewidths=0.6,
            ax=ax,
            cbar_kws={"label": METRIC_META[metric]["label"]},
        )
        ax.set_title(f'{METRIC_META[metric]["label"]} — model × strategy (PE v2 only)')
        ax.set_xlabel("Prompting strategy")
        ax.set_ylabel("Model")
        ax.set_xticklabels([STRATEGY_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
        ax.set_yticklabels([MODEL_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])
        fig.tight_layout()
        _save(fig, out_dir, f"01_heatmap_pe_v2_{csv_name}")
        print(f"  Wrote {out_dir.name}/01_heatmap_pe_v2_{csv_name}")


def plot_strategy_heatmaps_global_mean(det: pd.DataFrame, out_dir: Path) -> None:
    """Strategy × model, mean over all phases (like original visualization suite)."""
    for metric in ("sari", "bertscore"):
        apply_publication_style()
        d = det[det["strategy"].isin(STRATEGY_ORDER)].copy()
        if d.empty:
            continue
        pv = d.pivot_table(index="strategy", columns="model", values=metric, aggfunc="mean")
        idx = [s for s in STRATEGY_ORDER if s in pv.index]
        pv = pv.reindex(idx)
        cols = _ordered_models(det)
        pv = pv[[c for c in cols if c in pv.columns]]
        pv.index = [STRATEGY_LABELS.get(i, i) for i in pv.index]
        pv.columns = [MODEL_LABELS.get(c, c) for c in pv.columns]
        meta = METRIC_META[metric]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        cmap = "YlGnBu" if meta["higher_is_better"] else "YlOrRd"
        sns.heatmap(pv, annot=True, fmt=".3f", cmap=cmap, ax=ax, linewidths=0.5, cbar_kws={"label": meta["label"]})
        ax.set_title(f'{meta["label"]} — strategy × model (mean over all phases)')
        ax.set_ylabel("Strategy")
        ax.set_xlabel("Model")
        fig.tight_layout()
        _save(fig, out_dir, f"02_heatmap_strategy_model_mean_{metric}")
        print(f"  Wrote {out_dir.name}/02_heatmap_strategy_model_mean_{metric}")


def plot_strategy_boxplots(det: pd.DataFrame, out_dir: Path) -> None:
    apply_publication_style()
    d = det[det["strategy"].isin(STRATEGY_ORDER)].copy()
    d["strategy"] = pd.Categorical(d["strategy"], categories=STRATEGY_ORDER, ordered=True)
    d = filter_models(d)
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ("sari", "bertscore")):
        sns.boxplot(
            data=d,
            x="strategy",
            y=metric,
            hue="model",
            palette=MODEL_COLORS,
            ax=ax,
            linewidth=0.9,
            fliersize=2,
        )
        ax.set_title(METRIC_META[metric]["label"] + " — distribution by strategy (all phases)")
        ax.set_xlabel("Strategy")
        ax.set_ylabel(METRIC_META[metric]["label"])
        ax.set_xticklabels([STRATEGY_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            [MODEL_LABELS.get(l, l) for l in labels],
            title="Model",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
        )
    fig.tight_layout()
    _save(fig, out_dir, "03_boxplots_by_strategy")
    print(f"  Wrote {out_dir.name}/03_boxplots_by_strategy")


# ---------------------------------------------------------------------------
# 04 RAG — one figure per strategy (cleaner than 3 cramped panels)
# ---------------------------------------------------------------------------


def plot_rag_retriever_by_strategy(agg: pd.DataFrame, out_dir: Path) -> None:
    sub = agg[agg["phase"].isin(RAG_PHASES_V3)].copy()
    if sub.empty:
        print("  Skip RAG figures (no v3 RAG rows)")
        return
    sub["retriever"] = sub["phase"].map(
        {"Open AI RAG v3": "OpenAI v3", "E5 RAG v3": "E5 full", "BGE RAG v3": "BGE full"}
    )
    for strat in STRATEGY_ORDER:
        d = sub[sub["strategy"] == strat]
        if d.empty:
            continue
        apply_publication_style()
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        order_ret = ["OpenAI v3", "BGE full", "E5 full"]
        try:
            sns.barplot(
                data=d,
                x="model",
                y="sari_mean",
                hue="retriever",
                hue_order=[h for h in order_ret if h in set(d["retriever"])],
                palette="Set2",
                ax=ax,
                errorbar=None,
            )
        except TypeError:
            sns.barplot(
                data=d,
                x="model",
                y="sari_mean",
                hue="retriever",
                hue_order=[h for h in order_ret if h in set(d["retriever"])],
                palette="Set2",
                ax=ax,
                ci=None,
            )
        ax.set_title(f"SARI by model and retriever — {STRATEGY_LABELS.get(strat, strat)} (RAG v3 / full)")
        ax.set_xlabel("")
        ax.set_ylabel("SARI (mean ± aggregate SD in table)")
        ax.set_xticklabels([MODEL_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], rotation=18, ha="right")
        ax.legend(title="Retriever", fontsize=8, loc="upper right")
        fig.tight_layout()
        safe = strat.replace(" ", "_")
        _save(fig, out_dir, f"01_rag_retrievers_{safe}")
        print(f"  Wrote {out_dir.name}/01_rag_retrievers_{safe}")


# ---------------------------------------------------------------------------
# 05 Readability — split panels
# ---------------------------------------------------------------------------


def plot_readability_separate(agg: pd.DataFrame, out_dir: Path) -> None:
    d = agg[agg["phase"].isin(PHASE_ORDER)].copy()
    if d.empty:
        return
    d["phase"] = pd.Categorical(d["phase"], categories=PHASE_ORDER, ordered=True)
    g = (
        d.groupby(["phase", "model"], observed=False)
        .agg(
            fkgl_delta_mean=("fkgl_delta_mean", "mean"),
            fre_delta_mean=("fre_delta_mean", "mean"),
        )
        .reset_index()
    )
    models = _ordered_models(g)
    for col, fname, t in (
        ("fkgl_delta_mean", "01_fkgl_delta_by_phase", "FKGL Δ (mean over strategies)"),
        ("fre_delta_mean", "02_fre_delta_by_phase", "FRE Δ (mean over strategies)"),
    ):
        apply_publication_style()
        pivot = g.pivot(index="model", columns="phase", values=col).reindex(models)
        pivot = pivot[[p for p in PHASE_ORDER if p in pivot.columns]]
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        pivot.plot(kind="bar", ax=ax, width=0.82, rot=0)
        # Do not add to legend — otherwise the first legend entry is this line (dashed) and
        # phase labels shift, leaving the last bar color unlabeled.
        ax.axhline(0, color="0.35", linewidth=0.9, linestyle="--", label="_nolegend_", zorder=1)
        ax.set_title(t)
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_META["fkgl_delta" if "fkgl" in col else "fre_delta"]["label"])
        ax.set_xticklabels([MODEL_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], rotation=22, ha="right")
        leg_labels = []
        for c in pivot.columns:
            if c in PHASE_ORDER:
                idx = PHASE_ORDER.index(c)
                leg_labels.append(PHASE_AXIS_SHORT[idx] if idx < len(PHASE_AXIS_SHORT) else str(c))
            else:
                leg_labels.append(str(c))
        # Bar series order = pivot column order (left→right: PE_v1 … BGE RAG v3). Use
        # containers so legend colors match bars (not Simple Prompt v1 on brown — brown is last phase).
        n_series = len(pivot.columns)
        handles = list(ax.containers[:n_series]) if len(ax.containers) >= n_series else ax.get_legend_handles_labels()[0]
        ax.legend(handles, leg_labels, title="Phase", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        _save(fig, out_dir, fname)
        print(f"  Wrote {out_dir.name}/{fname}")


# ---------------------------------------------------------------------------
# 06 Rankings + PE vs RAG (split)
# ---------------------------------------------------------------------------


def plot_top_configurations(agg: pd.DataFrame, out_dir: Path) -> None:
    apply_publication_style()
    g = agg[agg["sari_mean"].notna()].copy()
    if g.empty:
        return
    g["phase_display"] = g["phase"].map(PHASE_DISPLAY_NAMES).fillna(g["phase"])
    g["label"] = g["model"].astype(str) + " | " + g["strategy"].astype(str) + " | " + g["phase_display"].astype(str)
    top = g.sort_values("sari_mean", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    y = np.arange(len(top))
    ax.barh(y, top["sari_mean"].values, xerr=top["sari_std"].values, color="#4C72B0", alpha=0.9, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("SARI (mean ± SD)")
    ax.set_title("Top configurations by aggregate SARI")
    fig.tight_layout()
    _save(fig, out_dir, "01_top_configurations")
    print(f"  Wrote {out_dir.name}/01_top_configurations")


def plot_pe_vs_rag_split(agg: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    models = _ordered_models(agg)
    for m in models:
        g = agg[agg["model"] == m]
        pe = g[g["phase"].isin(PE_PHASES)]
        rag = g[g["phase"].isin(ALL_RAG_PHASES)]
        if not pe.empty and pe["sari_mean"].notna().any():
            i = pe["sari_mean"].idxmax()
            rows.append(
                {
                    "model": m,
                    "setting": "Best PE",
                    "sari": pe.loc[i, "sari_mean"],
                    "sari_std": pe.loc[i, "sari_std"],
                    "bert": pe.loc[i, "bertscore_mean"],
                    "bert_std": pe.loc[i, "bertscore_std"],
                }
            )
        if not rag.empty and rag["sari_mean"].notna().any():
            i = rag["sari_mean"].idxmax()
            rows.append(
                {
                    "model": m,
                    "setting": "Best RAG",
                    "sari": rag.loc[i, "sari_mean"],
                    "sari_std": rag.loc[i, "sari_std"],
                    "bert": rag.loc[i, "bertscore_mean"],
                    "bert_std": rag.loc[i, "bertscore_std"],
                }
            )
    plot = pd.DataFrame(rows)
    if plot.empty:
        return
    for metric, key, err, fname in (
        ("SARI", "sari", "sari_std", "01_pe_vs_rag_sari"),
        ("BERTScore", "bert", "bert_std", "02_pe_vs_rag_bertscore"),
    ):
        apply_publication_style()
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        p = plot.pivot(index="model", columns="setting", values=key)
        errp = plot.pivot(index="model", columns="setting", values=err)
        p = p.reindex(models)
        errp = errp.reindex(models)
        x = np.arange(len(models))
        w = 0.36
        for j, col in enumerate(["Best PE", "Best RAG"]):
            if col not in p.columns:
                continue
            ax.bar(
                x + (j - 0.5) * w,
                p[col].values,
                width=w,
                label=col,
                yerr=errp[col].values,
                capsize=2,
                alpha=0.92,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=18, ha="right", fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}: best prompt-only vs best RAG per model")
        ax.legend(fontsize=9)
        fig.tight_layout()
        _save(fig, out_dir, fname)
        print(f"  Wrote {out_dir.name}/{fname}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def generate_all(
    agg: pd.DataFrame,
    det: pd.DataFrame,
    output_root: Path | None = None,
) -> None:
    output_root = output_root or (RESULTS_PACKAGE / "figures")
    baselines = load_comparison_baselines()

    print("Generating publication figures (subfolders)...")

    p01 = output_root / "01_phase_evolution"
    suffix = "\n" if not baselines.empty else ""
    plot_phase_lines_single_metric(
        det,
        "sari",
        "sari",
        baselines,
        "SARI",
        p01,
        "01_sari_by_phase",
        suffix,
    )
    plot_phase_lines_single_metric(
        det,
        "bertscore",
        "bertscore",
        baselines,
        "BERTScore",
        p01,
        "02_bertscore_by_phase",
        suffix,
    )

    p02 = output_root / "02_baselines"
    plot_best_llm_vs_baselines_grid(agg, baselines, p02)

    p03 = output_root / "03_strategy"
    plot_strategy_heatmaps_pe_v2(det, p03)
    plot_strategy_heatmaps_global_mean(det, p03)
    plot_strategy_boxplots(det, p03)

    p04 = output_root / "04_rag"
    plot_rag_retriever_by_strategy(agg, p04)

    p05 = output_root / "05_readability"
    plot_readability_separate(agg, p05)

    p06 = output_root / "06_rankings"
    plot_top_configurations(agg, p06)
    plot_pe_vs_rag_split(agg, p06)

    print("Done (publication figure suite).")
