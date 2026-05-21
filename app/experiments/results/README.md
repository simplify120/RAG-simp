# Final experiment results

Reproducible pipeline for aggregated metrics, paper-ready tables, and publication figures for the text-simplification experiments.

## Layout

| Path | Contents |
|------|----------|
| `aggregated/` | `results_all_experiments.csv` (means/SDs by model × strategy × phase), `results_detailed.csv` (per-item rows, includes LENS) |
| `final_tables/` | `table_01_pe_by_model_strategy.csv`, `table_02_rag_comparison.csv`, `table_03_best_configs.csv` |
| `figures/` | Subfolders `01_phase_evolution/` … `06_rankings/` (PNG + PDF per plot); see below |
| `publication_figures.py` | Figure suite implementation (called by `scripts/generate_figures.py`) |
| `phase_utils.py` | Shared mapping from DB `description` / prompt `version` → phase labels |
| `scripts/` | Runnable entry points |

## Prerequisites

- Python environment with project dependencies (`pandas`, `matplotlib`, `seaborn`, SQLAlchemy, etc.).
- Database URL: set `DATABASE_URL` or rely on the default in `app/core/config.py` (typically SQLite `llm_simplification.db` in the working directory, or PostgreSQL for full runs).
- API keys are **not** required for aggregation or plotting (read-only DB access).

## Regenerate everything

From the repository root:

```bash
python app/experiments/results/scripts/run_all.py
```

This runs:

1. **`collect_results.py`** — queries `Evaluation` + `PromptResult` + `PromptVersion` + `Prompt`, adds `phase` / `experiment_group` / `retrieval`, excludes **`sonar-pro`** (use **sonar** only), writes `aggregated/*.csv`.
2. **`generate_tables.py`** — builds `final_tables/*.csv` from the aggregated file.
3. **`generate_figures.py`** — builds the publication figure suite under `figures/` (subfolders). If `aggregated/*.csv` is missing, falls back to live DB queries. **Baselines** are loaded from:
   - `app/experiments/comparison_models/t5_model/outputs/t5_full_dataset_analysis.csv`
   - `app/experiments/comparison_models/plan_simp/outputs/plan_simp_full_dataset_analysis.csv`

### Skip database export

If `aggregated/results_all_experiments.csv` and `results_detailed.csv` already exist:

```bash
python app/experiments/results/scripts/run_all.py --skip-collect
```

### Run steps individually

```bash
python app/experiments/results/scripts/collect_results.py
python app/experiments/results/scripts/generate_tables.py
python app/experiments/results/scripts/generate_figures.py
```

Optional: `--output-dir` / `--aggregated-dir` (see `--help` on each script).

## Figure suite (under `figures/`)

| Folder | Contents |
|--------|----------|
| **`01_phase_evolution/`** | `01_sari_by_phase`, `02_bertscore_by_phase` — one metric per figure; compact x-axis labels; dashed reference lines for **T5-Large** and **Plan-Simp** (full-dataset CSVs). |
| **`02_baselines/`** | `01_sari_vs_baselines`, `02_bertscore_vs_baselines`, `03_lens_vs_baselines`, `04_bleu_vs_baselines` — one metric per figure (mean ± SD bars; value labels with margin past error bars). |
| **`03_strategy/`** | PE v2 heatmaps (SARI, BERTScore), strategy×model heatmaps (mean over phases), boxplots by strategy. |
| **`04_rag/`** | One file per prompting strategy: SARI by model × retriever (OpenAI v3 / BGE / E5 full). |
| **`05_readability/`** | `01_fkgl_delta_by_phase`, `02_fre_delta_by_phase` — separate figures. |
| **`06_rankings/`** | Top configurations; best PE vs best RAG split into **SARI** and **BERTScore** figures. |

This mirrors the spirit of [`app/experiments/visualization/outputs/figures/`](../visualization/outputs/figures) (separate topics per folder) with a publication-focused layout and supervised-model baselines.

## Relation to existing code

- **Aggregation queries** reuse [`app/experiments/analysis/aggregate_metrics.py`](../analysis/aggregate_metrics.py) (`aggregate_by_model_and_strategy`) and [`load_detailed_results`](../visualization/data_loader.py) for per-item metrics (including LENS).
- **Phase labels** align with [`app/experiments/visualization/config.py`](../visualization/config.py) (`DESCRIPTION_TO_PHASE`, `PHASE_ORDER`, `PHASE_AXIS_SHORT`). The corrected spelling `step 2 - RAG top k=3 with upgraded prompt` is supported in config and in collection.
