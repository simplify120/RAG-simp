# Automatic Text Simplification

This repository contains the code and experiment artifacts for a study on **automatic text simplification** using large language models (LLMs). The project implements and compares **prompt engineering** (zero-shot, structured, and constraint-based prompting) and **retrieval-augmented generation (RAG)** with multiple embedding backends. Generated outputs are evaluated with standard simplification metrics and compared against supervised baselines (T5-Large and Plan-Simp).

---

## 1. Project Overview

The system:

- Stores a multi-level text corpus (advanced, intermediate, elementary) and prompt templates in a relational database.
- Runs LLM simplification experiments across multiple models and prompting strategies.
- Augments prompts with retrieved similar examples (RAG) using OpenAI, E5, or BGE embeddings.
- Computes automatic evaluation metrics (SARI, BLEU, BERTScore, FKGL, FRE, Perplexity, LENS).
- Aggregates results into publication-ready tables and figures.

The experiments use a corpus of **189 texts** (advanced-level input with aligned elementary references). **Prompt engineering** runs all three strategy types (**zeroshot**, **structured**, **constraint**) on the **full 189-text corpus**. **RAG** experiments use **k-fold cross-validation** over five fixed splits (`random.seed(42)`): one holdout set of **40 texts** and four evaluation folds of **37, 37, 37, and 38 texts** on the remaining 149 items. RAG retrieves **top-k = 3** similar training examples by default.

A FastAPI backend is included for API access; the simplification endpoint is currently a stub.

---

## 2. Repository Structure

```
.
├── app/
│   ├── api/                    # FastAPI routes (health, simplification stub)
│   ├── core/                   # Settings and logging (config.py)
│   ├── db/                     # SQLAlchemy engine and session
│   ├── models/                 # ORM models (dataset, prompts, evaluations, embeddings)
│   ├── services/               # Business logic
│   ├── main.py                 # FastAPI entry point
│   └── experiments/
│       ├── llm_comparison/     # Prompt-engineering runners (OpenAI, Gemini, Claude, Sonar, Ollama)
│       ├── RAG/                # RAG pipelines (openai/, e5/, bge/) and CV splits
│       ├── evaluation/         # Metric calculators and evaluate_run orchestrator
│       ├── comparison_models/  # T5-Large and Plan-Simp baselines
│       ├── analysis/           # Ad-hoc metric aggregation and visualization helpers
│       ├── visualization/      # Exploratory figure suite
│       └── results/            # Aggregated CSVs, final tables, publication figures
├── alembic/                    # Database migrations (incremental)
├── scripts/
│   └── populate_dataset_items.py   # Load corpus from CSV files into the database
├── Docker/                     # Container definition
├── docker-compose.yml          # PostgreSQL + app stack
├── requirements.txt            # Core + full dependencies
├── requirements-runtime.txt      # Minimal API runtime dependencies
├── requirements-experiments.txt  # LLM clients, metrics, and plotting extras
└── .env.example                # Environment variable template
```

**Key result artifacts (included in the repo):**

| Path | Contents |
|------|----------|
| `app/experiments/results/aggregated/` | Per-experiment and per-item metric summaries |
| `app/experiments/results/final_tables/` | Paper-ready comparison tables (PE, RAG, best configs) |
| `app/experiments/results/figures/` | Publication figure suite (PNG/PDF) |
| `app/experiments/comparison_models/*/outputs/` | Baseline model analysis CSVs |

See also `app/experiments/results/README.md` for the results pipeline layout.

---

## 3. Setup Instructions

### Python version

The Docker image targets **Python 3.11** (`Docker/Dockerfile`). Use Python 3.11 locally for best compatibility.

### Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

### Dependencies

**API + database only:**

```bash
pip install -r requirements-runtime.txt
```

**Full project (experiments, metrics, RAG, baselines):**

```bash
pip install -r requirements.txt
pip install -r requirements-experiments.txt
```

RAG pipelines that use local embedding models (E5, BGE) import `sentence_transformers`. Install it if not already pulled in transitively:

```bash
pip install sentence-transformers
```

Plan-Simp has its own optional environment; see `app/experiments/comparison_models/plan_simp/requirements-inference-venv-py312.txt`.

### Environment variables

Copy the template and fill in values locally (never commit secrets):

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `APP_ENV` | Environment label (`local`, `production`, etc.) |
| `DATABASE_URL` | SQLAlchemy connection string. Default in code: `sqlite:///./llm_simplification.db`. **RAG embedding storage requires PostgreSQL with pgvector.** |
| `OPENAI_API_KEY` | OpenAI API access (GPT-4o-mini, text-embedding-3-small) |
| `GEMINI_API_KEY` | Google Gemini API access |
| `PERPLEXITYAI_API_KEY` | Perplexity Sonar API access |
| `CLAUDE_API_KEY` | Anthropic Claude access (also sets `ANTHROPIC_API_KEY` for LiteLLM) |
| `CORS_ORIGINS` | Comma-separated allowed origins for the API (optional) |

Optional deployment hooks in `.env.example` (`RENDER_DEPLOY_HOOK_*`) are not required for local reproduction.

### Database

**Option A — Docker (recommended for RAG):**

```bash
docker-compose up -d db
```

Set `DATABASE_URL` to the PostgreSQL URL (see `docker-compose.yml` for default credentials). Ensure the **pgvector** extension is enabled on the database before building embedding indices.

**Option B — SQLite:** Supported for read-only aggregation if a populated database file is available. Vector RAG tables require PostgreSQL.

Apply migrations:

```bash
alembic upgrade head
```

> **Note:** Alembic migrations in this repository are **incremental** (they add evaluation tables, prompt v3 templates, etc.). A base schema with `datasets`, `dataset_items`, `prompts`, and `prompt_versions` must already exist, or be created separately. Migration `002` inserts prompt version **v3** templates for the three strategy types.

### External services

| Service | Used for |
|---------|----------|
| OpenAI API | GPT-4o-mini simplification, OpenAI embeddings |
| Google Gemini API | Gemini 2.0 Flash simplification |
| Anthropic API (via LiteLLM) | Claude Haiku 4.5 simplification |
| Perplexity API | Sonar simplification |
| Ollama (local) | Llama 3.2 via `http://localhost:11434` |
| Hugging Face Hub | T5-Large, Plan-Simp, E5, BGE, LENS, distilgpt2 (perplexity) models |

---

## 4. Running the Project

All commands below assume the repository root as the working directory and an activated virtual environment.

### Prepare / load data

1. Configure the CSV source directory in `scripts/populate_dataset_items.py` (`CSV_DIRECTORY`). Each CSV should have columns **Elementary**, **Intermediate**, and **Advanced**; one file becomes one `dataset_items` row.

2. Load the corpus:

```bash
python scripts/populate_dataset_items.py
```

3. Ensure prompt templates (`zeroshot`, `structured`, `constraint`) and their versions exist in the database. Run `alembic upgrade head` to add **v3** RAG-optimized templates.

### Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker-compose up
```

Health check: `GET /api/v1/health`

### Prompt-engineering experiments

Prompt-engineering (basic prompt) experiments run on **all 189 texts**. Each client runs all three strategies and stores rows in `prompt_results`. Active prompt versions (`is_active=True`) are used.

```bash
# Claude Haiku 4.5 — full corpus (default)
python -m app.experiments.llm_comparison.claude_client --subset all

# Perplexity Sonar — full corpus
python -m app.experiments.llm_comparison.perplexity_client --subset all

# OpenAI GPT-4o-mini
python -m app.experiments.llm_comparison.openai_client

# Google Gemini 2.0 Flash
python -m app.experiments.llm_comparison.gemini_client

# Local Llama 3.2 via Ollama (requires Ollama running)
python -m app.experiments.llm_comparison.ollama_client
```

Clients that support `--subset` accept `all` (189 texts), `pilot` (40 texts), or `remaining` (149 texts). Use `--subset all` to match the full prompt-engineering evaluation.

Stored results use description `"step 1 - simple prompt engineering"` for Claude and Sonar clients.

### RAG experiments

RAG is evaluated with **k-fold cross-validation** over the 189-text corpus. The split manifest (`rag_cv_v1.json`) defines:

| Split | Size | Role |
|-------|------|------|
| Holdout | 40 | Fixed retrieval pool (not evaluated as a CV fold) |
| Fold 0 | 37 | Evaluation fold |
| Fold 1 | 37 | Evaluation fold |
| Fold 2 | 37 | Evaluation fold |
| Fold 3 | 38 | Evaluation fold |

For each CV fold, the pipeline evaluates that fold's texts while retrieving examples from the other three complement folds plus the 40-text holdout set.

**Step 1 — Generate the CV split manifest:**

```bash
python -m app.experiments.RAG.splits.generate_cv_splits
```

**Step 2 — Build embedding indices** (149-text complement for retrieval + 40-text holdout pool):

```bash
# OpenAI embeddings (text-embedding-3-small, dim 1536)
python -m app.experiments.RAG.openai.build_embedding_index
python -m app.experiments.RAG.openai.build_embedding_index_test_set

# E5 embeddings (intfloat/e5-base-v2, dim 768)
python -m app.experiments.RAG.e5.build_embedding_index
python -m app.experiments.RAG.e5.build_embedding_index_test_set

# BGE embeddings (BAAI/bge-base-en-v1.5, dim 768)
python -m app.experiments.RAG.bge.build_embedding_index
python -m app.experiments.RAG.bge.build_embedding_index_test_set
```

**Step 3 — Run RAG CV pipelines** (one fold at a time, or all folds sequentially):

```bash
python -m app.experiments.RAG.openai.run_rag_pipeline_cv \
  --model claude-haiku-4-5 \
  --description "step 3 - RAG cv top k=3" \
  --top-k 3 \
  --fold 0

# All four folds (37 + 37 + 37 + 38 texts) in one run:
python -m app.experiments.RAG.openai.run_rag_pipeline_cv \
  --model sonar \
  --description "step 2 - RAG top k=3" \
  --top-k 3 \
  --all-folds
```

Equivalent CV runners exist under `app/experiments/RAG/e5/` and `app/experiments/RAG/bge/`.

Available `--model` values: `openai`, `gemini`, `llama`, `sonar`, `sonar-pro`, `claude-haiku-4-5`. Use `--strategy zeroshot|structured|constraint|all` (default: `all`).

Non-CV RAG runners (`run_rag_pipeline`) are also available for single-split runs on the 40-text holdout set only.

### Evaluation

Run all metrics for a given experiment description:

```bash
python -m app.experiments.evaluation.evaluate_run \
  --description "step 1 - simple prompt engineering" \
  --model-name claude-haiku-4-5
```

Run individual metrics:

```bash
python -m app.experiments.evaluation.sari.calculate_sari --description "..."
python -m app.experiments.evaluation.bleu.calculate_bleu --description "..."
python -m app.experiments.evaluation.bertscore.calculate_bert --description "..."
python -m app.experiments.evaluation.fkgl.calculate_fkgl --description "..."
python -m app.experiments.evaluation.fre.calculate_fre --description "..."
python -m app.experiments.evaluation.perplexity.calculate_perplexity --description "..."
python -m app.experiments.evaluation.lens.calculate_lens --description "..."
```

Add `--force-recalculate` to recompute stored metrics. Add `--model-name` to filter by model.

### Baseline models

```bash
# T5-Large (local GPU recommended)
python -m app.experiments.comparison_models.t5_model.run_t5_large_text_simplification
python -m app.experiments.comparison_models.t5_model.analyze_t5_test_set --split all

# Plan-Simp (separate venv recommended; GPU recommended)
python -m app.experiments.comparison_models.plan_simp.run_plan_simp_pipeline
python -m app.experiments.comparison_models.plan_simp.analyze_plan_simp_test_set --split all
```

### Generate / export result tables and figures

**Publication pipeline** (aggregated CSVs → tables → figures):

```bash
python app/experiments/results/scripts/run_all.py
```

Skip database export if aggregated CSVs already exist:

```bash
python app/experiments/results/scripts/run_all.py --skip-collect
```

Individual steps:

```bash
python app/experiments/results/scripts/collect_results.py
python app/experiments/results/scripts/generate_tables.py
python app/experiments/results/scripts/generate_figures.py
```

**Exploratory visualization suite:**

```bash
python -m app.experiments.visualization.run_all
```

---

## 5. Evaluation

Metrics are computed against the **elementary-level reference** (`text_ele`) where applicable.

| Metric | What it measures (in this project) |
|--------|-------------------------------------|
| **SARI** | Similarity-based metric for simplification: rewards appropriate word additions, deletions, and paraphrases relative to source and reference. Primary quality metric. |
| **BLEU** | N-gram overlap between simplified output and reference. Higher indicates closer lexical match to the gold simplification. |
| **BERTScore** | Contextual embedding similarity (F1) between output and reference. Captures semantic similarity beyond surface overlap. |
| **FKGL** (Δ) | Change in Flesch–Kincaid Grade Level from input to output. Negative delta indicates reduced reading grade level (simpler text). |
| **FRE** (Δ) | Change in Flesch Reading Ease from input to output. Positive delta indicates easier-to-read text. |
| **Perplexity** | Fluency of the generated output under a fixed language model (`distilgpt2`). Lower perplexity suggests more natural, grammatical text. |
| **LENS** | Learned evaluation metric trained on human simplification judgments (`davidheineman/lens`). Higher scores indicate better simplification quality. |

Phase labels and metric metadata are defined in `app/experiments/visualization/config.py`.

---

## 6. Reproducing Results

### Quick reproduction (tables and figures only)

Pre-computed aggregated results are included under `app/experiments/results/aggregated/`. To regenerate tables and figures **without** LLM API calls or a populated database:

```bash
pip install pandas matplotlib seaborn
python app/experiments/results/scripts/run_all.py --skip-collect
```

Outputs appear in `app/experiments/results/final_tables/` and `app/experiments/results/figures/`.

### Full reproduction (from scratch)

1. Set up Python 3.11, install dependencies, and configure `.env` with API keys and a PostgreSQL + pgvector `DATABASE_URL`.
2. Start PostgreSQL (`docker-compose up -d db`) and run `alembic upgrade head`.
3. Ensure base database schema and prompt records exist; load the text corpus via `scripts/populate_dataset_items.py` (update the CSV path first).
4. **Prompt engineering:** Run LLM clients in `app/experiments/llm_comparison/` on the **full 189-text corpus** (`--subset all` where supported).
5. **RAG:** Generate the CV split manifest, build embedding indices, then run CV pipelines (`run_rag_pipeline_cv`) for OpenAI, E5, and BGE retrievers across all four folds (37 + 37 + 37 + 38 texts).
6. **Evaluation:** Run `python -m app.experiments.evaluation.evaluate_run` per experiment description (and `--model-name` where needed).
7. **Baselines (optional):** Run T5-Large and Plan-Simp pipelines.
8. **Aggregate:** Run `python app/experiments/results/scripts/run_all.py`.

Experiment phases tracked in the paper correspond to database `description` values and prompt `version` fields:

| Phase key | Typical `description` / notes |
|-----------|-------------------------------|
| PE v1 / v2 | `"step 1 - simple prompt engineering"` (split by prompt version v1 vs v2) |
| OpenAI RAG v2 | `"step 2 - RAG top k=3"` |
| OpenAI RAG v3 | `"step 2 - RAG top k=3 with upgraded prompt"` |
| E5 RAG v3 | `"E5-RAG-full"` |
| BGE RAG v3 | `"BGE-RAG-full"` |

Full end-to-end reproduction requires external LLM API access and substantial compute time. API keys should be supplied via environment variables, not hard-coded.

---

## 7. Configuration

| Setting | Location / default |
|---------|-------------------|
| App settings | `app/core/config.py` — reads from environment via `python-dotenv` |
| Database URL | `DATABASE_URL` env var; defaults to SQLite if unset |
| LLM model registry | `app/experiments/llm_comparison/model_registry.py` |
| RAG top-k | CLI `--top-k` (default `3`) on RAG runners |
| Corpus size | 189 texts with advanced input (`text_adv`) |
| Prompt engineering scope | All 189 texts |
| RAG evaluation split | K-fold CV: 40-text holdout + four folds of 37, 37, 37, 38 (`random.seed(42)`) |
| OpenAI embedding model | `text-embedding-3-small` (1536-dim) |
| E5 embedding model | `intfloat/e5-base-v2` (768-dim) |
| BGE embedding model | `BAAI/bge-base-en-v1.5` (768-dim) |
| T5 baseline model | `eilamc14/t5-large-text-simplification` |
| Plan-Simp models | `liamcripwell/pgdyn-plan`, `liamcripwell/pgdyn-simp` (Hugging Face) |
| LENS model | `davidheineman/lens` |
| Perplexity evaluator | `distilgpt2` (local, no API) |
| Phase / metric labels | `app/experiments/visualization/config.py` |
| CV fold manifest | `app/experiments/RAG/splits/rag_cv_v1.json` |
| Alembic | `alembic.ini` + `alembic/versions/` |

Prompt strategies: **zeroshot**, **structured**, **constraint**. Prompt versions **v1**, **v2**, and **v3** (v3 added by migration `002` for RAG-optimized templates).

---

## 8. Additional Notes

- Credentials and database connection strings must be provided through **environment variables** (`.env` file, not committed).
- Pre-computed result CSVs and figures are included so outcomes can be inspected without re-running costly LLM experiments.
- The original dataset CSV files are **not** bundled; loading them requires updating `CSV_DIRECTORY` in `scripts/populate_dataset_items.py`.

---

## 9. Limitations

- **External LLM dependency:** Most experiments require paid API access (OpenAI, Gemini, Anthropic, Perplexity) or a locally running Ollama instance for Llama 3.2.
- **API rate limits:** RAG CV pipelines include pacing and retry logic; strict provider rate limits may require increasing `--sleep-seconds`.
- **PostgreSQL + pgvector required for RAG:** Embedding storage and cosine-distance retrieval depend on pgvector; SQLite is insufficient for index building.
- **Incomplete base schema in migrations:** Alembic scripts assume existing core tables and prompt seed data. A database snapshot or manual schema setup may be required.
- **Compute cost and runtime:** Full reproduction across all models, strategies, and RAG retrievers involves thousands of LLM calls. Baselines (T5, Plan-Simp) and local embedding models benefit from GPU access.
- **Dataset availability:** The aligned multi-level corpus must be obtained separately; it is not included in this repository.
- **First-time model downloads:** Hugging Face models (T5, E5, BGE, LENS, distilgpt2) and Plan-Simp checkpoints are downloaded on first use and require network access and disk space.
- **API endpoint stub:** The FastAPI simplification route returns HTTP 501; the research focus is on the experiment pipeline, not production inference.
