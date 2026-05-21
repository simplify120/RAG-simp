# RAG Pipeline - How to Run

## Prerequisites

- Embeddings built: run `build_embedding_index.py` and `build_embedding_index_test_set.py`
- DB migration: `alembic upgrade head`
- API key for chosen model in `.env` (e.g. `GEMINI_API_KEY`, `OPENAI_API_KEY`)

## Command

```bash
python -m app.experiments.RAG.run_rag_pipeline --model <model> --description <description>
```

## Required

| Flag | Values |
|------|--------|
| `--model` | openai, gemini, llama, sonar, sonar-pro |
| `--description` | Any string (e.g. `RAG-top3-gemini`) |

## Optional

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k` | 3 | Similar chunks to retrieve |
| `--strategy` | all | zeroshot, structured, constraint, or all |
| `--limit` | none | Limit test items (e.g. `--limit 5` for quick test) |

## Examples

```bash
python -m app.experiments.RAG.run_rag_pipeline --model gemini --description "RAG-top3-gemini"
python -m app.experiments.RAG.run_rag_pipeline --model openai --description "RAG-top3-openai" --strategy zeroshot --limit 5
```

---

## How It Works

For each of the 40 test items:

1. **Retrieve** – Uses the item's stored embedding to find the top-k most similar texts in the 149 corpus items (cosine similarity).
2. **Build prompt** – Puts template + target text first, then examples below ("Here are some similar examples to the following text:").
3. **Generate** – Sends the full prompt to the LLM and gets the simplified output.
4. **Store** – Saves the result in `prompt_results` with `description`, `model_name`, etc.

With `--strategy all`, each item is processed 3 times (once per strategy). Results are stored in `prompt_results` and can be evaluated with the same metrics (SARI, BLEU, etc.) as the baseline runs.

## Evaluate RAG Results

```bash
# Run all metrics for RAG results only
python -m app.experiments.evaluation.evaluate_run --description "step 2 - RAG top k=3"

# Or run individual metrics
python -m app.experiments.evaluation.sari.calculate_sari --description "step 2 - RAG top k=3"

# Aggregate and compare (step 1 vs step 2 are kept separate)
python -m app.experiments.analysis.aggregate_metrics --description "step 2 - RAG top k=3"
```
