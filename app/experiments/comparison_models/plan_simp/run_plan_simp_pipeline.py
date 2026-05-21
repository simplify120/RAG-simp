"""
Export dataset items to plan_simp CSVs, run encode_contexts + generate.py dynamic
(planner + simplifier), then write plan_simp_raw_results.jsonl.

Default split is the 40-item BGE test set (same random seed/sample as RAG test-set builders).
Use --split complement for all other items with text_adv (e.g. 149 when the corpus has 189).

Phase 1 (brief DB access): fetch items + references, build CSVs + manifest, disconnect.
Phase 2 (GPU / no DB): encode contexts, run dynamic generation, merge to JSONL.

Usage:
    python -m app.experiments.comparison_models.plan_simp.run_plan_simp_pipeline
    python -m app.experiments.comparison_models.plan_simp.run_plan_simp_pipeline --export-only
    python -m app.experiments.comparison_models.plan_simp.run_plan_simp_pipeline \\
        --split complement --output-dir app/experiments/comparison_models/plan_simp/outputs_complement
    python -m app.experiments.comparison_models.plan_simp.run_plan_simp_pipeline \\
        --clf-model liamcripwell/pgdyn-plan --simp-model liamcripwell/pgdyn-simp --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple
from uuid import UUID

import pandas as pd

# Repo root for `app.*` imports
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from app.db.session import SessionLocal
from app.models.dataset import DatasetItem
from app.experiments.comparison_models.plan_simp.plan_simp_io import clean_plan_simp_output

# Mirrors app.experiments.RAG.bge.build_embedding_index_test_set (40 items, seed 42) without
# importing that module (it loads SentenceTransformer at import time).
TEST_SAMPLE_SIZE = 40
RANDOM_SEED = 42


def _all_items_with_text_adv(db):
    return (
        db.query(DatasetItem.item_id, DatasetItem.text_adv)
        .filter(DatasetItem.text_adv.isnot(None))
        .all()
    )


def _bge_test_item_ids(all_items: List[Tuple[UUID, str]]) -> set[UUID]:
    random.seed(RANDOM_SEED)
    k = min(TEST_SAMPLE_SIZE, len(all_items))
    return {item[0] for item in random.sample(all_items, k)}


def get_test_items_bge_split(db):
    """The 40-item (or fewer) BGE test split; same IDs as RAG test-set scripts."""
    all_items = _all_items_with_text_adv(db)
    test_ids = _bge_test_item_ids(all_items)
    return [item for item in all_items if item[0] in test_ids]


def get_complement_items_bge_split(db):
    """All items with text_adv that are not in the BGE test split."""
    all_items = _all_items_with_text_adv(db)
    test_ids = _bge_test_item_ids(all_items)
    return [item for item in all_items if item[0] not in test_ids]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PLAN_REPO_ROOT = Path(__file__).resolve().parent
ENCODE_SCRIPT = PLAN_REPO_ROOT / "plan_simp" / "scripts" / "encode_contexts.py"
GENERATE_SCRIPT = PLAN_REPO_ROOT / "plan_simp" / "scripts" / "generate.py"

DEFAULT_CLF = "liamcripwell/pgdyn-plan"
DEFAULT_SIMP = "liamcripwell/pgdyn-simp"


def _ensure_nltk_punkt() -> None:
    import nltk

    for name in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{name}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", name)
            nltk.download(name, quiet=True)


def _sentences(text: str) -> List[str]:
    import nltk

    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def export_test_items_to_csv(
    output_dir: Path,
    split: Literal["test", "complement"] = "test",
) -> Tuple[Path, Path, Path]:
    """
    Load items from DB for the chosen split, write plan_simp_docs.csv, plan_simp_sents.csv,
    manifest.json. Returns paths (docs_csv, sents_csv, manifest_json).
    """
    _ensure_nltk_punkt()
    output_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        all_items = _all_items_with_text_adv(db)
        test_ids = _bge_test_item_ids(all_items)
        if split == "test":
            rows = [item for item in all_items if item[0] in test_ids]
            if len(rows) != TEST_SAMPLE_SIZE and len(all_items) >= TEST_SAMPLE_SIZE:
                logger.warning(
                    "Expected %s test items, got %s",
                    TEST_SAMPLE_SIZE,
                    len(rows),
                )
        elif split == "complement":
            rows = [item for item in all_items if item[0] not in test_ids]
            expected = len(all_items) - min(TEST_SAMPLE_SIZE, len(all_items))
            if len(rows) != expected:
                logger.warning(
                    "Complement size unexpected: got %s, expected %s (non-test items)",
                    len(rows),
                    expected,
                )
        else:
            raise ValueError(f"Unknown split: {split!r}")

        item_ids = [r[0] for r in rows]
        extras = (
            db.query(DatasetItem.item_id, DatasetItem.text_ele)
            .filter(DatasetItem.item_id.in_(item_ids))
            .all()
        )
        ele_map: Dict[UUID, str | None] = {e.item_id: e.text_ele for e in extras}

        manifest_items: List[Dict[str, Any]] = []
        docs_rows: List[Dict[str, Any]] = []
        sents_rows: List[Dict[str, Any]] = []

        for idx, (item_id, text_adv) in enumerate(rows):
            pair_id = f"p{idx:04d}"
            text_ele = ele_map.get(item_id)
            sents = _sentences(text_adv)
            if not sents and (text_adv or "").strip():
                sents = [(text_adv or "").strip()]
            if not sents:
                logger.warning("Item %s is empty; skipping", item_id)
                continue

            n = len(sents)
            complex_doc = " <s> ".join(sents)

            docs_rows.append(
                {
                    "title": f"item_{pair_id}",
                    "pair_id": pair_id,
                    "complex": complex_doc,
                }
            )

            manifest_items.append(
                {
                    "item_id": str(item_id),
                    "pair_id": pair_id,
                    "text_adv": text_adv,
                    "text_ele": text_ele,
                    "num_sentences": n,
                }
            )

            for sent_id, sent_text in enumerate(sents):
                doc_pos = (sent_id + 1) / n
                doc_quint = 2
                simple_placeholder = "['']"
                sents_rows.append(
                    {
                        "title": f"item_{pair_id}",
                        "pair_id": pair_id,
                        "sent_id": sent_id,
                        "complex": sent_text,
                        "label": "ignore",
                        "simple": simple_placeholder,
                        "simp_sent_id": sent_id,
                        "doc_pos": doc_pos,
                        "doc_quint": doc_quint,
                        "doc_len": n,
                    }
                )

        docs_csv = output_dir / "plan_simp_docs.csv"
        sents_csv = output_dir / "plan_simp_sents.csv"
        manifest_path = output_dir / "plan_simp_manifest.json"

        pd.DataFrame(docs_rows).to_csv(docs_csv, index=False)
        pd.DataFrame(sents_rows).to_csv(sents_csv, index=False)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {"version": 1, "split": split, "items": manifest_items},
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Wrote %s docs, %s sentence rows, manifest with %s items",
            len(docs_rows),
            len(sents_rows),
            len(manifest_items),
        )
        return docs_csv, sents_csv, manifest_path
    finally:
        db.close()


def _plan_simp_env() -> dict:
    env = os.environ.copy()
    root = str(PLAN_REPO_ROOT)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_encode_contexts(
    docs_csv: Path,
    context_dir: Path,
    device: str,
) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ENCODE_SCRIPT),
        f"--data={docs_csv.resolve()}",
        f"--save_dir={context_dir.resolve()}",
        f"--x_col=complex",
        f"--id_col=pair_id",
        f"--device={device}",
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PLAN_REPO_ROOT), env=_plan_simp_env(), check=True)


def run_generate_dynamic(
    sents_csv: Path,
    out_csv: Path,
    context_dir: Path,
    temp_dir: Path,
    clf_model: str,
    simp_model: str,
    reading_lvl: int,
    device: str,
) -> None:
    # generate.py dynamic() requires temp_dir to NOT exist (unless results_cache is set):
    # it does os.makedirs(temp_dir) itself; an existing empty dir raises FileExistsError.
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    cmd = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "dynamic",
        f"--model_ckpt={simp_model}",
        f"--test_file={sents_csv.resolve()}",
        f"--out_file={out_csv.resolve()}",
        f"--clf_model_ckpt={clf_model}",
        f"--doc_id_col=pair_id",
        f"--context_doc_id=pair_id",
        f"--context_dir={context_dir.resolve()}",
        f"--reading_lvl={reading_lvl}",
        f"--temp_dir={temp_dir.resolve()}",
        f"--device={device}",
        f"--save_rate=5",
    ]
    logger.info("Running (this may take a long time): %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PLAN_REPO_ROOT), env=_plan_simp_env(), check=True)


def merge_dynamic_csv_to_jsonl(
    manifest_path: Path,
    dynamic_csv: Path,
    jsonl_path: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair_to_meta = {m["pair_id"]: m for m in manifest["items"]}

    df = pd.read_csv(dynamic_csv, keep_default_na=False)
    if "pred" not in df.columns:
        raise ValueError(f"Missing 'pred' column in {dynamic_csv}")

    lines: List[str] = []
    for pair_id, group in df.groupby("pair_id", sort=False):
        meta = pair_to_meta.get(str(pair_id).strip())
        if not meta:
            logger.warning("Unknown pair_id in output: %s", pair_id)
            continue
        group = group.sort_values("sent_id")
        sentence_preds = [
            clean_plan_simp_output(str(x)) for x in group["pred"].tolist()
        ]
        output_text = " ".join(sentence_preds).strip()
        rec = {
            "item_id": meta["item_id"],
            "pair_id": meta["pair_id"],
            "input_text": meta["text_adv"],
            "reference_text": meta["text_ele"],
            "output_text": output_text,
            "sentence_preds": sentence_preds,
        }
        lines.append(json.dumps(rec, ensure_ascii=False))

    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s records to %s", len(lines), jsonl_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan-Simp full pipeline (export + vendored generate.py)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLAN_REPO_ROOT / "outputs",
        help="Directory for CSVs, context embeds, JSONL",
    )
    parser.add_argument("--export-only", action="store_true", help="Only write CSVs + manifest from DB")
    parser.add_argument("--clf-model", type=str, default=DEFAULT_CLF)
    parser.add_argument("--simp-model", type=str, default=DEFAULT_SIMP)
    parser.add_argument("--reading-lvl", type=int, default=3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if _cuda_available() else "cpu",
        help="cuda or cpu (encode + generate)",
    )
    parser.add_argument(
        "--skip-encode",
        action="store_true",
        help="Reuse existing plan_simp_context_embeds in output-dir",
    )
    parser.add_argument(
        "--split",
        choices=("test", "complement"),
        default="test",
        help="test: 40-item BGE sample; complement: all other items with text_adv",
    )
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    docs_csv, sents_csv, manifest_path = export_test_items_to_csv(out, split=args.split)
    if args.export_only:
        logger.info("Export complete (--export-only).")
        return

    context_dir = out / "plan_simp_context_embeds"
    temp_dir = out / "plan_simp_dynamic_temp_embeds"
    raw_csv = out / "plan_simp_dynamic_output.csv"
    jsonl_out = out / "plan_simp_raw_results.jsonl"

    if not args.skip_encode:
        if context_dir.exists():
            shutil.rmtree(context_dir)
        run_encode_contexts(docs_csv, context_dir, args.device)
    else:
        logger.info("Skipping encode_contexts (--skip-encode)")

    run_generate_dynamic(
        sents_csv,
        raw_csv,
        context_dir,
        temp_dir,
        args.clf_model,
        args.simp_model,
        args.reading_lvl,
        args.device,
    )
    merge_dynamic_csv_to_jsonl(manifest_path, raw_csv, jsonl_out)
    logger.info("Done. Raw results: %s", jsonl_out)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
