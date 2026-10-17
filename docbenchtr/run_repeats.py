"""Run the seven run-3 methods with resumable, per-question checkpoints.

python -m docbenchtr.run_repeats --repeats 3 --output-root PATH [--resume]
Model/context settings are process environment variables; see README Next run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    DOCUMENTS, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT,
    PAGEINDEX_INDEX_MODEL, MODEL_DIGESTS, RAW_DIR, ROOT, model_slug,
)
from .documents import download_documents, extract_pdf_pages
from .cache import prepare_cache
from .evaluate import evaluate_question, summarize, write_rows, write_summary
from .questions import draft_questions
from .repeats import aggregate_repeats
from .rerank import HybridLLMRerankRetriever
from .retrievers import (
    BM25Retriever, DenseRetriever, HybridRetriever, PageIndexRetriever,
    PageIndexLLMTreeSearchRetriever, ollama_status,
)
from .run_pilot import _pipeline_run_lock


def config_output_dir(root):
    return (Path(root) / f"index_{model_slug(PAGEINDEX_INDEX_MODEL)}" /
            f"search_{model_slug(OLLAMA_MODEL)}" / f"ctx{OLLAMA_NUM_CTX}_out{OLLAMA_NUM_PREDICT}")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_repeat(output_dir, identity, resume):
    output_dir = Path(output_dir)
    config_path = output_dir / "configuration.json"
    if config_path.exists():
        if not resume:
            raise ValueError(f"Results already exist: {output_dir}; use --resume or a new root")
        if json.loads(config_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Resume configuration differs from the saved run")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Nonempty output without configuration: {output_dir}")


def _initialize_repeat(output_dir, identity, resume):
    _validate_repeat(output_dir, identity, resume)
    if not (output_dir / "configuration.json").exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "configuration.json", identity)


def evaluate_repeat(retrievers, questions, page_counts, output_dir, run_index,
                    identity, *, resume=False):
    """Resume only an identical protocol; failed rows remain failed, not retried."""
    output_dir = Path(output_dir)
    _initialize_repeat(output_dir, identity, resume)
    path = output_dir / "per_query.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    seen = {(r["method"], r["question_id"]) for r in rows}
    allowed = {(r.name, q["question_id"]) for r in retrievers for q in questions}
    if len(seen) != len(rows) or not seen <= allowed or any(r.get("run_index") != run_index for r in rows):
        raise ValueError("Invalid checkpoint row identities")
    for number, question in enumerate(questions, 1):
        for retriever in retrievers:
            if (retriever.name, question["question_id"]) in seen:
                continue
            print(f"[repeat {run_index}] [{number}/{len(questions)}] {question['question_id']} | {retriever.name} START", flush=True)
            row = evaluate_question(retriever, question, page_counts[question["document_id"]])
            row.update(run_index=run_index, index_model=PAGEINDEX_INDEX_MODEL, search_model=OLLAMA_MODEL)
            rows.append(row)
            write_rows(rows, output_dir)
            print(f"[repeat {run_index}] [{number}/{len(questions)}] {question['question_id']} | {retriever.name} {row['status']} R@5={row['recall@5']} {row['latency_ms'] or 0:.0f} ms", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = config_output_dir(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    for model in dict.fromkeys([PAGEINDEX_INDEX_MODEL, OLLAMA_MODEL]):
        reachable, reason = ollama_status(model)
        if not reachable:
            raise RuntimeError(reason)
    sources = download_documents()
    pages = extract_pdf_pages(sources)
    questions, dropped = draft_questions(pages)
    counts = Counter(p["document_id"] for p in pages)
    cache = prepare_cache()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = {"commit": commit, "index_model": PAGEINDEX_INDEX_MODEL,
                "model_digests": MODEL_DIGESTS,
                "search_model": OLLAMA_MODEL, "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT, "temperature": 0,
                "repeats": args.repeats,
                "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "docbenchtr").glob("*.py"))},
                "questions_sha256": hashlib.sha256(json.dumps(questions, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "pages_sha256": hashlib.sha256(json.dumps(pages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    # Fail before expensive model construction if resuming a different protocol.
    for run in range(1, args.repeats + 1):
        _validate_repeat(output / f"repeat_{run:03d}", identity, args.resume)
    bm25, stem = BM25Retriever(pages), BM25Retriever(pages, stem=True)
    dense = DenseRetriever(pages)
    hybrid = HybridRetriever(bm25, dense)
    agent = PageIndexRetriever({d["document_id"]: str(RAW_DIR / d["filename"]) for d in DOCUMENTS},
                               str(cache), instruction_variant="default")
    try:
        if len(agent.doc_ids) != len(DOCUMENTS):
            _write_json(output / "index_failure.json", agent.results_metadata())
            raise RuntimeError("Index preparation failed; see index_failure.json (resume after fixing infrastructure)")
        tree = PageIndexLLMTreeSearchRetriever(agent, str(cache))
        rerank = HybridLLMRerankRetriever(hybrid, pages)
        retrievers = [bm25, stem, dense, hybrid, tree, agent, rerank]
        manifest = cache / "manifest.json"
        # Tree fingerprint is checked separately because building is required to obtain it.
        fingerprint = {str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(cache.glob("docs/*/tree.json"))}
        tree_record = output / "tree_fingerprints.json"
        if tree_record.exists() and json.loads(tree_record.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError("Cached trees changed since the original run")
        _write_json(tree_record, fingerprint)
        shutil.copy2(manifest, output / "index_manifest.json")
        all_rows = []
        for run in range(1, args.repeats + 1):
            target = output / f"repeat_{run:03d}"
            _initialize_repeat(target, identity, args.resume)
            # Save provenance before the first query, including an interrupted run.
            data_target = target / "data"
            data_target.mkdir(exist_ok=True)
            for source in (ROOT / "data").glob("*.json"):
                if not (data_target / source.name).exists():
                    shutil.copy2(source, data_target / source.name)
            if not (target / "index_manifest.json").exists():
                shutil.copy2(manifest, target / "index_manifest.json")
            if not (target / "index_preparation.json").exists():
                _write_json(target / "index_preparation.json", agent.results_metadata())
            rows = evaluate_repeat(retrievers, questions, counts, target, run, identity, resume=True)
            all_rows.extend(rows)
            if (target / "complete.json").exists():
                continue  # Keep original build-time and environment records on resume.
            index_metadata = json.loads((target / "index_preparation.json").read_text(encoding="utf-8"))
            summary = summarize(rows, questions, [{"method": r.name, "reason": ""} for r in retrievers],
                                pages, dense.device, len(dropped), index_metadata)
            summary.update(run_index=run, configuration=identity, reranker=rerank.results_metadata())
            write_summary(summary, target)
            _write_json(target / "complete.json", {"rows": len(rows), "finished_at": datetime.now(timezone.utc).isoformat()})
        aggregate = aggregate_repeats(all_rows, questions, args.repeats)
        aggregate["configuration"] = identity
        _write_json(output / "aggregate.json", aggregate)
        print(f"Saved repeat means, SD, majority and paired statistics: {output / 'aggregate.json'}", flush=True)
    finally:
        agent.close()
    return 0


if __name__ == "__main__":
    with _pipeline_run_lock():
        raise SystemExit(main())
