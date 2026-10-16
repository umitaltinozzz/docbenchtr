from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import EMBEDDING_MODEL_REVISION, RESULTS_DIR
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, Retriever

KS = (1, 3, 5)


def compute_metrics(ranked_pages: list[int], relevant_pages: list[int]) -> dict[str, float]:
    """Compute page-level recall at k and reciprocal rank from a full ranking."""
    relevant = set(relevant_pages)
    if not relevant:
        raise ValueError("At least one relevant page is required")
    first_rank = next((rank for rank, page in enumerate(ranked_pages, start=1) if page in relevant), None)
    metrics = {f"recall@{k}": len(set(ranked_pages[:k]) & relevant) / len(relevant) for k in KS}
    metrics["reciprocal_rank"] = 1 / first_rank if first_rank else 0.0
    return metrics


def run_evaluation(
    pages: list[dict[str, Any]], questions: list[dict[str, Any]],
    extra_retrievers: list[Retriever] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bm25 = BM25Retriever(pages, stem=False)
    bm25_stem = BM25Retriever(pages, stem=True)
    dense = DenseRetriever(pages)
    retrievers: list[Retriever] = [bm25, bm25_stem, dense, HybridRetriever(bm25, dense)]
    retrievers.extend(extra_retrievers or [])
    all_pages_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        all_pages_by_doc[page["document_id"]].append(page)

    rows = []
    for question in questions:
        document_id = question["document_id"]
        full_k = len(all_pages_by_doc[document_id])
        for retriever in retrievers:
            start = time.perf_counter()
            ranked_pages = retriever.retrieve(question["question"], document_id, full_k)
            latency_ms = (time.perf_counter() - start) * 1000
            metrics = compute_metrics(ranked_pages, question["source_pages"])
            rows.append({
                "method": retriever.name,
                "question_id": question["question_id"],
                "question_type": question["question_type"],
                "document_id": document_id,
                "source_pages": question["source_pages"],
                "ranked_pages": ranked_pages,
                "recall@1": metrics["recall@1"],
                "recall@3": metrics["recall@3"],
                "recall@5": metrics["recall@5"],
                "reciprocal_rank": metrics["reciprocal_rank"],
                "latency_ms": latency_ms,
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "results.csv").open("w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "method", "question_id", "question_type", "document_id", "source_pages",
            "ranked_pages", "recall@1", "recall@3", "recall@5", "reciprocal_rank", "latency_ms",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "source_pages": json.dumps(row["source_pages"]),
                             "ranked_pages": json.dumps(row["ranked_pages"])})
    (RESULTS_DIR / "per_query.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, retrievers


def summarize(
    rows: list[dict[str, Any]], questions: list[dict[str, Any]],
    methods_status: list[dict[str, str]], pages: list[dict[str, Any]],
    model_device: str, excluded_questions: int = 0,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["method"]].append(row)
        by_type[row["method"]][row["question_type"]].append(row)
    summary_methods = []
    for status in methods_status:
        method = status["method"]
        method_rows = grouped.get(method, [])
        if not method_rows:
            summary_methods.append({**status, "question_count": 0, "overall": None, "by_question_type": {}})
            continue
        overall = {
            key: statistics.fmean(row[key] for row in method_rows)
            for key in ("recall@1", "recall@3", "recall@5", "reciprocal_rank", "latency_ms")
        }
        summary_methods.append({
            **status,
            "question_count": len(method_rows),
            "overall": overall,
            "by_question_type": {
                question_type: {
                    "question_count": len(type_rows),
                    **{
                        key: statistics.fmean(row[key] for row in type_rows)
                        for key in ("recall@1", "recall@3", "recall@5", "reciprocal_rank", "latency_ms")
                    },
                }
                for question_type, type_rows in sorted(by_type[method].items())
            },
        })
    doc_page_counts: dict[str, int] = defaultdict(int)
    for page in pages:
        doc_page_counts[page["document_id"]] += 1
    return {
        "benchmark": "DocBenchTR public pilot",
        "dataset_label": "pilot set, auto-drafted from source text, span-verified; not yet human-reviewed",
        "retrieval_only": True,
        "generation_and_faithfulness": "out of scope",
        "num_documents": len(doc_page_counts),
        "num_questions": len(questions),
        "num_excluded_questions": excluded_questions,
        "question_type_counts": {
            question_type: sum(q["question_type"] == question_type for q in questions)
            for question_type in sorted({q["question_type"] for q in questions})
        },
        "pages_per_document": dict(doc_page_counts),
        "embedding_model": "intfloat/multilingual-e5-base",
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "embedding_device": model_device,
        "page_numbering": "one-based physical PDF page index",
        "metrics": "page-level Recall@1/3/5, MRR over full returned ranking, and per-query latency",
        "methods": summary_methods,
    }


def write_summary(summary: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
