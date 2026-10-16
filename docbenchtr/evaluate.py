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
            outcome: dict[str, Any] = {
                "status": "ran",
                "error": None,
                "latency_ms": None,
                "ranked_pages": [],
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "retrieval_trace": [],
                "tool_calls": [],
                "tool_errors": [],
                "tool_call_count": None,
                "page_content_call_count": None,
                "cited_pages": [],
                "citation_error": None,
                "idle_retries": [],
                "parse_failure": None,
                "tree_search_thinking": None,
                "tree_search_node_ids": [],
            }
            document_run = getattr(retriever, "document_runs", {}).get(document_id)
            if document_run and document_run.get("status") not in ("indexed", "reused"):
                outcome["status"] = "not_run" if document_run.get("status") == "not_run" else "failed"
                outcome["error"] = document_run.get("error") or "PageIndex document was not indexed"
            else:
                start = time.perf_counter()
                try:
                    if hasattr(retriever, "retrieve_with_metadata"):
                        details = retriever.retrieve_with_metadata(
                            question["question"], document_id, full_k,
                            question_id=question["question_id"],
                        )
                        outcome.update(details)
                    else:
                        outcome["ranked_pages"] = retriever.retrieve(
                            question["question"], document_id, full_k
                        )
                except Exception as error:
                    outcome["status"] = "failed"
                    outcome["error"] = str(error)
                    outcome.update(getattr(error, "details", {}))
                finally:
                    outcome["latency_ms"] = (time.perf_counter() - start) * 1000
                if outcome["status"] == "ran" and outcome.get("tool_errors"):
                    outcome["status"] = "partial"

            ranked_pages = outcome["ranked_pages"]
            metric_values = (
                compute_metrics(ranked_pages, question["source_pages"])
                if outcome["status"] in ("ran", "partial", "zero_ranking") else {
                    "recall@1": None,
                    "recall@3": None,
                    "recall@5": None,
                    "reciprocal_rank": None,
                }
            )
            rows.append({
                "method": retriever.name,
                "question_id": question["question_id"],
                "question_type": question["question_type"],
                "document_id": document_id,
                "source_pages": question["source_pages"],
                "ranked_pages": ranked_pages,
                **metric_values,
                "status": outcome["status"],
                "error": outcome["error"],
                "latency_ms": outcome["latency_ms"],
                "input_tokens": outcome["input_tokens"],
                "output_tokens": outcome["output_tokens"],
                "total_tokens": outcome["total_tokens"],
                "retrieval_trace": outcome["retrieval_trace"],
                "tool_calls": outcome["tool_calls"],
                "tool_errors": outcome["tool_errors"],
                "tool_call_count": outcome["tool_call_count"],
                "page_content_call_count": outcome["page_content_call_count"],
                "cited_pages": outcome["cited_pages"],
                "citation_error": outcome["citation_error"],
                "idle_retries": outcome["idle_retries"],
                "parse_failure": outcome["parse_failure"],
                "tree_search_thinking": outcome["tree_search_thinking"],
                "tree_search_node_ids": outcome["tree_search_node_ids"],
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "results.csv").open("w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "method", "question_id", "question_type", "document_id", "source_pages",
            "ranked_pages", "recall@1", "recall@3", "recall@5", "reciprocal_rank",
            "status", "error", "latency_ms", "input_tokens", "output_tokens", "total_tokens",
            "retrieval_trace", "tool_calls", "tool_errors", "tool_call_count",
            "page_content_call_count", "cited_pages", "citation_error", "idle_retries",
            "parse_failure", "tree_search_thinking", "tree_search_node_ids",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "source_pages": json.dumps(row["source_pages"]),
                "ranked_pages": json.dumps(row["ranked_pages"]),
                "retrieval_trace": json.dumps(row["retrieval_trace"], ensure_ascii=False),
                "tool_calls": json.dumps(row["tool_calls"], ensure_ascii=False),
                "tool_errors": json.dumps(row["tool_errors"], ensure_ascii=False),
                "cited_pages": json.dumps(row["cited_pages"]),
                "idle_retries": json.dumps(row["idle_retries"], ensure_ascii=False),
                "tree_search_node_ids": json.dumps(row["tree_search_node_ids"], ensure_ascii=False),
            })
    (RESULTS_DIR / "per_query.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, retrievers


def summarize(
    rows: list[dict[str, Any]], questions: list[dict[str, Any]],
    methods_status: list[dict[str, str]], pages: list[dict[str, Any]],
    model_device: str, excluded_questions: int = 0,
    pageindex_metadata: dict[str, Any] | None = None,
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
        scored_rows = [
            row for row in method_rows
            if row["status"] in ("ran", "partial", "zero_ranking") and row["recall@1"] is not None
        ]
        attempted_rows = [row for row in method_rows if row["status"] in ("ran", "partial", "zero_ranking")]
        zero_ranking_rows = [row for row in attempted_rows if not row["ranked_pages"]]
        excluded_rows = [row for row in method_rows if row not in attempted_rows]
        failed_rows = [row for row in method_rows if row["status"] == "failed"]
        not_run_rows = [row for row in method_rows if row["status"] == "not_run"]
        partial_rows = [row for row in method_rows if row["status"] == "partial"]
        parse_failure_rows = [row for row in method_rows if row.get("parse_failure")]
        if not method_rows or len(scored_rows) == len(method_rows) and not partial_rows:
            method_status = "ran" if method_rows else "not_run"
        elif scored_rows:
            method_status = "partial"
        elif failed_rows:
            method_status = "failed"
        else:
            method_status = "not_run"
        latency_rows = [row for row in method_rows if row["latency_ms"] is not None]
        overall = None
        if scored_rows or latency_rows:
            overall = {
                key: statistics.fmean(row[key] for row in scored_rows) if scored_rows else None
                for key in ("recall@1", "recall@3", "recall@5", "reciprocal_rank")
            }
            overall["latency_ms"] = statistics.fmean(
                row["latency_ms"] for row in latency_rows
            ) if latency_rows else None
        token_rows = [
            row for row in method_rows
            if isinstance(row["input_tokens"], int) and isinstance(row["output_tokens"], int)
        ]
        token_usage = {
            "query_count": len(token_rows),
            "input_tokens_total": sum(row["input_tokens"] for row in token_rows),
            "output_tokens_total": sum(row["output_tokens"] for row in token_rows),
            "total_tokens": sum(row["total_tokens"] or 0 for row in token_rows),
        }
        type_summary = {}
        for question_type, type_rows in sorted(by_type[method].items()):
            type_scored = [
                row for row in type_rows
                if row["status"] in ("ran", "partial", "zero_ranking") and row["recall@1"] is not None
            ]
            type_attempted = [row for row in type_rows if row["status"] in ("ran", "partial", "zero_ranking")]
            type_zero_ranking = [row for row in type_attempted if not row["ranked_pages"]]
            type_tokens = [
                row for row in type_rows
                if isinstance(row["input_tokens"], int) and isinstance(row["output_tokens"], int)
            ]
            type_metrics = {
                "attempted_question_count": len(type_attempted),
                "question_count": len(type_scored),
                "attempted_count": len(type_attempted),
                "scored_count": len(type_scored),
                "zero_ranking_count": len(type_zero_ranking),
                "excluded_count": len(type_rows) - len(type_attempted),
                "failed_question_count": sum(row["status"] == "failed" for row in type_rows),
                "not_run_question_count": sum(row["status"] == "not_run" for row in type_rows),
                "partial_question_count": sum(row["status"] == "partial" for row in type_rows),
            }
            if type_scored:
                type_metrics.update({
                    key: statistics.fmean(row[key] for row in type_scored)
                    for key in ("recall@1", "recall@3", "recall@5", "reciprocal_rank")
                })
            else:
                type_metrics.update({
                    "recall@1": None, "recall@3": None, "recall@5": None,
                    "reciprocal_rank": None,
                })
            type_latency = [row["latency_ms"] for row in type_rows if row["latency_ms"] is not None]
            type_metrics["latency_ms"] = statistics.fmean(type_latency) if type_latency else None
            type_metrics["token_query_count"] = len(type_tokens)
            type_metrics["input_tokens_total"] = sum(row["input_tokens"] for row in type_tokens)
            type_metrics["output_tokens_total"] = sum(row["output_tokens"] for row in type_tokens)
            type_summary[question_type] = type_metrics

        error_texts = list(dict.fromkeys(
            row["error"] for row in method_rows if row.get("error")
        ))
        reason = status.get("reason", "")
        if not reason and error_texts:
            reason = error_texts[0]
        summary_methods.append({
            **status,
            "status": method_status,
            "reason": reason,
            "question_count": len(method_rows),
            "attempted_count": len(attempted_rows),
            "scored_count": len(scored_rows),
            "zero_ranking_count": len(zero_ranking_rows),
            "excluded_count": len(excluded_rows),
            "scored_question_count": len(scored_rows),
            "failed_question_count": len(failed_rows),
            "not_run_question_count": len(not_run_rows),
            "partial_question_count": len(partial_rows),
            "parse_failure_count": len(parse_failure_rows),
            "overall": overall,
            "token_usage": token_usage,
            "by_question_type": type_summary,
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
        "metrics": "page-level Recall@1/3/5 and MRR; per-query latency and token usage when exposed",
        "pageindex": pageindex_metadata,
        "methods": summary_methods,
    }


def write_summary(summary: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
