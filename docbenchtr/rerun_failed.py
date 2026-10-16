"""Re-run failed rows of one method and rebuild the published results.

Usage: python -m docbenchtr.rerun_failed "PageIndex LLM Tree Search"

Rows with status "failed" for the named method are evaluated again with the
current code; every other row in results/per_query.json is kept unchanged.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from .config import DOCUMENTS, RAW_DIR, RESULTS_DIR
from .documents import download_documents, extract_pdf_pages
from .evaluate import evaluate_question, summarize, write_rows, write_summary
from .plotting import plot_recall_by_type
from .questions import draft_questions
from .reporting import print_results_table, update_readme
from .retrievers import PageIndexLLMTreeSearchRetriever, PageIndexRetriever, ollama_status
from .run_pilot import ROOT_PAGEINDEX_DIR, _pipeline_run_lock

SUPPORTED_METHOD = "PageIndex LLM Tree Search"


def main(method: str) -> int:
    if method != SUPPORTED_METHOD:
        raise SystemExit(f"Only {SUPPORTED_METHOD!r} reruns are supported.")
    rows = json.loads((RESULTS_DIR / "per_query.json").read_text(encoding="utf-8"))
    previous = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    failed = [i for i, row in enumerate(rows) if row["method"] == method and row["status"] == "failed"]
    print(f"{len(failed)} failed {method} rows to re-run.")
    if not failed:
        return 0

    pages = extract_pdf_pages(download_documents())
    questions, dropped = draft_questions(pages)
    by_id = {question["question_id"]: question for question in questions}
    pages_per_doc: dict[str, int] = defaultdict(int)
    for page in pages:
        pages_per_doc[page["document_id"]] += 1

    reachable, ollama_reason = ollama_status()
    pdf_paths = {document["document_id"]: str(RAW_DIR / document["filename"]) for document in DOCUMENTS}
    pageindex = PageIndexRetriever(
        pdf_paths,
        str(ROOT_PAGEINDEX_DIR()),
        unavailable_reason=None if reachable else ollama_reason,
        instruction_variant="default",
    )
    tree_search = PageIndexLLMTreeSearchRetriever(pageindex, str(ROOT_PAGEINDEX_DIR()))
    reruns = []
    try:
        for index in failed:
            old = rows[index]
            question = by_id[old["question_id"]]
            new = evaluate_question(tree_search, question, pages_per_doc[question["document_id"]])
            print(f"  {old['question_id']}: {old['status']} -> {new['status']} "
                  f"(R@5={new['recall@5']}, {new['latency_ms'] / 1000:.0f}s)", flush=True)
            reruns.append({
                "question_id": old["question_id"],
                "previous_status": old["status"],
                "previous_error": old["error"],
                "new_status": new["status"],
            })
            rows[index] = new
    finally:
        pageindex.close()

    write_rows(rows)
    rebuild_summary(rows, previous, questions, pages, len(dropped), tree_search.results_metadata(), reruns)
    return 0


def rebuild_summary(rows, previous, questions, pages, dropped_count, tree_search_metadata, reruns) -> None:
    pageindex_metadata = dict(previous["pageindex"])
    pageindex_metadata["llm_tree_search"] = {
        **tree_search_metadata,
        "failed_row_reruns": {
            "rerun_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": "First pass used a 180 s HTTP timeout on /api/chat; reruns use 1800 s.",
            "rows": reruns,
        },
    }
    statuses = [{"method": entry["method"], "reason": entry.get("reason", "")} for entry in previous["methods"]]
    summary = summarize(
        rows, questions, statuses, pages, previous["embedding_device"], dropped_count, pageindex_metadata,
    )
    write_summary(summary)
    plot_recall_by_type(summary)
    update_readme()
    print_results_table()


if __name__ == "__main__":
    with _pipeline_run_lock():
        sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else SUPPORTED_METHOD))
