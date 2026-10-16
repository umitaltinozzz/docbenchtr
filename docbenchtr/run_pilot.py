from __future__ import annotations

import json
import sys

import torch

from .config import DOCUMENTS, RAW_DIR, RESULTS_DIR
from .documents import download_documents, extract_pdf_pages
from .evaluate import run_evaluation, summarize, write_summary
from .plotting import plot_recall_by_type
from .questions import draft_questions
from .reporting import print_results_table, update_readme
from .retrievers import PageIndexRetriever, ollama_status, pageindex_local_preflight


def main() -> int:
    print("Downloading and hashing public source PDFs...")
    sources = download_documents()
    pages = extract_pdf_pages(sources)
    print(f"Extracted {len(pages)} physical PDF pages across {len(sources)} documents.")

    questions, dropped = draft_questions(pages)
    if not questions:
        raise RuntimeError("No source-span-verified questions survived; evaluation stopped.")
    print(f"Kept {len(questions)} span-verified questions; dropped {len(dropped)} failed drafts.")

    pageindex_status = {"method": "PageIndex (local Ollama)", "status": "not_run", "reason": ""}
    extra_retrievers = []
    reachable, ollama_reason = ollama_status()
    if reachable:
        try:
            pdf_paths = {document["document_id"]: str(RAW_DIR / document["filename"]) for document in DOCUMENTS}
            pageindex = PageIndexRetriever(pdf_paths, str(ROOT_PAGEINDEX_DIR()))
            extra_retrievers.append(pageindex)
            pageindex_status = {"method": pageindex.name, "status": "ran", "reason": ""}
        except Exception as error:
            pageindex_status["reason"] = f"PageIndex local SDK failed during setup/indexing: {type(error).__name__}: {error}"
    else:
        try:
            sdk_check = pageindex_local_preflight(str(ROOT_PAGEINDEX_DIR()))
            pageindex_status["reason"] = (
                f"{ollama_reason}. PageIndex local SDK preflight passed ({sdk_check}); "
                "document indexing and retrieval were not run."
            )
        except Exception as error:
            pageindex_status["reason"] = (
                f"{ollama_reason}. PageIndex local SDK preflight failed: "
                f"{type(error).__name__}: {error}"
            )

    rows, retrievers = run_evaluation(pages, questions, extra_retrievers)
    statuses = [{"method": retriever.name, "status": "ran", "reason": ""} for retriever in retrievers]
    if not extra_retrievers:
        statuses.append(pageindex_status)
    summary = summarize(rows, questions, statuses, pages, "cuda" if torch.cuda.is_available() else "cpu", len(dropped))
    write_summary(summary)
    plot_recall_by_type(summary)
    update_readme()
    print_results_table()
    print(f"Results written to {RESULTS_DIR}")
    return 0


def ROOT_PAGEINDEX_DIR():
    return RESULTS_DIR.parent / ".pageindex"


if __name__ == "__main__":
    sys.exit(main())
