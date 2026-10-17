from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import torch

from .config import DOCUMENTS, RAW_DIR, RESULTS_DIR
from .cache import prepare_cache
from .documents import download_documents, extract_pdf_pages
from .evaluate import run_evaluation, summarize, write_summary
from .questions import draft_questions
from .reporting import print_results_table, update_readme
from .retrievers import PageIndexLLMTreeSearchRetriever, PageIndexRetriever, ollama_status


def main() -> int:
    from .plotting import plot_recall_by_type
    print("Downloading and hashing public source PDFs...")
    sources = download_documents()
    pages = extract_pdf_pages(sources)
    print(f"Extracted {len(pages)} physical PDF pages across {len(sources)} documents.")

    questions, dropped = draft_questions(pages)
    if not questions:
        raise RuntimeError("No source-span-verified questions survived; evaluation stopped.")
    print(
        f"Retained {len(questions)} questions; {len(dropped)} fail the planned-term validator "
        "and remain in the scored set."
    )

    pdf_paths = {
        document["document_id"]: str(RAW_DIR / document["filename"])
        for document in DOCUMENTS
    }
    reachable, ollama_reason = ollama_status()
    pageindex_default = PageIndexRetriever(
        pdf_paths,
        str(ROOT_PAGEINDEX_DIR()),
        unavailable_reason=None if reachable else ollama_reason,
        instruction_variant="default",
    )
    pageindex_read_pages = PageIndexRetriever(
        pdf_paths,
        str(ROOT_PAGEINDEX_DIR()),
        unavailable_reason=None if reachable else ollama_reason,
        instruction_variant="read_pages",
    )
    pageindex_tree_search = PageIndexLLMTreeSearchRetriever(
        pageindex_default,
        str(ROOT_PAGEINDEX_DIR()),
    )

    try:
        rows, retrievers = run_evaluation(
            pages, questions, [pageindex_default, pageindex_read_pages, pageindex_tree_search]
        )
    finally:
        pageindex_default.close()
        pageindex_read_pages.close()
    statuses = [{"method": retriever.name, "reason": ""} for retriever in retrievers]
    summary = summarize(
        rows,
        questions,
        statuses,
        pages,
        "cuda" if torch.cuda.is_available() else "cpu",
        len(dropped),
        {
            **pageindex_default.results_metadata(),
            "agentic_default_instructions": pageindex_default.results_metadata(),
            "agentic_read_pages_instruction": pageindex_read_pages.results_metadata(),
            "llm_tree_search": pageindex_tree_search.results_metadata(),
        },
    )
    write_summary(summary)
    plot_recall_by_type(summary)
    update_readme()
    print_results_table()
    print(f"Results written to {RESULTS_DIR}")
    return 0


def ROOT_PAGEINDEX_DIR():
    return prepare_cache()


class PipelineAlreadyRunning(RuntimeError):
    pass


@contextmanager
def _pipeline_run_lock():
    """Prevent concurrent one-command runs from racing on cache/results."""
    lock_path = RESULTS_DIR.parent / ".pageindex" / ".docbenchtr-run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise PipelineAlreadyRunning(
            "Another DocBenchTR pipeline run is already active."
        ) from error

    try:
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


if __name__ == "__main__":
    try:
        with _pipeline_run_lock():
            sys.exit(main())
    except PipelineAlreadyRunning as error:
        print(error, file=sys.stderr)
        sys.exit(2)
