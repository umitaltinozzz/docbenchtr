from __future__ import annotations

import json

from .config import ROOT, RESULTS_DIR, SOURCES_PATH

START = "<!-- GENERATED:START -->"
END = "<!-- GENERATED:END -->"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def update_readme() -> None:
    readme_path = ROOT / "README.md"
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    doc_titles = ", ".join(item["title"] for item in sources)
    method_rows = []
    type_rows = []
    for method in summary["methods"]:
        overall = method["overall"]
        if overall is not None:
            method_rows.append(
                f"| {method['method']} | {method['attempted_count']} | {method['scored_count']} | "
                f"{method['zero_ranking_count']} | {method['excluded_count']} | "
                f"{_fmt(overall['recall@1'])} | {_fmt(overall['recall@3'])} | "
                f"{_fmt(overall['recall@5'])} | {_fmt(overall['reciprocal_rank'])} | "
                f"{_fmt(overall['latency_ms'])} ms |"
            )
            for question_type, values in method["by_question_type"].items():
                type_rows.append(
                    f"| {method['method']} | {question_type.replace('_', ' ')} | "
                    f"{values['attempted_count']} | {values['scored_count']} | "
                    f"{values['zero_ranking_count']} | {values['excluded_count']} | "
                    f"{_fmt(values['recall@1'])} | {_fmt(values['recall@3'])} | "
                    f"{_fmt(values['recall@5'])} | {_fmt(values['reciprocal_rank'])} | "
                    f"{_fmt(values['latency_ms'])} ms |"
                )
        else:
            label = "not run" if method["status"] == "not_run" else method["status"]
            method_rows.append(
                f"| {method['method']} | {method['attempted_count']} | {method['scored_count']} | "
                f"{method['zero_ranking_count']} | {method['excluded_count']} | {label} | "
                f"{label} | {label} | {label} |"
            )

    pageindex = summary.get("pageindex") or {}
    pageindex_doc_lines = []
    for document in pageindex.get("documents", []):
        if document.get("build_time_ms") is not None:
            duration = f"{document['build_time_ms'] / 1000:.1f} s"
        elif document.get("status") == "reused":
            duration = "cached; prior build time unavailable"
        else:
            duration = "not recorded"
        toc_source = document.get("toc_source") or "unavailable"
        line = (
            f"- `{document['document_id']}`: {document['status']}; "
            f"build {duration}; `toc_source={toc_source}`"
        )
        if document.get("reuse_validation_time_ms") is not None:
            line += f"; cache validation {document['reuse_validation_time_ms'] / 1000:.1f} s"
        if document.get("toc_source_error"):
            line += f"; toc_source recovery error: {document['toc_source_error']}"
        if document.get("error"):
            line += f"; error: {document['error']}"
        pageindex_doc_lines.append(line)

    pageindex_methods = [
        item for item in summary["methods"]
        if item["method"].startswith("PageIndex agentic (")
    ]
    if pageindex_methods:
        pageindex_query_line = "\n".join(
            f"- {item['method']} queries: {item['attempted_count']} attempted, "
            f"{item['scored_count']} scored, {item['zero_ranking_count']} zero-ranking, "
            f"{item['excluded_count']} excluded."
            for item in pageindex_methods
        )
        pageindex_token_line = "\n".join(
            f"- {item['method']} token counts: available for {item.get('token_usage', {}).get('query_count', 0)} queries; "
            f"{item.get('token_usage', {}).get('input_tokens_total', 0)} input and "
            f"{item.get('token_usage', {}).get('output_tokens_total', 0)} output tokens total."
            for item in pageindex_methods
        )
    else:
        pageindex_query_line = "- PageIndex agentic query status: unavailable."
        pageindex_token_line = "- PageIndex agentic token counts: unavailable."
    tree_search_method = next(
        (item for item in summary["methods"] if item["method"] == "PageIndex LLM Tree Search"),
        None,
    )
    if tree_search_method:
        tree_search_query_line = (
            f"- PageIndex LLM Tree Search queries: {tree_search_method['attempted_count']} attempted, "
            f"{tree_search_method['scored_count']} scored, {tree_search_method['zero_ranking_count']} zero-ranking, "
            f"{tree_search_method['excluded_count']} excluded; {tree_search_method['parse_failure_count']} parse failures."
        )
    else:
        tree_search_query_line = "- PageIndex LLM Tree Search query status: unavailable."
    pageindex_retry_line = (
        f"- PageIndex LLM HTTP safety net: retry a request after "
        f"{pageindex.get('llm_http_idle_timeout_seconds', 1800) // 60} minutes without response progress, "
        f"up to {pageindex.get('llm_http_idle_max_retries', 3)} retries; "
        f"{len(pageindex.get('llm_idle_retries', []))} retries recorded; "
        "there is no total index or query deadline."
    ) if pageindex else "- PageIndex LLM HTTP retry policy: unavailable."

    generated = f"""{START}
## Pilot results

This generated section reads its metrics from `results/summary.json` (per-question records: `results/results.csv` and `results/per_query.json`).

| Method | Attempted | Scored | Zero ranking | Excluded | Recall@1 | Recall@3 | Recall@5 | MRR | Mean query latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(method_rows)}

![Recall@5 by method and question type](results/recall_at_5_by_question_type.png)

### Metrics by question type

| Method | Question type | Attempted | Scored | Zero ranking | Excluded | Recall@1 | Recall@3 | Recall@5 | MRR | Mean query latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(type_rows)}

### Run metadata

- Documents: {summary['num_documents']} — {doc_titles}.
- Span-verified pilot questions: {summary['num_questions']}; excluded after verification: {summary['num_excluded_questions']}.
- Physical PDF pages extracted: {sum(summary['pages_per_document'].values())} across {', '.join(f"{key}: {value}" for key, value in summary['pages_per_document'].items())}.
- Embedding model: `{summary['embedding_model']}` revision `{summary['embedding_model_revision']}` on `{summary['embedding_device']}`.
- Question set label: **{summary['dataset_label']}**.
{pageindex_query_line}
{pageindex_token_line}
{tree_search_query_line}
{pageindex_retry_line}

### PageIndex index builds

{chr(10).join(pageindex_doc_lines) if pageindex_doc_lines else '- No PageIndex document index was built.'}

PageIndex agentic default-instruction and read-pages-instruction variants use `{pageindex.get('index_model', 'unavailable')}` through Ollama and share the cached trees. The read-pages instruction is an explicit modification to the SDK default. `tool_call_count` counts every agent tool invocation; page ranking uses only successful `get_page_content` results, in tool-call order and each result's `returned_pages` order, retaining each physical page's first occurrence. Outline-only calls count as tool calls but do not enter the ranking. PageIndex LLM Tree Search asks the same local model to select nodes from the cached tree using Ollama's JSON-schema output. It expands each selected node's inclusive one-based `start_index..end_index` range in returned `node_list` order and deduplicates pages by first occurrence. Attempted queries without a usable ranking, including parse failures, score zero; infrastructure exclusions remain outside the means. Both methods return no PageIndex page scores; per-query errors and traces are preserved in `results/per_query.json` and `results/results.csv`.
{END}"""
    if readme_path.exists():
        original = readme_path.read_text(encoding="utf-8")
        before = original.split(START)[0] if START in original else original
        after = original.split(END, 1)[1].rstrip("\r\n") if END in original else ""
        readme_path.write_text(before.rstrip() + "\n\n" + generated + after + "\n", encoding="utf-8")


def print_results_table() -> None:
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    print("method | attempted | scored | zero ranking | excluded | Recall@1 | Recall@3 | Recall@5 | MRR | mean latency ms")
    for method in summary["methods"]:
        if method["overall"] is not None:
            values = method["overall"]
            latency = f"{values['latency_ms']:.1f}" if values["latency_ms"] is not None else "n/a"
            print(
                f"{method['method']} | {method['attempted_count']} | {method['scored_count']} | "
                f"{method['zero_ranking_count']} | {method['excluded_count']} | "
                f"{_fmt(values['recall@1'])} | {_fmt(values['recall@3'])} | "
                f"{_fmt(values['recall@5'])} | {_fmt(values['reciprocal_rank'])} | "
                f"{latency}"
            )
        else:
            print(f"{method['method']} | {method['attempted_count']} | {method['scored_count']} | {method['zero_ranking_count']} | {method['excluded_count']} | {method['status']} | {method['reason']}")
