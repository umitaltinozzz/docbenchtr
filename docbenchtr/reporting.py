from __future__ import annotations

import json

from .config import ROOT, RESULTS_DIR, SOURCES_PATH

START = "<!-- GENERATED:START -->"
END = "<!-- GENERATED:END -->"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_delta(result: dict[str, object]) -> str:
    value = result["mean_difference"]
    interval = result["bootstrap_95_ci"]
    if value is None or interval is None:
        return "—"
    return f"{value:+.3f} [{interval[0]:+.3f}, {interval[1]:+.3f}]"


def _fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def update_readme() -> None:
    readme_path = ROOT / "README.md"
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    paired = json.loads((RESULTS_DIR / "paired_vs_hybrid.json").read_text(encoding="utf-8"))
    baselines = json.loads((RESULTS_DIR / "baselines.json").read_text(encoding="utf-8"))
    doc_titles = ", ".join(item["title"] for item in sources)
    method_rows = []
    type_rows = []
    for method in summary["methods"]:
        overall = method["overall"]
        if overall is not None:
            method_rows.append(
                f"| {method['method']} | {method['attempted_count']} | {method['scored_count']} | "
                f"{method.get('failed_as_zero_count', 0)} | {method['zero_ranking_count']} | {method['excluded_count']} | "
                f"{_fmt(overall['recall@1'])} | {_fmt(overall['recall@3'])} | "
                f"{_fmt(overall['recall@5'])} | {_fmt(overall['reciprocal_rank'])} | "
                f"{_fmt(overall['latency_ms'])} ms |"
            )
            for question_type, values in method["by_question_type"].items():
                type_rows.append(
                    f"| {method['method']} | {question_type.replace('_', ' ')} | "
                    f"{values['attempted_count']} | {values['scored_count']} | "
                    f"{values.get('failed_as_zero_count', 0)} | {values['zero_ranking_count']} | {values['excluded_count']} | "
                    f"{_fmt(values['recall@1'])} | {_fmt(values['recall@3'])} | "
                    f"{_fmt(values['recall@5'])} | {_fmt(values['reciprocal_rank'])} | "
                    f"{_fmt(values['latency_ms'])} ms |"
                )
        else:
            label = "not run" if method["status"] == "not_run" else method["status"]
            method_rows.append(
                f"| {method['method']} | {method['attempted_count']} | {method['scored_count']} | "
                f"{method.get('failed_as_zero_count', 0)} | {method['zero_ranking_count']} | {method['excluded_count']} | {label} | "
                f"{label} | {label} | {label} | {label} |"
            )

    paired_rows = []
    for run_id, comparisons in paired["runs"].items():
        for method, comparison in comparisons.items():
            original = comparison["all_questions"]["original_gold"]["metrics"]
            expanded = comparison["all_questions"]["expanded_gold"]["metrics"]
            r5 = original["recall@5"]
            mrr = original["reciprocal_rank"]
            paired_rows.append(
                f"| {run_id} | {method.removeprefix('PageIndex ')} | {_fmt_delta(r5)} | "
                f"{r5['wins']}/{r5['losses']}/{r5['ties']} | {_fmt_p(r5['exact_two_sided_sign_test_p'])} | "
                f"{_fmt_delta(mrr)} | {mrr['wins']}/{mrr['losses']}/{mrr['ties']} | "
                f"{_fmt_p(mrr['exact_two_sided_sign_test_p'])} | "
                f"{_fmt_delta(expanded['recall@5'])} |"
            )

    validator_sensitivity = paired["runs"]["colab_a100_models/qwen3.8-27b_ctx64k"][
        "PageIndex LLM Tree Search"
    ]["validator_pass_only"]
    sensitivity_rows = []
    for label, values in (
        ("Recall@5", validator_sensitivity["original_gold"]["metrics"]["recall@5"]),
        ("MRR", validator_sensitivity["original_gold"]["metrics"]["reciprocal_rank"]),
    ):
        sensitivity_rows.append(
            f"| {label} | {_fmt_delta(values)} | {values['wins']}/{values['losses']}/{values['ties']} | "
            f"{_fmt_p(values['exact_two_sided_sign_test_p'])} |"
        )
    expanded_validator_r5 = validator_sensitivity["expanded_gold"]["metrics"]["recall@5"]

    baseline_rows = []
    original_baselines = baselines["original_gold"]
    baseline_names = {
        "kvkk_6698": "KVKK",
        "tuketicinin_korunmasi_6502": "Consumer law",
        "yuksekogretim_2547": "Higher education law",
    }
    baseline_scopes = [("Overall", original_baselines["overall"])]
    baseline_scopes.extend(
        (baseline_names[doc_id], values)
        for doc_id, values in original_baselines["by_document"].items()
    )
    for scope, values in baseline_scopes:
        baseline_rows.append(
            f"| {scope} | {_fmt(values['first5_recall@5'])} | "
            f"{_fmt(values['random_order_expected_recall@5'])} |"
        )

    citation_rows = []
    expanded_gold_rows = []
    for summary_path in sorted(RESULTS_DIR.rglob("summary.json")):
        run_dir = summary_path.parent
        relative_run = run_dir.relative_to(RESULTS_DIR).as_posix()
        if "/repeat_" in relative_run and run_dir.parent.joinpath("aggregate.json").exists():
            continue
        run_id = "local" if relative_run == "." else relative_run
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for method in run_summary.get("methods", []):
            citation = method.get("cited_pages_secondary")
            if citation is not None:
                original = citation["original_gold"]
                expanded = citation["expanded_gold"]
                citation_rows.append(
                    f"| {run_id} | {method['method']} | {_fmt(original.get('cited_recall@5'))} | "
                    f"{_fmt(original.get('cited_reciprocal_rank'))} | "
                    f"{_fmt(expanded.get('cited_recall@5'))} |"
                )
            if (
                method["method"] in ("Hybrid (RRF)", "Hybrid + LLM rerank")
                or method["method"].startswith("PageIndex ")
            ):
                values = run_summary["gold_sensitivity"]["methods"][method["method"]]
                original = values["original_gold"]
                expanded = values["expanded_gold"]
                expanded_gold_rows.append(
                    f"| {run_id} | {method['method']} | "
                    f"{_fmt(original['recall@5'])} → {_fmt(expanded['expanded_recall@5'])} | "
                    f"{_fmt(original['reciprocal_rank'])} → {_fmt(expanded['expanded_reciprocal_rank'])} |"
                )

    for agg_path in sorted(RESULTS_DIR.rglob("aggregate.json")):
        agg_dir = agg_path.parent
        run_id = (
            "colab_a100_run3/qwen3.8-27b_27b-trees_ctx64k (3 repeats)"
            if "colab_a100_run3" in agg_dir.as_posix()
            else agg_dir.relative_to(RESULTS_DIR).as_posix()
        )
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        rep_pqs = sorted(agg_dir.glob("repeat_*/per_query.json"))
        if rep_pqs:
            from .reanalyze import QUESTIONS_PATH, _gold_groups, _metric_mean, _metrics_for_row, METRIC_KEYS
            questions_by_id = {
                q["question_id"]: q
                for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
            }
            all_rep_rows = []
            for rep_pq in rep_pqs:
                for raw_row in json.loads(rep_pq.read_text(encoding="utf-8")):
                    if not raw_row["method"].startswith("PageIndex agentic ("):
                        continue
                    q = questions_by_id[raw_row["question_id"]]
                    cited = dict(raw_row)
                    cited["ranked_pages"] = raw_row.get("cited_pages") or []
                    expanded_groups = _gold_groups(q)
                    expanded_gold = sorted({p for g in expanded_groups for p in g})
                    cited.update({
                        f"cited_{k}": v
                        for k, v in _metrics_for_row(cited, q["source_pages"]).items()
                    })
                    cited.update({
                        f"expanded_cited_{k}": v
                        for k, v in _metrics_for_row(
                            cited, expanded_gold, field_prefix="", relevant_groups=expanded_groups
                        ).items()
                    })
                    all_rep_rows.append(cited)
            for m_name in sorted({r["method"] for r in all_rep_rows}):
                m_rows = [r for r in all_rep_rows if r["method"] == m_name and r.get("status") != "not_run"]
                original = _metric_mean(m_rows, tuple(f"cited_{k}" for k in METRIC_KEYS), include_failed=True)
                expanded = _metric_mean(m_rows, tuple(f"expanded_cited_{k}" for k in METRIC_KEYS), include_failed=True)
                citation_rows.append(
                    f"| {run_id} | {m_name} | {_fmt(original.get('cited_recall@5'))} | "
                    f"{_fmt(original.get('cited_reciprocal_rank'))} | "
                    f"{_fmt(expanded.get('expanded_cited_recall@5'))} |"
                )
        for m_name, m_data in agg.get("methods", {}).items():
            if m_name in ("Hybrid (RRF)", "Hybrid + LLM rerank") or m_name.startswith("PageIndex "):
                mt = m_data["metrics"]
                expanded_gold_rows.append(
                    f"| {run_id} | {m_name} | "
                    f"{_fmt(mt['recall@5']['mean'])} → {_fmt(mt['expanded_recall@5']['mean'])} | "
                    f"{_fmt(mt['reciprocal_rank']['mean'])} → {_fmt(mt['expanded_reciprocal_rank']['mean'])} |"
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
            f"{item['scored_count']} scored, {item.get('failed_as_zero_count', 0)} failed-as-zero, "
            f"{item['zero_ranking_count']} zero-ranking, "
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
            f"{tree_search_method['scored_count']} scored, {tree_search_method.get('failed_as_zero_count', 0)} failed-as-zero, "
            f"{tree_search_method['zero_ranking_count']} zero-ranking, "
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

Reported means count every failed query as zero. Infrastructure rows marked not-run are excluded. Scored-only values are kept under `methods[].scored_only` in `summary.json`.

| Method | Attempted | Scored | Failed as zero | Zero ranking | Excluded | Recall@1 | Recall@3 | Recall@5 | MRR | Mean query latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(method_rows)}

![Recall@5 by method and question type](results/recall_at_5_by_question_type.png)

### Metrics by question type

| Method | Question type | Attempted | Scored | Failed as zero | Zero ranking | Excluded | Recall@1 | Recall@3 | Recall@5 | MRR | Mean query latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(type_rows)}

### PageIndex and LLM rerank vs Hybrid: paired question-level comparison

Each row uses the same questions for the evaluated method and Hybrid. Intervals are percentile 95% paired-bootstrap CIs (10,000 resamples; fixed seed); p values are exact two-sided sign tests. Win/loss/tie counts are shown for each metric. Original `source_pages` are the primary labels; expanded-gold results treat documented equivalent pages as alternatives within the same provision, so the number of targets per question stays fixed. Full results and validator-pass subsets are in [`results/paired_vs_hybrid.json`](results/paired_vs_hybrid.json).

| Run | Method / PageIndex variant | Recall@5 Δ [95% CI] | W/L/T | p | MRR Δ [95% CI] | W/L/T | p | Expanded-gold Recall@5 Δ [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(paired_rows)}

### Original and expanded gold: absolute scores by run

Each pair shows the primary pre-registered score followed by the expanded-gold sensitivity score. Equivalent pages are alternatives within the same provision, so the target count per question is unchanged.

| Run | Method | Recall@5 primary → expanded | MRR primary → expanded |
|---|---|---:|---:|
{chr(10).join(expanded_gold_rows)}

Validator-pass sensitivity for qwen3.8 27B at 64k (32 of 36 questions; Tree Search vs Hybrid):

| Metric | Δ [95% CI] | W/L/T | Exact p |
|---|---:|---:|---:|
{chr(10).join(sensitivity_rows)}

Expanded-gold Recall@5 for that validator-pass subset: {_fmt_delta(expanded_validator_r5)}.

### Simple baselines (original gold)

The first-five baseline always returns physical pages 1–5. Random-order values are the analytic expected Recall@5 for a uniformly random page ranking. The 6502 equivalent-page additions are outside page 5, so the first-five baseline is unchanged under expanded gold.

| Scope | First 5 pages Recall@5 | Random order expected Recall@5 |
|---|---:|---:|
{chr(10).join(baseline_rows)}

### Agentic citation pages (secondary score)

This row scores `cited_pages` rather than reconstructed page-read order. Failed queries count as zero under the same rule.

| Run | Agentic variant | Citation Recall@5, original | Citation MRR, original | Citation Recall@5, expanded |
|---|---|---:|---:|---:|
{chr(10).join(citation_rows)}

### Run metadata

- Documents: {summary['num_documents']} — {doc_titles}.
- Retained pilot questions: {summary['num_questions']}; failed planned-term validator checks retained in results: {summary.get('num_validator_failed_questions', 0)}; dropped: {summary['num_excluded_questions']}.
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
    print("method | attempted | scored | failed as zero | zero ranking | excluded | Recall@1 | Recall@3 | Recall@5 | MRR | mean latency ms")
    for method in summary["methods"]:
        if method["overall"] is not None:
            values = method["overall"]
            latency = f"{values['latency_ms']:.1f}" if values["latency_ms"] is not None else "n/a"
            print(
                f"{method['method']} | {method['attempted_count']} | {method['scored_count']} | "
                f"{method.get('failed_as_zero_count', 0)} | {method['zero_ranking_count']} | {method['excluded_count']} | "
                f"{_fmt(values['recall@1'])} | {_fmt(values['recall@3'])} | "
                f"{_fmt(values['recall@5'])} | {_fmt(values['reciprocal_rank'])} | "
                f"{latency}"
            )
        else:
            print(f"{method['method']} | {method['attempted_count']} | {method['scored_count']} | {method.get('failed_as_zero_count', 0)} | {method['zero_ranking_count']} | {method['excluded_count']} | {method['status']} | {method['reason']}")
