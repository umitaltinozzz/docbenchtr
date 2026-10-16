from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        if method["status"] == "ran":
            overall = method["overall"]
            method_rows.append(
                f"| {method['method']} | {_fmt(overall['recall@1'])} | {_fmt(overall['recall@3'])} | "
                f"{_fmt(overall['recall@5'])} | {_fmt(overall['reciprocal_rank'])} | {overall['latency_ms']:.1f} ms |"
            )
            for question_type, values in method["by_question_type"].items():
                type_rows.append(f"| {method['method']} | {question_type.replace('_', ' ')} | {_fmt(values['recall@5'])} | {values['question_count']} |")
        else:
            method_rows.append(f"| {method['method']} | not run | not run | not run | not run | not run |")
    reasons = [f"- **{item['method']}:** not run — {item['reason']}" for item in summary["methods"] if item["status"] != "ran"]
    generated = f"""{START}
## Pilot results

This generated section reads its metrics from `results/summary.json` (per-question records: `results/results.csv` and `results/per_query.json`).

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Mean query latency |
|---|---:|---:|---:|---:|---:|
{chr(10).join(method_rows)}

![Recall@5 by method and question type](results/recall_at_5_by_question_type.png)

### Recall@5 by question type

| Method | Question type | Recall@5 | Questions |
|---|---|---:|---:|
{chr(10).join(type_rows)}

### Run metadata

- Documents: {summary['num_documents']} — {doc_titles}.
- Span-verified pilot questions: {summary['num_questions']}; excluded after verification: {summary['num_excluded_questions']}.
- Physical PDF pages extracted: {sum(summary['pages_per_document'].values())} across {', '.join(f"{key}: {value}" for key, value in summary['pages_per_document'].items())}.
- Embedding model: `{summary['embedding_model']}` revision `{summary['embedding_model_revision']}` on `{summary['embedding_device']}`.
- Question set label: **{summary['dataset_label']}**.
{chr(10).join(reasons)}
{END}"""
    if readme_path.exists():
        original = readme_path.read_text(encoding="utf-8")
        before = original.split(START)[0] if START in original else original
        after = original.split(END, 1)[1].rstrip("\r\n") if END in original else ""
        readme_path.write_text(before.rstrip() + "\n\n" + generated + after + "\n", encoding="utf-8")


def print_results_table() -> None:
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    print("method | Recall@1 | Recall@3 | Recall@5 | MRR | mean latency ms")
    for method in summary["methods"]:
        if method["status"] == "ran":
            values = method["overall"]
            print(
                f"{method['method']} | {values['recall@1']:.3f} | {values['recall@3']:.3f} | "
                f"{values['recall@5']:.3f} | {values['reciprocal_rank']:.3f} | {values['latency_ms']:.1f}"
            )
        else:
            print(f"{method['method']} | not run | {method['reason']}")
