"""Re-score saved retrieval rankings without running any models."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
QUESTIONS_PATH = ROOT / "data" / "pilot_questions.json"
VERIFICATION_PATH = ROOT / "data" / "question_verification.json"
METRIC_KEYS = ("recall@1", "recall@3", "recall@5", "reciprocal_rank")
PAGEINDEX_PREFIX = "PageIndex "
HYBRID_METHOD = "Hybrid (RRF)"
RERANK_METHOD = "Hybrid + LLM rerank"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261017


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_ranking(ranked_pages: list[int], relevant_pages: list[int]) -> dict[str, float]:
    relevant = set(relevant_pages)
    first_rank = next(
        (rank for rank, page in enumerate(ranked_pages, start=1) if page in relevant),
        None,
    )
    return {
        f"recall@{k}": len(set(ranked_pages[:k]) & relevant) / len(relevant)
        for k in (1, 3, 5)
    } | {"reciprocal_rank": 1 / first_rank if first_rank else 0.0}


def score_ranking_groups(
    ranked_pages: list[int], relevant_page_groups: list[list[int]]
) -> dict[str, float]:
    """Count each provision once, even when several pages are acceptable alternatives."""
    if not relevant_page_groups or any(not group for group in relevant_page_groups):
        raise ValueError("Each question needs at least one non-empty relevant page group")
    first_rank = next(
        (rank for rank, page in enumerate(ranked_pages, start=1)
         if any(page in group for group in relevant_page_groups)),
        None,
    )
    return {
        f"recall@{k}": sum(
            bool(set(ranked_pages[:k]).intersection(group))
            for group in relevant_page_groups
        ) / len(relevant_page_groups)
        for k in (1, 3, 5)
    } | {"reciprocal_rank": 1 / first_rank if first_rank else 0.0}


def _metrics_for_row(
    row: dict[str, Any], relevant: list[int], *, field_prefix: str = "",
    relevant_groups: list[list[int]] | None = None,
) -> dict[str, float | None]:
    status = row.get("status")
    if status == "not_run":
        return {f"{field_prefix}{key}": None for key in METRIC_KEYS}
    # Failed attempts are explicitly scored as zero even if a partial ranking was saved.
    if status == "failed":
        return {f"{field_prefix}{key}": 0.0 for key in METRIC_KEYS}
    values = (
        score_ranking_groups(row.get("ranked_pages") or [], relevant_groups)
        if relevant_groups is not None
        else score_ranking(row.get("ranked_pages") or [], relevant)
    )
    return {f"{field_prefix}{key}": value for key, value in values.items()}


def _metric_mean(rows: list[dict[str, Any]], keys: tuple[str, ...], *, include_failed: bool) -> dict[str, float | None]:
    included = [
        row for row in rows
        if row.get("status") != "not_run"
        and (include_failed or row.get("status") != "failed")
    ]
    return {
        key: fmean(float(row[key]) for row in included if row.get(key) is not None)
        if any(row.get(key) is not None for row in included) else None
        for key in keys
    }


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256((str(BOOTSTRAP_SEED) + "|" + "|".join(parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(tail + 1)) / (2**n))


def _paired_metric(
    pageindex: dict[str, dict[str, Any]],
    hybrid: dict[str, dict[str, Any]],
    question_ids: list[str],
    metric_key: str,
    seed_parts: tuple[str, ...],
) -> dict[str, Any]:
    differences = np.asarray([
        float(pageindex[qid][metric_key]) - float(hybrid[qid][metric_key])
        for qid in question_ids
    ], dtype=float)
    wins = int(np.count_nonzero(differences > 0))
    losses = int(np.count_nonzero(differences < 0))
    ties = int(np.count_nonzero(differences == 0))
    if len(differences):
        rng = np.random.default_rng(_stable_seed(*seed_parts, metric_key))
        indices = rng.integers(0, len(differences), size=(BOOTSTRAP_RESAMPLES, len(differences)))
        bootstrap_means = differences[indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        mean_difference = float(differences.mean())
    else:
        lower = upper = mean_difference = None
    return {
        "mean_difference": mean_difference,
        "bootstrap_95_ci": [float(lower), float(upper)] if lower is not None else None,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "exact_two_sided_sign_test_p": _sign_test_p(wins, losses),
        "n_questions": len(question_ids),
    }


def _paired_comparison(
    rows: list[dict[str, Any]],
    pageindex_method: str,
    gold_key: str,
    question_ids: list[str],
    run_id: str,
    subset: str,
) -> dict[str, Any]:
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["question_id"]] = row
    pi_rows = by_method[pageindex_method]
    hybrid_rows = by_method[HYBRID_METHOD]
    paired_ids = [
        qid for qid in question_ids
        if qid in pi_rows and qid in hybrid_rows
        and pi_rows[qid][f"{gold_key}recall@5"] is not None
        and hybrid_rows[qid][f"{gold_key}recall@5"] is not None
    ]
    # The per-query fields use no prefix for the pre-registered original gold.
    prefix = "" if gold_key == "" else gold_key
    stats = {
        metric: _paired_metric(
            pi_rows, hybrid_rows, paired_ids, f"{prefix}{metric}",
            (run_id, pageindex_method, gold_key or "original", subset),
        )
        for metric in ("recall@5", "reciprocal_rank")
    }
    return {"n_questions": len(paired_ids), "metrics": stats}


def _baseline_summary(
    questions: list[dict[str, Any]], pages_per_document: dict[str, int], gold_key: str
) -> dict[str, Any]:
    by_document: dict[str, list[dict[str, float]]] = {}
    all_values: list[dict[str, float]] = []
    for question in questions:
        doc_id = question["document_id"]
        groups = _gold_groups(question) if gold_key == "expanded" else [[p] for p in question["source_pages"]]
        page_count = pages_per_document[doc_id]
        first_five = score_ranking_groups(list(range(1, min(5, page_count) + 1)), groups)
        random_expected = fmean(
            1 - math.comb(page_count - len(group), min(5, page_count)) / math.comb(page_count, min(5, page_count))
            for group in groups
        )
        value = {
            "first5_recall@5": first_five["recall@5"],
            "random_order_expected_recall@5": random_expected,
        }
        by_document.setdefault(doc_id, []).append(value)
        all_values.append(value)

    def average(values: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: fmean(row[key] for row in values)
            for key in ("first5_recall@5", "random_order_expected_recall@5")
        }

    return {
        "overall": average(all_values),
        "by_document": {doc: average(values) for doc, values in by_document.items()},
    }


def _gold_groups(question: dict[str, Any]) -> list[list[int]]:
    equivalent = question.get("equivalent_pages", {})
    return [
        [page, *equivalent.get(str(page), [])]
        for page in question["source_pages"]
    ]


def _summary_for_run(
    previous: dict[str, Any],
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    pages_per_document: dict[str, int],
    failed_validator_ids: set[str],
) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    by_type: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
        by_type.setdefault(row["method"], {}).setdefault(row["question_type"], []).append(row)

    old_methods = {item["method"]: item for item in previous.get("methods", [])}
    methods = []
    for method, method_rows in by_method.items():
        successful = [r for r in method_rows if r["status"] in ("ran", "partial", "zero_ranking")]
        failed = [r for r in method_rows if r["status"] == "failed"]
        attempted = [*successful, *failed]
        excluded = [r for r in method_rows if r["status"] == "not_run"]
        method_entry = dict(old_methods.get(method, {}))
        latency = [r["latency_ms"] for r in method_rows if r.get("latency_ms") is not None]
        reported = _metric_mean(method_rows, METRIC_KEYS, include_failed=True)
        scored_only = _metric_mean(method_rows, METRIC_KEYS, include_failed=False)
        reported["latency_ms"] = fmean(latency) if latency else None
        method_entry.update({
            "method": method,
            "status": (
                "not_run" if not attempted and excluded else
                "failed" if failed and not successful else
                "partial" if failed or excluded or any(r["status"] == "partial" for r in successful) else
                "ran"
            ),
            "question_count": len(method_rows),
            "attempted_count": len(attempted),
            "scored_count": len(successful),
            "scored_question_count": len(successful),
            "failed_as_zero_count": len(failed),
            "failed_question_count": len(failed),
            "zero_ranking_count": sum(not r.get("ranked_pages") for r in successful),
            "excluded_count": len(excluded),
            "not_run_question_count": len(excluded),
            "partial_question_count": sum(r["status"] == "partial" for r in successful),
            "parse_failure_count": sum(bool(r.get("parse_failure")) for r in method_rows),
            "overall": reported if attempted or latency else None,
            "scored_only": scored_only,
            "by_question_type": {},
        })
        for question_type, type_rows in by_type.get(method, {}).items():
            type_successful = [r for r in type_rows if r["status"] in ("ran", "partial", "zero_ranking")]
            type_failed = [r for r in type_rows if r["status"] == "failed"]
            type_attempted = [*type_successful, *type_failed]
            type_metrics = _metric_mean(type_rows, METRIC_KEYS, include_failed=True)
            type_metrics["scored_only"] = _metric_mean(type_rows, METRIC_KEYS, include_failed=False)
            type_latencies = [r["latency_ms"] for r in type_rows if r.get("latency_ms") is not None]
            type_metrics.update({
                "attempted_question_count": len(type_attempted),
                "question_count": len(type_successful),
                "attempted_count": len(type_attempted),
                "scored_count": len(type_successful),
                "zero_ranking_count": sum(not r.get("ranked_pages") for r in type_successful),
                "failed_as_zero_count": len(type_failed),
                "excluded_count": sum(r["status"] == "not_run" for r in type_rows),
                "failed_question_count": len(type_failed),
                "not_run_question_count": sum(r["status"] == "not_run" for r in type_rows),
                "partial_question_count": sum(r["status"] == "partial" for r in type_successful),
                "latency_ms": fmean(type_latencies) if type_latencies else None,
                "token_query_count": sum(isinstance(r.get("input_tokens"), int) for r in type_rows),
                "input_tokens_total": sum(r.get("input_tokens") or 0 for r in type_rows),
                "output_tokens_total": sum(r.get("output_tokens") or 0 for r in type_rows),
            })
            method_entry["by_question_type"][question_type] = type_metrics

        if method.startswith("PageIndex agentic ("):
            citations = [r for r in method_rows if r["status"] != "not_run"]
            expanded_citations = _metric_mean(
                citations,
                tuple(f"expanded_cited_{key}" for key in METRIC_KEYS),
                include_failed=True,
            )
            method_entry["cited_pages_secondary"] = {
                "original_gold": _metric_mean(citations, tuple(f"cited_{k}" for k in METRIC_KEYS), include_failed=True),
                "expanded_gold": {
                    key.removeprefix("expanded_"): value
                    for key, value in expanded_citations.items()
                },
            }
        methods.append(method_entry)

    validator_failed_questions = [q for q in questions if q["question_id"] in failed_validator_ids]
    result = dict(previous)
    result.update({
        "dataset_label": (
            "pilot set; hand-written question stems and source spans checked; "
            f"{len(validator_failed_questions)} planned-term validator failures retained; not yet human-reviewed"
        ),
        "num_questions": len(questions),
        "num_excluded_questions": 0,
        "num_validator_failed_questions": len(validator_failed_questions),
        "metrics": "page-level Recall@1/3/5 and MRR; failed query = 0; not-run infrastructure exclusions omitted",
        "scoring_rule": {
            "failed_query": "zero for Recall@1/3/5 and MRR",
            "not_run": "excluded when retrieval did not start because the document was unavailable",
            "scored_only": "reported separately under methods[].scored_only",
        },
        "methods": methods,
        "baselines": {
            "original_gold": _baseline_summary(questions, pages_per_document, "original"),
            "expanded_gold": _baseline_summary(questions, pages_per_document, "expanded"),
        },
        "gold_sensitivity": {
            "primary_gold": "original source_pages (pre-registered labels)",
            "expanded_gold": "source_pages plus documented equivalent_pages; sensitivity analysis only",
            "validator_pass_question_count": len(questions) - len(validator_failed_questions),
        },
    })
    return result


def reanalyze_saved_results() -> dict[str, Any]:
    questions = _read_json(QUESTIONS_PATH)
    question_by_id = {q["question_id"]: q for q in questions}
    verification = _read_json(VERIFICATION_PATH)
    failed_validator_ids = {
        item["question_id"] for item in verification.get("checks", []) if not item.get("verified")
    }
    pages_per_document = {k: int(v) for k, v in _read_json(ROOT / "results" / "summary.json")["pages_per_document"].items()}

    run_files = sorted(RESULTS.rglob("per_query.json"))
    all_runs: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    run_paths: dict[str, Path] = {}
    for per_query_path in run_files:
        run_dir = per_query_path.parent
        relative_run = run_dir.relative_to(RESULTS).as_posix()
        if "/repeat_" in relative_run and run_dir.parent.joinpath("aggregate.json").exists():
            continue
        rows = _read_json(per_query_path)
        if not rows:
            continue
        run_id = "local" if relative_run == "." else relative_run
        summary_path = run_dir / "summary.json"
        previous = _read_json(summary_path)
        for row in rows:
            question = question_by_id[row["question_id"]]
            original_gold = question["source_pages"]
            expanded_groups = _gold_groups(question)
            expanded_gold = sorted({page for group in expanded_groups for page in group})
            original = _metrics_for_row(row, original_gold)
            expanded = _metrics_for_row(
                row, expanded_gold, field_prefix="expanded_", relevant_groups=expanded_groups
            )
            row.update(original)
            row.update(expanded)
            cited = dict(row)
            cited["ranked_pages"] = row.get("cited_pages") or []
            row.update({
                f"cited_{key}": value
                for key, value in _metrics_for_row(cited, original_gold).items()
            })
            row.update({
                f"expanded_cited_{key}": value
                for key, value in _metrics_for_row(
                    cited, expanded_gold, field_prefix="", relevant_groups=expanded_groups
                ).items()
            })
            row["expanded_gold_pages"] = expanded_gold
            row["expanded_gold_groups"] = expanded_groups
        _write_json(per_query_path, rows)
        summary = _summary_for_run(
            previous, rows, questions, pages_per_document, failed_validator_ids
        )
        # Aggregate original and expanded-gold means from the saved full rankings.
        summary["gold_sensitivity"]["methods"] = {
            method: {
                "original_gold": _metric_mean(
                    method_rows, METRIC_KEYS, include_failed=True
                ),
                "original_gold_scored_only": _metric_mean(
                    method_rows, METRIC_KEYS, include_failed=False
                ),
                "expanded_gold": _metric_mean(
                    method_rows, tuple(f"expanded_{key}" for key in METRIC_KEYS), include_failed=True
                ),
                "expanded_gold_scored_only": _metric_mean(
                    method_rows, tuple(f"expanded_{key}" for key in METRIC_KEYS), include_failed=False
                ),
            }
            for method, method_rows in _group_by_method(rows).items()
        }
        document_sensitivity: dict[str, dict[str, Any]] = {}
        for method, method_rows in _group_by_method(rows).items():
            by_document: dict[str, list[dict[str, Any]]] = {}
            for row in method_rows:
                by_document.setdefault(row["document_id"], []).append(row)
            for document_id, document_rows in by_document.items():
                document_sensitivity.setdefault(document_id, {})[method] = {
                    "original_gold": _metric_mean(
                        document_rows, METRIC_KEYS, include_failed=True
                    ),
                    "original_gold_scored_only": _metric_mean(
                        document_rows, METRIC_KEYS, include_failed=False
                    ),
                    "expanded_gold": _metric_mean(
                        document_rows,
                        tuple(f"expanded_{key}" for key in METRIC_KEYS),
                        include_failed=True,
                    ),
                    "expanded_gold_scored_only": _metric_mean(
                        document_rows,
                        tuple(f"expanded_{key}" for key in METRIC_KEYS),
                        include_failed=False,
                    ),
                }
        summary["gold_sensitivity"]["by_document"] = document_sensitivity
        _write_json(summary_path, summary)
        all_runs[run_id] = rows
        summaries[run_id] = summary
        run_paths[run_id] = run_dir

    paired = {
        "analysis": "Paired PageIndex and LLM-rerank vs Hybrid comparison from saved per-query rankings",
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "interval": "percentile 95% CI over paired question resamples",
            "seed": BOOTSTRAP_SEED,
            "unit": "question",
        },
        "sign_test": "exact two-sided binomial sign test; ties excluded",
        "runs": {},
    }
    for run_id, rows in all_runs.items():
        method_names = list(dict.fromkeys(row["method"] for row in rows))
        compared_methods = [
            name for name in method_names
            if name.startswith(PAGEINDEX_PREFIX) or name == RERANK_METHOD
        ]
        ids = sorted({row["question_id"] for row in rows})
        validator_pass = [qid for qid in ids if qid not in failed_validator_ids]
        comparisons = {}
        for method in compared_methods:
            comparisons[method] = {
                "vs_method": HYBRID_METHOD,
                "all_questions": {
                    "original_gold": _paired_comparison(rows, method, "", ids, run_id, "all"),
                    "expanded_gold": _paired_comparison(rows, method, "expanded_", ids, run_id, "all"),
                },
                "validator_pass_only": {
                    "original_gold": _paired_comparison(rows, method, "", validator_pass, run_id, "validator_pass"),
                    "expanded_gold": _paired_comparison(rows, method, "expanded_", validator_pass, run_id, "validator_pass"),
                },
            }
        paired["runs"][run_id] = comparisons

    for agg_path in sorted(RESULTS.rglob("aggregate.json")):
        agg_dir = agg_path.parent
        repeat_rows: list[dict[str, Any]] = []
        for rep_pq in sorted(agg_dir.glob("repeat_*/per_query.json")):
            for raw_row in _read_json(rep_pq):
                row = dict(raw_row)
                question = question_by_id[row["question_id"]]
                original_gold = question["source_pages"]
                expanded_groups = _gold_groups(question)
                expanded_gold = sorted({page for group in expanded_groups for page in group})
                row.update(_metrics_for_row(row, original_gold))
                row.update(_metrics_for_row(row, expanded_gold, field_prefix="expanded_", relevant_groups=expanded_groups))
                repeat_rows.append(row)
        if not repeat_rows:
            continue
        run_id = "colab_a100_run3/qwen3.8-27b_27b-trees_ctx64k (3 repeats)" if "colab_a100_run3" in agg_dir.as_posix() else agg_dir.relative_to(RESULTS).as_posix()
        ids = sorted({row["question_id"] for row in repeat_rows})
        validator_pass = [qid for qid in ids if qid not in failed_validator_ids]
        method_names = list(dict.fromkeys(row["method"] for row in repeat_rows))
        compared_methods = [
            name for name in method_names
            if name.startswith(PAGEINDEX_PREFIX) or name == RERANK_METHOD
        ]
        # Average matched repeats within question before question-level bootstrap.
        by_m_q_run: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
        for r in repeat_rows:
            by_m_q_run.setdefault((r["method"], r["question_id"]), {})[int(r["run_index"])] = r
        comparisons = {}
        for method in compared_methods:
            paired_rows = []
            for qid in ids:
                m_trials = by_m_q_run.get((method, qid), {})
                h_trials = by_m_q_run.get((HYBRID_METHOD, qid), {})
                matched = [
                    (m_trials[k], h_trials[k])
                    for k in sorted(set(m_trials) & set(h_trials))
                    if m_trials[k].get("status") != "not_run" and h_trials[k].get("status") != "not_run"
                ]
                for side_name, idx in ((method, 0), (HYBRID_METHOD, 1)):
                    avg_row: dict[str, Any] = {"method": side_name, "question_id": qid}
                    for prefix in ("", "expanded_"):
                        for mk in METRIC_KEYS:
                            key = f"{prefix}{mk}"
                            avg_row[key] = fmean(float(pair[idx][key]) for pair in matched) if matched else None
                    paired_rows.append(avg_row)
            comparisons[method] = {
                "vs_method": HYBRID_METHOD,
                "all_questions": {
                    "original_gold": _paired_comparison(paired_rows, method, "", ids, "run3_repeat_means", "all"),
                    "expanded_gold": _paired_comparison(paired_rows, method, "expanded_", ids, "run3_repeat_means", "all"),
                },
                "validator_pass_only": {
                    "original_gold": _paired_comparison(paired_rows, method, "", validator_pass, "run3_repeat_means", "validator_pass"),
                    "expanded_gold": _paired_comparison(paired_rows, method, "expanded_", validator_pass, "run3_repeat_means", "validator_pass"),
                },
            }
        paired["runs"][run_id] = comparisons

    _write_json(RESULTS / "paired_vs_hybrid.json", paired)
    # Store run-independent baseline tables once at results/ for the generated README.
    _write_json(RESULTS / "baselines.json", {
        "original_gold": _baseline_summary(questions, pages_per_document, "original"),
        "expanded_gold": _baseline_summary(questions, pages_per_document, "expanded"),
    })
    return {"runs": list(all_runs), "paired": paired}


def _group_by_method(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    return grouped


if __name__ == "__main__":
    report = reanalyze_saved_results()
    print(f"Re-scored {len(report['runs'])} saved runs; no model calls were made.")
