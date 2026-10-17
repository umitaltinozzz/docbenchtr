"""Aggregate repeated runs without treating repeats as independent questions."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean, stdev

from .reanalyze import (
    HYBRID_METHOD, METRIC_KEYS, _gold_groups, _metrics_for_row, _paired_comparison,
)


def mean_sd(values):
    return {"n": len(values), "mean": fmean(values) if values else None,
            "sd": stdev(values) if len(values) > 1 else None}


def aggregate_repeats(rows, questions, expected_repeats):
    if expected_repeats < 1:
        raise ValueError("repeats must be positive")
    questions_by_id = {q["question_id"]: q for q in questions}
    groups = defaultdict(dict)
    scored = []
    for row in rows:
        key = (row["method"], row["question_id"])
        run = row["run_index"]
        if run in groups[key]:
            raise ValueError(f"Duplicate repeat row: {key}, run {run}")
        question = questions_by_id[row["question_id"]]
        row = {**row, **_metrics_for_row(row, question["source_pages"]),
               **_metrics_for_row(row, question["source_pages"], field_prefix="expanded_",
                                  relevant_groups=_gold_groups(question))}
        groups[key][run] = row
        scored.append(row)
    methods = sorted({r["method"] for r in rows})
    if HYBRID_METHOD not in methods:
        raise ValueError("Repeated comparisons require Hybrid rows")
    expected = set(range(1, expected_repeats + 1))
    for method in methods:
        for qid in questions_by_id:
            if set(groups[(method, qid)]) != expected:
                raise ValueError(f"Incomplete repeats for {method}: {qid}")

    keys = [prefix + metric for prefix in ("", "expanded_") for metric in METRIC_KEYS]
    output = {}
    for method in methods:
        by_run = []
        for run in sorted(expected):
            selected = [r for r in scored if r["method"] == method and r["run_index"] == run]
            by_run.append({"run_index": run,
                           **{key: fmean(vals) if (vals := [r[key] for r in selected if r[key] is not None]) else None for key in keys},
                           "attempted": sum(r["status"] != "not_run" for r in selected),
                           "failed_as_zero": sum(r["status"] == "failed" for r in selected),
                           "excluded": sum(r["status"] == "not_run" for r in selected)})
        per_question, paired_rows = [], []
        for qid in questions_by_id:
            trials = list(groups[(method, qid)].values())
            attempted = [r for r in trials if r["status"] != "not_run"]
            votes = Counter(p for r in attempted if r["status"] != "failed" for p in set(r["ranked_pages"][:5]))
            majority = sorted((p for p, count in votes.items() if count > expected_repeats / 2),
                              key=lambda p: (-votes[p], p))
            per_question.append({
                "question_id": qid, "attempted": len(attempted),
                "failed_as_zero": sum(r["status"] == "failed" for r in trials),
                "majority_top5_pages": majority,
                "top5_page_votes": dict(sorted(votes.items())),
                "majority_full_recall_at_5": sum(r["recall@5"] == 1 for r in attempted) > expected_repeats / 2,
                "metrics": {key: mean_sd([r[key] for r in attempted]) for key in keys},
            })
            # Match repeats within question before averaging, then bootstrap QUESTIONS.
            matched = [(r, groups[(HYBRID_METHOD, qid)][r["run_index"]]) for r in trials
                       if r["status"] != "not_run" and groups[(HYBRID_METHOD, qid)][r["run_index"]]["status"] != "not_run"]
            for name, side in ((method, 0), (HYBRID_METHOD, 1)):
                paired_rows.append({"method": name, "question_id": qid,
                                    **{key: fmean(pair[side][key] for pair in matched) if matched else None for key in keys}})
            if method != HYBRID_METHOD:
                directions = Counter("win" if a["recall@5"] > b["recall@5"] else "loss" if a["recall@5"] < b["recall@5"] else "tie" for a, b in matched)
                per_question[-1]["majority_vs_hybrid_recall_at_5"] = next(
                    (v for v in ("win", "loss", "tie") if directions[v] > expected_repeats / 2), "no_majority")
                per_question[-1]["paired_repeat_count"] = len(matched)
        paired = {}
        if method != HYBRID_METHOD:
            for prefix, label in (("", "original_gold"), ("expanded_", "expanded_gold")):
                paired[label] = _paired_comparison(paired_rows, method, prefix, list(questions_by_id), "run3_repeat_means", "all")
        output[method] = {"per_repeat": by_run,
                          "metrics": {key: mean_sd([r[key] for r in by_run if r[key] is not None]) for key in keys},
                          "per_question": per_question, "paired_vs_hybrid": paired}
    return {
        "repeats": expected_repeats,
        "rules": {"sd": "sample SD across repeat means; null for fewer than two repeats",
                  "failures": "failed attempts score zero; not_run excluded and counted",
                  "majority": "strictly more than N/2 planned repeats; failed attempts cast empty page votes; majority pages are descriptive, not a new ranked method",
                  "paired": "match repeats within question, average, then existing 10000-resample paired question bootstrap and exact two-sided sign test vs Hybrid",
                  "limitations": "questions share targets; repeats do not add independent questions; fixed trees reused across repeats"},
        "methods": output,
    }
