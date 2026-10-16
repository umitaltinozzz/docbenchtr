import sys
from types import ModuleType

import pytest

from docbenchtr.evaluate import compute_metrics, summarize
from docbenchtr.retrievers import (
    PAGEINDEX_AGENTIC_RETRIEVAL_INSTRUCTIONS,
    PageIndexRetriever,
    _new_pageindex_client,
    derive_pageindex_retrieval,
    derive_pageindex_tree_search_ranking,
    parse_pageindex_tree_search_response,
)


def test_page_level_recall_and_mrr():
    metrics = compute_metrics([8, 3, 5, 2, 9], [3, 5])
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["reciprocal_rank"] == 0.5


def test_recall_for_multiple_relevant_pages():
    metrics = compute_metrics([1, 2, 7, 4, 5], [4, 5])
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 0.0
    assert metrics["recall@5"] == 1.0


def test_pageindex_ranking_uses_successful_reads_in_selection_order():
    events = [
        {"type": "tool_call", "call_id": "first", "name": "get_page_content",
         "arguments": {"pages": "9-10,3"}},
        {"type": "tool_call", "call_id": "second", "name": "get_page_content",
         "arguments": {"pages": "3,12"}},
        {"type": "tool_call", "call_id": "outline", "name": "get_document_structure",
         "arguments": {"part": 1}},
        {"type": "tool_call", "call_id": "failed", "name": "get_page_content",
         "arguments": {"pages": "40"}},
        # Results can arrive out of order; selection order remains tool-call order.
        {"type": "tool_result", "call_id": "second", "name": "get_page_content",
         "output": '{"success": true, "returned_pages": "3,12"}'},
        {"type": "tool_result", "call_id": "failed", "name": "get_page_content",
         "output": '{"error": "out of range", "success": false}'},
        {"type": "tool_result", "call_id": "first", "name": "get_page_content",
         "output": '{"success": true, "returned_pages": "10,9,3"}'},
    ]

    result = derive_pageindex_retrieval(events)

    assert result["ranked_pages"] == [10, 9, 3, 12]
    assert result["tool_call_count"] == 4
    assert result["page_content_call_count"] == 3
    assert [call["name"] for call in result["tool_calls"]] == [
        "get_page_content", "get_page_content", "get_document_structure",
        "get_page_content",
    ]
    assert result["tool_errors"] == [{"tool": "get_page_content", "error": "out of range"}]


def test_pageindex_tree_search_expands_node_ranges_in_response_order_and_deduplicates():
    tree = [
        {"node_id": "first", "start_index": 1, "end_index": 3, "nodes": []},
        {"node_id": "second", "start_index": 3, "end_index": 5, "nodes": []},
        {"node_id": "third", "start_index": 7, "end_index": 7, "nodes": []},
    ]

    ranking = derive_pageindex_tree_search_ranking(tree, ["second", "first", "third"])

    assert ranking == [3, 4, 5, 1, 2, 7]


def test_pageindex_tree_search_parses_json_and_validates_selected_node_ids():
    thinking, node_ids = parse_pageindex_tree_search_response(
        '{"thinking":"Relevant section first","node_list":["0002","0001"]}',
        {"0001", "0002"},
    )

    assert thinking == "Relevant section first"
    assert node_ids == ["0002", "0001"]

    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_pageindex_tree_search_response("not json", {"0001"})
    with pytest.raises(ValueError, match="unknown node IDs"):
        parse_pageindex_tree_search_response(
            '{"thinking":"","node_list":["missing"]}', {"0001"}
        )


def test_pageindex_agentic_without_ranked_pages_is_zero_ranking():
    class Worker:
        def request(self, command):
            assert command["operation"] == "query"
            return {
                "ok": True,
                "details": {
                    "ranked_pages": [],
                    "tool_calls": [{"name": "get_document_structure", "arguments": {}}],
                    "tool_call_count": 1,
                    "page_content_call_count": 0,
                    "retrieval_trace": [],
                    "tool_errors": [],
                },
                "idle_retries": [],
            }

    retriever = PageIndexRetriever.__new__(PageIndexRetriever)
    retriever.worker = Worker()
    retriever.doc_ids = {"doc": "pi-doc"}
    retriever.document_runs = {}
    retriever.retry_log = []

    result = retriever.retrieve_with_metadata("question", "doc", 5)

    assert result["status"] == "zero_ranking"
    assert result["tool_call_count"] == 1
    assert result["page_content_call_count"] == 0
    assert result["ranked_pages"] == []
    assert "without reading page content" in result["error"]


def test_pageindex_default_variant_omits_custom_instructions(monkeypatch):
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_pageindex = ModuleType("pageindex")
    fake_pageindex.PageIndexClient = Client
    monkeypatch.setitem(sys.modules, "pageindex", fake_pageindex)

    _new_pageindex_client("cache", "default")
    assert "instructions" not in captured
    _new_pageindex_client("cache", "read_pages")
    assert captured["instructions"] == PAGEINDEX_AGENTIC_RETRIEVAL_INSTRUCTIONS


def test_summarize_counts_empty_ranking_as_zero_and_excludes_infrastructure_error():
    def row(question_id, question_type, status, ranked_pages, metrics):
        return {
            "method": "test method",
            "question_id": question_id,
            "question_type": question_type,
            "status": status,
            "ranked_pages": ranked_pages,
            "latency_ms": 10.0,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "parse_failure": None,
            **metrics,
        }

    zero_metrics = {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "reciprocal_rank": 0.0}
    missing_metrics = {"recall@1": None, "recall@3": None, "recall@5": None, "reciprocal_rank": None}
    rows = [
        row("q1", "direct_fact", "zero_ranking", [], zero_metrics),
        row("q2", "direct_fact", "failed", [], missing_metrics),
    ]

    summary = summarize(
        rows,
        [{"question_type": "direct_fact"}, {"question_type": "direct_fact"}],
        [{"method": "test method", "reason": ""}],
        [],
        "cpu",
    )
    method = summary["methods"][0]
    by_type = method["by_question_type"]["direct_fact"]

    assert method["overall"]["recall@5"] == 0.0
    assert method["overall"]["reciprocal_rank"] == 0.0
    assert (method["attempted_count"], method["scored_count"], method["zero_ranking_count"], method["excluded_count"]) == (1, 1, 1, 1)
    assert (by_type["attempted_count"], by_type["scored_count"], by_type["zero_ranking_count"], by_type["excluded_count"]) == (1, 1, 1, 1)
