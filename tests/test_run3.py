import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docbenchtr import config, retrievers
from docbenchtr.evaluate import evaluate_question
from docbenchtr.repeats import aggregate_repeats
from docbenchtr.rerank import HybridLLMRerankRetriever, RERANK_METHOD
from docbenchtr.run_repeats import evaluate_repeat


def question(qid="q"):
    return {"question_id": qid, "question": "question", "document_id": "doc",
            "question_type": "direct_fact", "source_pages": [1]}


def test_model_cache_paths_are_portable_and_separate(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCBENCHTR_CACHE_ROOT", str(tmp_path))
    small = config.pageindex_cache_dir("qwen2.5:7b-instruct")
    big = config.pageindex_cache_dir("qwen3.8:27b")
    assert small != big
    assert ":" not in big.relative_to(tmp_path).as_posix()
    assert config.model_slug("a:b") != config.model_slug("a/b")
    with pytest.raises(ValueError, match="read-only"):
        retrievers._new_pageindex_client(str(config.ROOT / ".pageindex/docbenchtr-pilot"))


def test_models_are_independent_in_client_config(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pageindex", SimpleNamespace(PageIndexClient=lambda **kw: kw))
    monkeypatch.setattr(retrievers, "PAGEINDEX_INDEX_MODEL", "index:large")
    monkeypatch.setattr(retrievers, "OLLAMA_MODEL", "search:small")
    result = retrievers._new_pageindex_client("model-cache", "default")
    assert result["index"]["model"] == "ollama_chat/index:large"
    assert result["chat"]["model"] == "ollama_chat/search:small"
    assert "instructions" not in result


def test_cache_rejects_wrong_model_and_requires_provenance(monkeypatch):
    meta = {"id": "pi-doc", "name": "doc.pdf", "status": "completed", "pageNum": 1,
            "metadata": {"index_model": "wrong", "num_ctx": retrievers.OLLAMA_NUM_CTX,
                         "num_predict": retrievers.OLLAMA_NUM_PREDICT,
                         "temperature": 0, "source_sha256": "snapshot"}}
    store = SimpleNamespace(list_metas=lambda: [meta],
                            get_pages=lambda _: [{"page_index": 1, "markdown": "text"}],
                            get_tree=lambda _: [{"start_index": 1, "end_index": 1}])
    client = SimpleNamespace(_api=SimpleNamespace(_store=store),
                             get_tree=lambda *a, **kw: {"status": "completed", "retrieval_ready": True})
    assert retrievers._find_valid_cached_document(client, "doc.pdf", ["text"]) is None
    meta["metadata"]["index_model"] = retrievers.PAGEINDEX_INDEX_MODEL
    assert retrievers._find_valid_cached_document(client, "doc.pdf", ["text"])["id"] == "pi-doc"
    assert retrievers._find_valid_cached_document(client, "doc.pdf", ["text"], "different") is None
    assert retrievers._find_valid_cached_document(client, "doc.pdf", ["text"], "snapshot")["id"] == "pi-doc"
    monkeypatch.setattr(retrievers, "MODEL_DIGESTS", {retrievers.PAGEINDEX_INDEX_MODEL: "weights"})
    assert retrievers._find_valid_cached_document(client, "doc.pdf", ["text"]) is None
    meta["metadata"]["index_model_digest"] = "weights"
    assert retrievers._find_valid_cached_document(client, "doc.pdf", ["text"])["id"] == "pi-doc"
    meta["metadata"]["temperature"] = 1
    assert retrievers._find_valid_cached_document(client, "doc.pdf", ["text"]) is None
    meta["metadata"]["temperature"] = 0
    meta["metadata"]["num_ctx"] += 1
    assert retrievers._find_valid_cached_document(client, "doc.pdf", ["text"]) is None


def test_build_persists_model_in_sdk_manifest_metadata(monkeypatch, tmp_path):
    import sys
    from contextlib import nullcontext
    monkeypatch.setitem(sys.modules, "pageindex.local_api", SimpleNamespace(
        LocalAPI=SimpleNamespace(_extract_page_texts=lambda _: ["text"])))
    monkeypatch.setattr(retrievers, "_find_valid_cached_document", lambda *a: None)
    monkeypatch.setattr(retrievers, "_capture_pageindex_toc_source", lambda *a: nullcontext())
    captured = {}
    def submit(*a, **kw):
        captured.update(kw["metadata"])
        return {"doc_id": "new"}
    client = SimpleNamespace(submit_document=submit,
                             get_tree=lambda *a, **kw: {"status": "completed", "retrieval_ready": True})
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"snapshot")
    result = retrievers._prepare_or_index_document(client, str(pdf))
    assert captured["index_model"] == retrievers.PAGEINDEX_INDEX_MODEL
    assert captured["num_ctx"] == retrievers.OLLAMA_NUM_CTX
    assert result["index_model"] == captured["index_model"]


@pytest.mark.parametrize("page_count", [8, 25])
def test_reranker_exact_top20_and_untouched_tail(page_count):
    order = list(range(1, page_count + 1))
    hybrid = SimpleNamespace(retrieve=lambda q, d, k: order[:k])
    def llm(body):
        assert body["model"] == "same:search"
        assert body["options"] == {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX, "num_predict": config.OLLAMA_NUM_PREDICT}
        assert body["format"]["properties"]["pages"]["items"]["enum"] == order[:20]
        return {"message": {"content": json.dumps({"pages": order[:20][::-1]})},
                "prompt_eval_count": 100, "eval_count": 20}
    reranker = HybridLLMRerankRetriever(hybrid, [{"document_id": "doc", "page_number": n, "text": str(n)} for n in order], llm=llm, model="same:search")
    result = reranker.retrieve_with_metadata("q", "doc", page_count)
    assert result["ranked_pages"] == order[:20][::-1] + order[20:]
    assert result["total_tokens"] == 120
    assert reranker.retrieve("q", "doc", 3) == order[:20][::-1][:3]


@pytest.mark.parametrize("reply", ['not json', '{"pages":[1,1]}', '{"pages":[1,3]}', '{"pages":[true,2]}', '{"pages":[1]}'])
def test_rerank_invalid_output_is_failed_not_hybrid_fallback(reply):
    reranker = HybridLLMRerankRetriever(SimpleNamespace(retrieve=lambda *a: [1, 2]),
        [{"document_id": "doc", "page_number": n, "text": str(n)} for n in [1, 2]],
        llm=lambda _: {"message": {"content": reply}})
    row = evaluate_question(reranker, question(), 2)
    assert row["status"] == "failed"
    assert row["ranked_pages"] == []
    assert row["recall@5"] == 0 and row["reciprocal_rank"] == 0
    assert row["parse_failure"]
    assert row["rerank_candidates"] == [1, 2]


def test_repeats_mean_sd_majority_and_question_level_pairing():
    rows = []
    for run in [1, 2, 3]:
        for q in [question("q1"), question("q2")]:
            rows.append({**q, "method": "Hybrid (RRF)", "run_index": run, "status": "ran", "ranked_pages": [2]})
            rows.append({**q, "method": RERANK_METHOD, "run_index": run,
                         "status": "failed" if run == 3 else "ran", "ranked_pages": [1]})
    result = aggregate_repeats(rows, [question("q1"), question("q2")], 3)["methods"][RERANK_METHOD]
    assert result["metrics"]["recall@5"]["mean"] == pytest.approx(2/3)
    assert result["metrics"]["recall@5"]["sd"] == pytest.approx(0.577350269)
    assert result["per_question"][0]["majority_top5_pages"] == [1]
    assert result["per_question"][0]["majority_full_recall_at_5"] is True
    paired = result["paired_vs_hybrid"]["original_gold"]
    assert paired["n_questions"] == 2  # Not six pseudo-independent observations.
    assert paired["metrics"]["recall@5"]["mean_difference"] == pytest.approx(2/3)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_repeats(rows + rows[:1], [question("q1"), question("q2")], 3)
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate_repeats(rows[:-1], [question("q1"), question("q2")], 3)


def test_repeat_resume_keeps_failed_rows_and_checkpoints_each_question(tmp_path):
    class Stub:
        name = RERANK_METHOD
        calls = 0
        def retrieve(self, *a):
            self.calls += 1
            raise ValueError("recorded failure")
    stub = Stub()
    identity = {"model": "test"}
    rows = evaluate_repeat([stub], [question()], {"doc": 2}, tmp_path, 1, identity)
    assert stub.calls == 1 and rows[0]["run_index"] == 1
    resumed = evaluate_repeat([stub], [question()], {"doc": 2}, tmp_path, 1, identity, resume=True)
    assert resumed == rows and stub.calls == 1
    with pytest.raises(ValueError, match="differs"):
        evaluate_repeat([stub], [question()], {"doc": 2}, tmp_path, 1, {"model": "other"}, resume=True)


def test_paired_statistics_match_only_common_attempted_repeats():
    rows = []
    for run in [1, 2, 3]:
        rows.append({**question(), "method": RERANK_METHOD, "run_index": run,
                     "status": "not_run" if run == 3 else "ran", "ranked_pages": [1]})
        rows.append({**question(), "method": "Hybrid (RRF)", "run_index": run,
                     "status": "ran", "ranked_pages": [2] if run == 3 else [1]})
    method = aggregate_repeats(rows, [question()], 3)["methods"][RERANK_METHOD]
    assert method["per_question"][0]["paired_repeat_count"] == 2
    assert method["paired_vs_hybrid"]["original_gold"]["metrics"]["recall@5"]["mean_difference"] == 0
    assert method["per_repeat"][2]["excluded"] == 1


def test_cache_copy_preserves_original_bytes(monkeypatch, tmp_path):
    from docbenchtr import cache
    legacy = tmp_path / ".pageindex/docbenchtr-pilot"
    (legacy / "docs/pi-doc").mkdir(parents=True)
    manifest = {"docs": {"pi-doc": {"id": "pi-doc", "metadata": {}}}}
    raw = json.dumps(manifest).encode()
    (legacy / "manifest.json").write_bytes(raw)
    (legacy / "docs/pi-doc/doc.json").write_bytes(raw)
    (legacy / "docs/pi-doc/tree.json").write_bytes(b"[]")
    target = tmp_path / "models/qwen"
    monkeypatch.setattr(cache, "ROOT", tmp_path)
    monkeypatch.setattr(cache, "PAGEINDEX_INDEX_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setattr(cache, "OLLAMA_NUM_CTX", 32768)
    monkeypatch.setattr(cache, "OLLAMA_NUM_PREDICT", 4096)
    monkeypatch.setattr(cache, "pageindex_cache_dir", lambda: target)
    assert cache.prepare_cache() == target
    assert (legacy / "manifest.json").read_bytes() == raw
    assert json.loads((target / "manifest.json").read_text())["docs"]["pi-doc"]["metadata"]["index_model"] == "qwen2.5:7b-instruct"


def test_run3_notebook_code_cells_compile_and_protocol_is_explicit():
    notebook = json.loads((Path(__file__).resolve().parents[1] / "colab/run3_a100.ipynb").read_text(encoding="utf-8"))
    sources = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"notebook-cell-{i}", "exec")
            sources.append(cell["source"])
            assert cell["outputs"] == []
    source = "\n".join(sources)
    assert "INDEX_MODEL = 'qwen3.8:27b'" in source
    assert "SEARCH_MODEL = 'qwen3.8:27b'" in source
    assert "NUM_CTX = 65536" in source and "NUM_PREDICT = 32768" in source
    assert "REPEATS = 3" in source and "MPLBACKEND='Agg'" in source
    assert "'checkout', '--detach', REPO_COMMIT" in source
    assert "'--resume'" in source


def test_runner_seven_methods_three_repeats_and_resume(monkeypatch, tmp_path):
    from docbenchtr import run_repeats as runner
    calls = []
    class Stub:
        device = "cpu"
        document_runs = {}
        doc_ids = {"doc": "pi-doc"}
        def __init__(self, name): self.name = name
        def retrieve(self, query, document_id, k):
            calls.append(self.name)
            return [1, 2][:k]
        def results_metadata(self): return {"stub": True}
        def close(self): pass
    monkeypatch.setattr(runner, "BM25Retriever", lambda pages, stem=False: Stub("BM25 stem" if stem else "BM25"))
    monkeypatch.setattr(runner, "DenseRetriever", lambda pages: Stub("Dense"))
    monkeypatch.setattr(runner, "HybridRetriever", lambda *a: Stub("Hybrid (RRF)"))
    monkeypatch.setattr(runner, "PageIndexRetriever", lambda *a, **kw: Stub("PageIndex agentic (default instructions)"))
    monkeypatch.setattr(runner, "PageIndexLLMTreeSearchRetriever", lambda *a: Stub("PageIndex LLM Tree Search"))
    monkeypatch.setattr(runner, "HybridLLMRerankRetriever", lambda *a: Stub(RERANK_METHOD))
    monkeypatch.setattr(runner, "download_documents", lambda: [])
    monkeypatch.setattr(runner, "extract_pdf_pages", lambda _: [
        {"document_id": "doc", "page_number": n, "text": str(n)} for n in [1, 2]])
    monkeypatch.setattr(runner, "draft_questions", lambda _: ([question("q1"), question("q2")], []))
    monkeypatch.setattr(runner, "DOCUMENTS", [{"document_id": "doc", "filename": "doc.pdf"}])
    monkeypatch.setattr(runner, "ollama_status", lambda _: (True, "stub"))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "1" * 40)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/sources.json").write_text("[]")
    cache = tmp_path / "cache"
    (cache / "docs/pi-doc").mkdir(parents=True)
    (cache / "docs/pi-doc/tree.json").write_text("[]")
    (cache / "manifest.json").write_text('{"docs":{}}')
    monkeypatch.setattr(runner, "prepare_cache", lambda: cache)
    args = ["--repeats", "3", "--output-root", str(tmp_path / "results"), "--resume"]
    assert runner.main(args) == 0
    assert len(calls) == 42
    output = runner.config_output_dir(tmp_path / "results")
    assert len(json.loads((output / "aggregate.json").read_text())["methods"]) == 7
    original_summary = (output / "repeat_001/summary.json").read_bytes()
    for i in [1, 2, 3]:
        rows = json.loads((output / f"repeat_{i:03d}/per_query.json").read_text())
        assert len(rows) == 14 and {r["run_index"] for r in rows} == {i}
    assert runner.main(args) == 0
    assert len(calls) == 42
    assert (output / "repeat_001/summary.json").read_bytes() == original_summary
    # Both configuration mismatch and orphaned results fail before model setup.
    def forbidden(*args, **kwargs):
        raise AssertionError("A model was constructed before checkpoint validation")
    monkeypatch.setattr(runner, "DenseRetriever", forbidden)
    monkeypatch.setattr(runner, "OLLAMA_NUM_CTX", runner.OLLAMA_NUM_CTX + 1)
    mismatched_output = runner.config_output_dir(tmp_path / "results")
    orphan = mismatched_output / "repeat_001"
    orphan.mkdir(parents=True)
    (orphan / "per_query.json").write_text("[]")
    with pytest.raises(ValueError, match="Nonempty output"):
        runner.main(args)
    (orphan / "configuration.json").write_text('{}')
    with pytest.raises(ValueError, match="differs"):
        runner.main(args)


def test_checkpoint_csv_preserves_structured_fields(tmp_path):
    import csv
    from docbenchtr.evaluate import write_rows
    row = {"method": RERANK_METHOD, "rerank_candidates": [3, 1],
           "transport_events": [{"event": "shortened", "original_bytes": 99}]}
    write_rows([row], tmp_path)
    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as source:
        result = next(csv.DictReader(source))
    for key in ("rerank_candidates", "transport_events"):
        assert json.loads(result[key]) == row[key]
