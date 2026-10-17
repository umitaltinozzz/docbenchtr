from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import time
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import snowballstemmer
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT, PAGEINDEX_INDEX_MODEL, MODEL_DIGESTS, ROOT,
)


TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+", re.IGNORECASE)
PAGEINDEX_AGENTIC_RETRIEVAL_INSTRUCTIONS = (
    "RETRIEVAL REQUIREMENT: For every question about a document, call "
    "get_page_content and read its returned page text before answering. "
    "A document outline or node summary is for navigation only and is not "
    "evidence. If you call get_document_structure, follow it with "
    "get_page_content for the relevant page range. For a small document, "
    "you may call get_page_content directly. Do not answer unless a "
    "get_page_content result was received."
)
TURKISH_TRANSLATION = str.maketrans({"I": "ı", "İ": "i"})


def tokenize(text: str, stem: bool = False) -> list[str]:
    tokens = [token.translate(TURKISH_TRANSLATION).casefold() for token in TOKEN_RE.findall(text)]
    if stem:
        stemmer = snowballstemmer.stemmer("turkish")
        tokens = stemmer.stemWords(tokens)
    return tokens


class Retriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, query: str, document_id: str, k: int = 5) -> list[int]:
        """Return ranked, one-based physical PDF page numbers."""


class BM25Retriever(Retriever):
    def __init__(self, pages: list[dict[str, Any]], *, stem: bool = False, name: str | None = None):
        self.name = name or ("BM25 (Turkish stemmed)" if stem else "BM25")
        self.stem = stem
        self.page_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.models: dict[str, BM25Okapi] = {}
        for page in pages:
            if page["text"].strip():
                self.page_groups[page["document_id"]].append(page)
        for doc_id, doc_pages in self.page_groups.items():
            self.models[doc_id] = BM25Okapi([tokenize(page["text"], stem) for page in doc_pages])

    def retrieve(self, query: str, document_id: str, k: int = 5) -> list[int]:
        doc_pages = self.page_groups[document_id]
        scores = self.models[document_id].get_scores(tokenize(query, self.stem))
        order = sorted(range(len(doc_pages)), key=lambda i: (-float(scores[i]), doc_pages[i]["page_number"]))
        return [doc_pages[index]["page_number"] for index in order[:k]]


class DenseRetriever(Retriever):
    def __init__(
        self, pages: list[dict[str, Any]], model_name: str = EMBEDDING_MODEL,
        model_revision: str = EMBEDDING_MODEL_REVISION,
    ):
        self.name = "Dense (multilingual-e5-base)"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_revision = model_revision
        self.model = SentenceTransformer(model_name, revision=model_revision, device=self.device)
        self.page_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.embeddings: dict[str, np.ndarray] = {}
        for page in pages:
            if page["text"].strip():
                self.page_groups[page["document_id"]].append(page)
        for doc_id, doc_pages in self.page_groups.items():
            passages = ["passage: " + page["text"] for page in doc_pages]
            self.embeddings[doc_id] = self.model.encode(
                passages, batch_size=16, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            )

    def retrieve(self, query: str, document_id: str, k: int = 5) -> list[int]:
        query_embedding = self.model.encode(
            ["query: " + query], convert_to_numpy=True, normalize_embeddings=True,
        )[0]
        scores = self.embeddings[document_id] @ query_embedding
        doc_pages = self.page_groups[document_id]
        order = sorted(range(len(doc_pages)), key=lambda i: (-float(scores[i]), doc_pages[i]["page_number"]))
        return [doc_pages[index]["page_number"] for index in order[:k]]


class HybridRetriever(Retriever):
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.name = "Hybrid (RRF)"
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def retrieve(self, query: str, document_id: str, k: int = 5) -> list[int]:
        # Full per-document rankings avoid a top-k cutoff changing RRF scores.
        page_count = len(self.bm25.page_groups[document_id])
        rankings = [
            self.bm25.retrieve(query, document_id, page_count),
            self.dense.retrieve(query, document_id, page_count),
        ]
        scores: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, page_number in enumerate(ranking, start=1):
                scores[page_number] += 1 / (self.rrf_k + rank)
        return [page for page, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]


def ollama_status(model: str = OLLAMA_MODEL, timeout: float = 2.0) -> tuple[bool, str]:
    executable = shutil.which("ollama")
    try:
        request = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            tags = json.loads(response.read().decode("utf-8"))
        available = {entry.get("name") for entry in tags.get("models", [])}
        if model in available:
            return True, f"Ollama is serving {model}"
        return False, f"Ollama is reachable but model {model!r} is not installed; found {sorted(available)}"
    except Exception as error:
        cli_state = "Ollama executable was not found on PATH; " if not executable else ""
        return False, f"{cli_state}Ollama API {OLLAMA_BASE_URL}/api/tags is unreachable: {type(error).__name__}: {error}"


def _new_pageindex_client(storage_path: str, instruction_variant: str = "read_pages"):
    # The committed 7B cache is an immutable reference for earlier experiments.
    if Path(storage_path).resolve() == (ROOT / ".pageindex" / "docbenchtr-pilot").resolve():
        raise ValueError("Use a model-specific cache folder; the published 7B trees are read-only")
    from pageindex import PageIndexClient
    index_backend = {"api_base": OLLAMA_BASE_URL}
    chat_backend = {"api_base": OLLAMA_BASE_URL}
    config = dict(
        index={
            "model": f"ollama_chat/{PAGEINDEX_INDEX_MODEL}",
            "backend": index_backend,
            "storage_path": storage_path,
            "optimize": "merge",
            "summary_concurrency": 1,
        },
        chat={
            "model": f"ollama_chat/{OLLAMA_MODEL}",
            "backend": chat_backend,
        },
    )
    if instruction_variant == "read_pages":
        config["instructions"] = PAGEINDEX_AGENTIC_RETRIEVAL_INSTRUCTIONS
    elif instruction_variant != "default":
        raise ValueError(f"Unknown PageIndex instruction variant: {instruction_variant!r}")
    return PageIndexClient(**config)


def pageindex_local_preflight(storage_path: str) -> str:
    """Initialize the SDK's local config without submitting documents or making model calls."""
    client = _new_pageindex_client(storage_path)
    return f"Local SDK config initialized with index model {client.index_model!r} and chat model {client.chat_model!r}."


class PageIndexRetriever(Retriever):
    """PageIndex local SDK adapter with page-read traces and query usage."""

    def __init__(
        self,
        pdf_paths: dict[str, str],
        storage_path: str,
        *,
        unavailable_reason: str | None = None,
        instruction_variant: str = "read_pages",
    ):
        if instruction_variant not in ("default", "read_pages"):
            raise ValueError(f"Unknown PageIndex instruction variant: {instruction_variant!r}")
        self.instruction_variant = instruction_variant
        self.name = (
            "PageIndex agentic (default instructions)" if instruction_variant == "default"
            else "PageIndex agentic (read-pages instruction)"
        )
        self.client = None
        self.doc_ids: dict[str, str] = {}
        self.document_runs: dict[str, dict[str, Any]] = {}
        self.model = f"ollama_chat/{OLLAMA_MODEL}"
        self.ranking_derivation = (
            "Successful get_page_content tool results, in tool-call order; "
            "within each call, use the returned_pages order; deduplicate by "
            "first occurrence. Outline-only calls are not ranked."
        )
        self.llm_idle_timeout_seconds = 1800
        self.llm_idle_max_retries = 3
        self.retry_log: list[dict[str, Any]] = []
        self.runtime_reason = unavailable_reason or ""
        self.worker: PageIndexWorker | None = None
        if unavailable_reason:
            for doc_id in pdf_paths:
                self.document_runs[doc_id] = {
                    "document_id": doc_id,
                    "status": "not_run",
                    "build_time_ms": None,
                    "toc_source": None,
                    "error": unavailable_reason,
                }
            return

        resolved_storage = Path(storage_path).resolve()
        resolved_storage.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.worker = PageIndexWorker(str(resolved_storage), instruction_variant)
        except Exception as error:
            reason = f"PageIndex worker setup failed: {type(error).__name__}: {error}"
            self.runtime_reason = reason
            for doc_id in pdf_paths:
                self.document_runs[doc_id] = {
                    "document_id": doc_id,
                    "status": "failed",
                    "build_time_ms": None,
                    "toc_source": None,
                    "error": reason,
                }
            return

        for doc_id, pdf_path in pdf_paths.items():
            started = time.perf_counter()
            toc_source = None
            print(f"Checking PageIndex cache for {doc_id}...", flush=True)
            try:
                result = self.worker.request(
                    {"operation": "prepare_document", "document_id": doc_id, "pdf_path": pdf_path},
                )
                idle_retries = result.get("idle_retries", [])
                self.retry_log.extend(idle_retries)
                if not result.get("ok"):
                    raise RuntimeError(result.get("error") or "PageIndex indexing failed")
                self.doc_ids[doc_id] = result["doc_id"]
                elapsed_ms = (time.perf_counter() - started) * 1000
                self.document_runs[doc_id] = {
                    "document_id": doc_id,
                    "status": "reused" if result.get("reused") else "indexed",
                    "build_time_ms": result.get("build_time_ms"),
                    "prepare_wall_time_ms": elapsed_ms,
                    "reuse_validation_time_ms": elapsed_ms if result.get("reused") else None,
                    "toc_source": result.get("toc_source"),
                    "toc_source_error": result.get("toc_source_error"),
                    "pageindex_doc_id": result["doc_id"],
                    "source_sha256": result.get("source_sha256"),
                    "index_model": result.get("index_model"),
                    "idle_retry_count": len(idle_retries),
                    "error": None,
                }
                if result.get("reused"):
                    print(f"Reused verified PageIndex tree for {doc_id}.", flush=True)
                else:
                    build_ms = self.document_runs[doc_id]["build_time_ms"]
                    print(f"Indexed {doc_id} ({build_ms / 1000:.1f}s).", flush=True)
            except Exception as error:
                self.document_runs[doc_id] = {
                    "document_id": doc_id,
                    "status": "failed",
                    "build_time_ms": (time.perf_counter() - started) * 1000,
                    "toc_source": toc_source,
                    "error": str(error),
                }
                print(f"PageIndex indexing failed for {doc_id}: {error}", flush=True)

    def results_metadata(self) -> dict[str, Any]:
        try:
            from importlib.metadata import version
            sdk_version = version("pageindex")
        except Exception:
            sdk_version = None
        return {
            "mode": "local",
            "sdk_version": sdk_version,
            "index_model": f"ollama_chat/{PAGEINDEX_INDEX_MODEL}",
            "chat_model": self.model,
            "transport": "native Ollama /api/chat; buffered SDK stream bridge",
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": 0,
            "pdfium_parser_workers": 1,
            "ollama_base_url": OLLAMA_BASE_URL,
            "llm_http_idle_timeout_seconds": self.llm_idle_timeout_seconds,
            "llm_http_idle_max_retries": self.llm_idle_max_retries,
            "llm_idle_retries": self.retry_log,
            "index_build_time_definition": (
                "Wall-clock around local SDK submit_document and retrieval-readiness verification; "
                "null when a verified cached tree is reused."
            ),
            "query_latency_definition": (
                "Wall-clock for the PageIndex agent run plus citation extraction."
            ),
            "cache_reuse_validation": (
                "Completed manifest entry, exact extracted PDF page text, valid node page ranges, "
                "and a retrieval-ready SDK get_tree response."
            ),
            "ranking_derivation": self.ranking_derivation,
            "token_usage_source": (
                "PageIndex SDK stream's raw OpenAI-compatible response usage; "
                "null when the provider does not return counts."
            ),
            "documents": list(self.document_runs.values()),
        }

    def retrieve(self, query: str, document_id: str, k: int = 5) -> list[int]:
        return self.retrieve_with_metadata(query, document_id, k)["ranked_pages"]

    def retrieve_with_metadata(
        self, query: str, document_id: str, k: int = 5, *, question_id: str | None = None,
    ) -> dict[str, Any]:
        if self.worker is None or document_id not in self.doc_ids:
            run = self.document_runs.get(document_id, {})
            raise RuntimeError(run.get("error") or f"PageIndex has no index for {document_id}")

        try:
            result = self.worker.request(
                {
                    "operation": "query",
                    "query": query,
                    "doc_id": self.doc_ids[document_id],
                    "k": k,
                    "question_id": question_id,
                }
            )
            idle_retries = result.get("idle_retries", [])
            self.retry_log.extend(idle_retries)
        except Exception as error:
            raise PageIndexQueryError(
                str(error), details={}
            ) from error
        if not result.get("ok"):
            details = result.get("details") or {}
            details["idle_retries"] = idle_retries
            raise PageIndexQueryError(
                result.get("error") or "PageIndex query failed",
                details=details,
            )
        details = result["details"]
        details["idle_retries"] = idle_retries
        if not details.get("ranked_pages"):
            details["status"] = "zero_ranking"
            if details.get("page_content_call_count", 0) == 0:
                details["error"] = (
                    "The agent completed without reading page content; no page ranking was produced."
                )
            else:
                details["error"] = (
                    "Page-content tool calls produced no ranked pages; no retrieval score was recorded."
                )
        return details

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()


class PageIndexLLMTreeSearchRetriever(Retriever):
    """Non-agentic PageIndex tree selection over the existing cached trees."""

    def __init__(
        self,
        pageindex: PageIndexRetriever,
        storage_path: str,
    ):
        self.name = "PageIndex LLM Tree Search"
        self.doc_ids = pageindex.doc_ids
        self.document_runs = pageindex.document_runs
        self.storage_path = Path(storage_path).resolve()
        self.model = OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL
        self._tree_cache: dict[str, list[dict[str, Any]]] = {}
        self.ranking_derivation = (
            "Expand each selected node's inclusive one-based start_index..end_index "
            "range in returned node_list order; deduplicate pages by first occurrence."
        )

    def retrieve(self, query: str, document_id: str, k: int = 5) -> list[int]:
        return self.retrieve_with_metadata(query, document_id, k)["ranked_pages"]

    def retrieve_with_metadata(
        self,
        query: str,
        document_id: str,
        k: int = 5,
        *,
        question_id: str | None = None,
    ) -> dict[str, Any]:
        pageindex_doc_id = self.doc_ids.get(document_id)
        if pageindex_doc_id is None:
            raise RuntimeError(f"PageIndex has no cached tree for {document_id}")

        tree = self._get_cached_tree(pageindex_doc_id)
        projected_tree = project_pageindex_tree(tree)
        known_node_ids = {
            node["node_id"] for node in flatten_pageindex_tree(tree)
        }
        raw, usage = _ollama_tree_search(
            self.model,
            self.base_url,
            query,
            projected_tree,
            known_node_ids,
        )
        try:
            thinking, node_list = parse_pageindex_tree_search_response(
                raw, known_node_ids
            )
        except ValueError as error:
            return {
                "status": "zero_ranking",
                "error": str(error),
                "parse_failure": f"{error}; response={raw[:500]!r}",
                "tree_search_thinking": None,
                "tree_search_node_ids": [],
                "ranked_pages": [],
                "tool_call_count": None,
                "page_content_call_count": None,
                **usage,
            }

        return {
            "status": "ran",
            "error": None,
            "parse_failure": None,
            "tree_search_thinking": thinking,
            "tree_search_node_ids": node_list,
            "ranked_pages": derive_pageindex_tree_search_ranking(tree, node_list)[:k],
            "tool_call_count": None,
            "page_content_call_count": None,
            **usage,
        }

    def _get_cached_tree(self, pageindex_doc_id: str) -> list[dict[str, Any]]:
        if pageindex_doc_id not in self._tree_cache:
            tree_path = self.storage_path / "docs" / pageindex_doc_id / "tree.json"
            tree = json.loads(tree_path.read_text(encoding="utf-8"))
            if not isinstance(tree, list) or not tree:
                raise ValueError(f"Cached PageIndex tree is empty or invalid: {tree_path}")
            # Validate required fields before sending the full cached tree to Ollama.
            flatten_pageindex_tree(tree)
            self._tree_cache[pageindex_doc_id] = tree
        return self._tree_cache[pageindex_doc_id]

    def results_metadata(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "model": self.model,
            "ollama_endpoint": f"{self.base_url}/api/chat",
            "response_format": "Ollama JSON schema with thinking:string and node_list:string[]",
            "ranking_derivation": self.ranking_derivation,
            "parse_failures": "Retained as failed per-query rows with the parse error and response excerpt.",
        }


class PageIndexWorker:
    """Persistent SDK worker; SDK operations run to completion without deadlines."""

    def __init__(self, storage_path: str, instruction_variant: str = "read_pages"):
        self.storage_path = storage_path
        self.instruction_variant = instruction_variant
        self.context = multiprocessing.get_context("spawn")
        self.process = None
        self.connection = None

    def _start(self) -> None:
        parent, child = self.context.Pipe(duplex=True)
        process = self.context.Process(
            target=_pageindex_worker_loop,
            args=(child, self.storage_path, self.instruction_variant),
            name="docbenchtr-pageindex-worker",
        )
        process.start()
        child.close()
        self.process = process
        self.connection = parent
        ready = parent.recv()
        if not ready.get("ready"):
            reason = ready.get("error") or "PageIndex worker failed during startup"
            self._terminate()
            raise RuntimeError(reason)

    def request(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.process is None or not self.process.is_alive():
            self._terminate()
            self._start()
        assert self.connection is not None
        self.connection.send(command)
        try:
            return self.connection.recv()
        except (EOFError, OSError) as error:
            self._terminate()
            raise RuntimeError(f"PageIndex worker exited before returning a result: {error}") from error

    def _terminate(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except OSError:
                pass
        if self.process is not None:
            if self.process.is_alive():
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    self.process.terminate()
            self.process.join(timeout=5)
        self.connection = None
        self.process = None

    def close(self) -> None:
        if self.process is not None and self.process.is_alive() and self.connection is not None:
            try:
                self.connection.send({"operation": "stop"})
                self.process.join()
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._terminate()


def _pageindex_worker_loop(connection, storage_path: str, instruction_variant: str = "read_pages") -> None:
    try:
        from importlib.metadata import version
        if version("pageindex") != "0.4.0":
            raise RuntimeError("The native adapter requires pageindex==0.4.0")
        from .ollama_native import install
        install()
        _install_pageindex_llm_idle_retries()
        client = _new_pageindex_client(storage_path, instruction_variant)
        connection.send({"ready": True})
    except Exception as error:
        connection.send({"ready": False, "error": f"{type(error).__name__}: {error}"})
        connection.close()
        return

    try:
        while True:
            try:
                command = connection.recv()
            except EOFError:
                break
            operation = command.get("operation")
            if operation == "stop":
                break
            retry_log_start = len(_PAGEINDEX_RETRY_LOG)
            source: dict[str, Any] = {}
            try:
                if operation == "prepare_document":
                    label = f"index:{command['document_id']}"
                    _PAGEINDEX_REQUEST_CONTEXT.update(operation="index", label=label)
                    response = _prepare_or_index_document(client, command["pdf_path"])
                elif operation == "query":
                    label = f"query:{command.get('question_id') or command['doc_id']}"
                    _PAGEINDEX_REQUEST_CONTEXT.update(operation="query", label=label)
                    response = {"ok": True, "details": _run_pageindex_query(
                        client,
                        command["query"],
                        command["doc_id"],
                        command["k"],
                    )}
                else:
                    raise ValueError(f"Unknown PageIndex worker operation: {operation!r}")
            except Exception as error:
                response = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "details": getattr(error, "details", {}),
                    "toc_source": source.get("toc_source"),
                }
            response["idle_retries"] = _PAGEINDEX_RETRY_LOG[retry_log_start:]
            try:
                connection.send(response)
            except (BrokenPipeError, EOFError, OSError):
                break
            finally:
                _PAGEINDEX_REQUEST_CONTEXT.update(operation="idle", label="idle")
    finally:
        connection.close()


_PAGEINDEX_REQUEST_CONTEXT: dict[str, str] = {"operation": "idle", "label": "idle"}
_PAGEINDEX_RETRY_LOG: list[dict[str, Any]] = []


def _prepare_or_index_document(client: Any, pdf_path: str) -> dict[str, Any]:
    """Reuse a cache only when its PDF page text and tree validate exactly."""
    import time

    from pageindex.local_api import LocalAPI

    pdf = Path(pdf_path)
    source_sha256 = hashlib.sha256(pdf.read_bytes()).hexdigest()
    source_pages = LocalAPI._extract_page_texts(str(pdf))
    cached = _find_valid_cached_document(client, pdf.name, source_pages, source_sha256)
    if cached is not None:
        toc_source = cached.get("toc_source")
        toc_source_error = None
        if not toc_source:
            try:
                from pageindex.flash.main import extract_toc

                toc_source = extract_toc(
                    str(pdf), use_embedded_toc=True, workers=1
                ).get("toc_source")
            except Exception as error:
                toc_source_error = f"{type(error).__name__}: {error}"
        return {
            "ok": True,
            "doc_id": cached["id"],
            "reused": True,
            "build_time_ms": None,
            "toc_source": toc_source,
            "toc_source_error": toc_source_error,
            "source_sha256": source_sha256,
            "index_model": PAGEINDEX_INDEX_MODEL,
        }

    started = time.perf_counter()
    source: dict[str, Any] = {}
    metadata = {"source_sha256": source_sha256, "index_model": PAGEINDEX_INDEX_MODEL,
                "index_model_digest": MODEL_DIGESTS.get(PAGEINDEX_INDEX_MODEL),
                "num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT,
                "temperature": 0, "transport": "ollama_native"}
    with _capture_pageindex_toc_source(source, metadata):
        submitted = client.submit_document(
            str(pdf),
            wait=True,
            metadata=metadata,
        )
    tree = client.get_tree(submitted["doc_id"], include_text=False)
    if tree.get("status") != "completed" or not tree.get("retrieval_ready"):
        raise RuntimeError(f"PageIndex tree was not retrieval-ready: {tree.get('status')!r}")
    return {
        "ok": True,
        "doc_id": submitted["doc_id"],
        "reused": False,
        "build_time_ms": (time.perf_counter() - started) * 1000,
        "toc_source": source.get("toc_source"),
        "toc_source_error": None,
        "source_sha256": source_sha256,
        "index_model": PAGEINDEX_INDEX_MODEL,
    }


def _find_valid_cached_document(
    client: Any, filename: str, source_pages: list[str], source_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Find a complete same-name cache whose stored text and node spans match."""
    store = getattr(getattr(client, "_api", None), "_store", None)
    if store is None:
        return None

    filename_re = re.compile(rf"^{re.escape(Path(filename).stem)}(?:_\d+)?\.pdf$", re.IGNORECASE)
    for metadata in store.list_metas():
        provenance = metadata.get("metadata") or {}
        if (provenance.get("index_model") != PAGEINDEX_INDEX_MODEL
                or provenance.get("num_ctx") != OLLAMA_NUM_CTX
                or provenance.get("num_predict") != OLLAMA_NUM_PREDICT
                or provenance.get("temperature") != 0):
            continue
        if source_sha256 is not None and provenance.get("source_sha256") != source_sha256:
            continue
        if MODEL_DIGESTS.get(PAGEINDEX_INDEX_MODEL) and provenance.get("index_model_digest") != MODEL_DIGESTS[PAGEINDEX_INDEX_MODEL]:
            continue
        if not filename_re.fullmatch(metadata.get("name", "")):
            continue
        if metadata.get("status") != "completed" or metadata.get("pageNum") != len(source_pages):
            continue
        doc_id = metadata.get("id")
        stored_pages = store.get_pages(doc_id)
        tree = store.get_tree(doc_id)
        if not isinstance(stored_pages, list) or len(stored_pages) != len(source_pages):
            continue
        if [page.get("page_index") for page in stored_pages] != list(range(1, len(source_pages) + 1)):
            continue
        if [page.get("markdown", "") for page in stored_pages] != source_pages:
            continue
        if not isinstance(tree, list) or not tree or not _valid_pageindex_tree_ranges(tree, len(source_pages)):
            continue
        try:
            envelope = client.get_tree(doc_id, include_text=False)
        except Exception:
            continue
        if envelope.get("status") != "completed" or not envelope.get("retrieval_ready"):
            continue
        toc_source = (metadata.get("metadata") or {}).get("toc_source")
        return {**metadata, "toc_source": toc_source}
    return None


def _valid_pageindex_tree_ranges(tree: list[dict[str, Any]], page_count: int) -> bool:
    stack = list(tree)
    if not stack:
        return False
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            return False
        start, end = node.get("start_index"), node.get("end_index")
        if not isinstance(start, int) or not isinstance(end, int) or not (1 <= start <= end <= page_count):
            return False
        children = node.get("nodes") or []
        if not isinstance(children, list):
            return False
        stack.extend(children)
    return True


class PageIndexLLMIdleRetriesExhausted(TimeoutError):
    """One local-model request stayed idle for 30 minutes through all retries."""

    # PageIndex's own completion loop treats 401 as non-retryable. This keeps
    # it from retrying the same exhausted request another ten times.
    status_code = 401


def _is_http_timeout(error: BaseException, httpx_timeout_type: type[BaseException]) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (httpx_timeout_type, TimeoutError)):
            return True
        if type(current).__name__.lower() in {
            "timeout", "readtimeout", "connecttimeout", "writetimeout", "pooltimeout",
        }:
            return True
        current = current.__cause__ or current.__context__
    return False


def _log_pageindex_llm_retry(retry_number: int, error: BaseException) -> None:
    log_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operation": _PAGEINDEX_REQUEST_CONTEXT["operation"],
        "label": _PAGEINDEX_REQUEST_CONTEXT["label"],
        "retry": retry_number,
        "error_type": type(error).__name__,
    }
    _PAGEINDEX_RETRY_LOG.append(log_entry)
    print(
        "PageIndex LLM HTTP request was idle for 30 minutes; "
        f"retry {retry_number}/3 ({_PAGEINDEX_REQUEST_CONTEXT['operation']} "
        f"{_PAGEINDEX_REQUEST_CONTEXT['label']}; {type(error).__name__}; "
        f"{log_entry['timestamp_utc']}).",
        flush=True,
    )


def _install_pageindex_llm_idle_retries() -> None:
    """Set a 30-minute HTTP inactivity timeout and retry idle LLM calls 3 times."""
    import httpx
    import litellm

    if getattr(litellm, "_docbenchtr_idle_retries_installed", False):
        return
    original_completion = litellm.completion
    original_acompletion = litellm.acompletion
    timeout_type = httpx.TimeoutException

    def with_idle_timeout(kwargs: dict[str, Any]) -> dict[str, Any]:
        updated = dict(kwargs)
        updated["timeout"] = 1800
        return updated

    def completion(*args, **kwargs):
        call_kwargs = with_idle_timeout(kwargs)
        for retry_count in range(4):
            try:
                return original_completion(*args, **call_kwargs)
            except Exception as error:
                if not _is_http_timeout(error, timeout_type):
                    raise
                if retry_count == 3:
                    raise PageIndexLLMIdleRetriesExhausted(
                        "HTTP request remained idle for 30 minutes after 3 retries"
                    ) from error
                _log_pageindex_llm_retry(retry_count + 1, error)

    async def acompletion(*args, **kwargs):
        call_kwargs = with_idle_timeout(kwargs)

        async def issue_request():
            for retry_count in range(4):
                try:
                    return await original_acompletion(*args, **call_kwargs)
                except Exception as error:
                    if not _is_http_timeout(error, timeout_type):
                        raise
                    if retry_count == 3:
                        raise PageIndexLLMIdleRetriesExhausted(
                            "HTTP request remained idle for 30 minutes after 3 retries"
                        ) from error
                    _log_pageindex_llm_retry(retry_count + 1, error)
            raise AssertionError("unreachable retry loop")

        response = await issue_request()
        if not call_kwargs.get("stream") or not hasattr(response, "__aiter__"):
            return response

        async def retrying_stream():
            for request_attempt in range(4):
                buffered_chunks = []
                try:
                    current_stream = response if request_attempt == 0 else await original_acompletion(
                        *args, **call_kwargs
                    )
                    async for chunk in current_stream:
                        buffered_chunks.append(chunk)
                except Exception as error:
                    if not _is_http_timeout(error, timeout_type):
                        raise
                    if request_attempt == 3:
                        raise PageIndexLLMIdleRetriesExhausted(
                            "stream stayed idle for 30 minutes after 3 retries"
                        ) from error
                    _log_pageindex_llm_retry(request_attempt + 1, error)
                    continue
                for chunk in buffered_chunks:
                    yield chunk
                return

        return retrying_stream()

    litellm.completion = completion
    litellm.acompletion = acompletion
    litellm._docbenchtr_idle_retries_installed = True


def _run_pageindex_query(
    client: Any, query: str, pageindex_doc_id: str, k: int,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    answer_parts: list[str] = []
    streams: list[Any] = []
    from .ollama_native import TRANSPORT_EVENTS, _THINKING
    TRANSPORT_EVENTS.clear()
    _THINKING.clear()
    try:
        with _capture_pageindex_streams(streams):
            stream = client.chat(
                query,
                doc_id=pageindex_doc_id,
                stream=True,
                citations=True,
            )
            for event in stream.events:
                events.append(event)
                if event.get("type") == "answer":
                    answer_parts.append(event.get("delta", ""))
    except Exception as error:
        details = derive_pageindex_retrieval(events)
        details.update(_usage_from_pageindex_streams(streams))
        details["transport_events"] = list(TRANSPORT_EVENTS)
        raise PageIndexQueryError(
            f"{type(error).__name__}: {error}", details=details
        ) from error

    details = derive_pageindex_retrieval(events)
    details.update(_usage_from_pageindex_streams(streams))
    details["transport_events"] = list(TRANSPORT_EVENTS)
    citations = []
    try:
        citations = client.get_citations("".join(answer_parts), doc_id=pageindex_doc_id)
    except Exception as error:
        details["citation_error"] = f"{type(error).__name__}: {error}"
    details["cited_pages"] = list(dict.fromkeys(
        int(citation["page"]) for citation in citations
        if isinstance(citation.get("page"), int) and citation["page"] > 0
    ))
    details["ranked_pages"] = details["ranked_pages"][:k]
    return details


class PageIndexQueryError(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


@contextmanager
def _capture_pageindex_toc_source(
    captured: dict[str, Any], persisted_metadata: dict[str, Any] | None = None,
):
    """Capture toc metadata and keep PageIndex's PDFium parser sequential.

    The SDK's default parser starts one spawned process per CPU core. Nested
    under our isolated local-SDK worker this fails on Windows with invalid
    semaphore handles. Its documented ``workers=1`` path is structurally
    identical and avoids that nested process pool.
    """
    from pageindex.flash import api as flash_api

    original = flash_api.extract_toc

    def capture(*args, **kwargs):
        kwargs["workers"] = 1
        result = original(*args, **kwargs)
        captured["toc_source"] = result.get("toc_source")
        if persisted_metadata is not None and captured["toc_source"]:
            persisted_metadata["toc_source"] = captured["toc_source"]
        return result

    flash_api.extract_toc = capture
    try:
        yield
    finally:
        flash_api.extract_toc = original


@contextmanager
def _capture_pageindex_streams(captured: list[Any]):
    """Keep the SDK's streaming tool trace and its raw response usage together."""
    from agents import Runner

    original_descriptor = Runner.__dict__["run_streamed"]
    original = Runner.run_streamed

    def capture(*args, **kwargs):
        stream = original(*args, **kwargs)
        captured.append(stream)
        return stream

    Runner.run_streamed = capture
    try:
        yield
    finally:
        Runner.run_streamed = original_descriptor


def _usage_from_pageindex_streams(streams: list[Any]) -> dict[str, int | None]:
    prompt_tokens = 0
    completion_tokens = 0
    found_usage = False
    try:
        from pageindex.local_chat import _openai_usage
    except Exception:
        _openai_usage = None

    for stream in streams:
        raw_responses = getattr(stream, "raw_responses", None) or []
        if not raw_responses:
            continue
        if _openai_usage is not None:
            usage = _openai_usage(raw_responses)
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
        else:
            prompt = completion = None
            for response in raw_responses:
                usage = getattr(response, "usage", None)
                prompt_value = getattr(usage, "input_tokens", None)
                completion_value = getattr(usage, "output_tokens", None)
                if isinstance(prompt_value, int):
                    prompt = (prompt or 0) + prompt_value
                if isinstance(completion_value, int):
                    completion = (completion or 0) + completion_value
        if isinstance(prompt, int) and isinstance(completion, int):
            prompt_tokens += prompt
            completion_tokens += completion
            found_usage = True

    return {
        "input_tokens": prompt_tokens if found_usage else None,
        "output_tokens": completion_tokens if found_usage else None,
        "total_tokens": prompt_tokens + completion_tokens if found_usage else None,
    }


_PAGE_SPEC_PART_RE = re.compile(r"^(\d+)(?:-(\d+))?$")


def _expand_page_spec(page_spec: Any) -> list[int]:
    if isinstance(page_spec, list):
        if all(isinstance(page, int) and not isinstance(page, bool) and page > 0 for page in page_spec):
            return list(dict.fromkeys(page_spec))
        return []
    if not isinstance(page_spec, str) or not page_spec.strip():
        return []
    pages: list[int] = []
    for part in re.split(r"\s*,\s*", page_spec.strip()):
        match = _PAGE_SPEC_PART_RE.fullmatch(part)
        if not match:
            return []
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start:
            return []
        pages.extend(range(start, end + 1))
    return pages


def _tool_result_details(output: Any) -> tuple[list[int] | None, str | None]:
    """Read PageIndex's JSON tool envelope and return actual pages/error."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return None, None
    if isinstance(output, list):
        for item in output:
            pages, error = _tool_result_details(item)
            if pages is not None or error is not None:
                return pages, error
        return None, None
    if not isinstance(output, dict):
        return None, None
    if output.get("success") is False or output.get("isError") is True:
        return None, str(output.get("error") or output.get("message") or output)
    if output.get("error"):
        return None, str(output["error"])
    if "returned_pages" in output:
        raw_pages = output.get("returned_pages")
        pages = _expand_page_spec(raw_pages)
        if raw_pages not in (None, "", []) and not pages:
            return None, f"Invalid returned_pages value: {raw_pages!r}"
        return pages, None
    for key in ("text", "content", "structuredContent", "data"):
        if key in output:
            pages, error = _tool_result_details(output[key])
            if pages is not None or error is not None:
                return pages, error
    return None, None


def flatten_pageindex_tree(tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and flatten nodes needed for lookup and range expansion."""
    if not isinstance(tree, list) or not tree:
        raise ValueError("PageIndex tree must be a non-empty node list")
    flattened: list[dict[str, Any]] = []
    seen: set[str] = set()
    stack = list(reversed(tree))
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            raise ValueError("PageIndex tree contains a non-object node")
        node_id = node.get("node_id")
        start, end = node.get("start_index"), node.get("end_index")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("PageIndex tree node is missing a string node_id")
        if node_id in seen:
            raise ValueError(f"PageIndex tree contains duplicate node_id {node_id!r}")
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or start < 1 or end < start):
            raise ValueError(f"PageIndex tree node {node_id!r} has an invalid page range")
        children = node.get("nodes") or []
        if not isinstance(children, list):
            raise ValueError(f"PageIndex tree node {node_id!r} has invalid children")
        seen.add(node_id)
        flattened.append({
            "node_id": node_id,
            "start_index": start,
            "end_index": end,
        })
        stack.extend(reversed(children))
    return flattened


def project_pageindex_tree(tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the documented tree fields and hierarchy, omitting stored extras."""
    flatten_pageindex_tree(tree)

    def project(node: dict[str, Any]) -> dict[str, Any]:
        result = {
            "title": node.get("title") if isinstance(node.get("title"), str) else "",
            "node_id": node["node_id"],
            "start_index": node["start_index"],
            "end_index": node["end_index"],
            "summary": node.get("summary") if isinstance(node.get("summary"), str) else "",
        }
        children = node.get("nodes") or []
        if children:
            result["nodes"] = [project(child) for child in children]
        return result

    return [project(node) for node in tree]


def parse_pageindex_tree_search_response(
    response_text: str,
    known_node_ids: set[str],
) -> tuple[str, list[str]]:
    """Parse the structured LLM response and reject malformed or unknown IDs."""
    try:
        response = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Invalid JSON response: {error}") from error
    if not isinstance(response, dict):
        raise ValueError("Response must be a JSON object")
    thinking = response.get("thinking")
    node_list = response.get("node_list")
    if not isinstance(thinking, str):
        raise ValueError("Response field 'thinking' must be a string")
    if not isinstance(node_list, list) or any(not isinstance(node_id, str) for node_id in node_list):
        raise ValueError("Response field 'node_list' must be an array of string node IDs")
    unknown = list(dict.fromkeys(node_id for node_id in node_list if node_id not in known_node_ids))
    if unknown:
        raise ValueError(f"Response contains unknown node IDs: {unknown}")
    return thinking, node_list


def derive_pageindex_tree_search_ranking(
    tree: list[dict[str, Any]],
    node_ids: list[str],
) -> list[int]:
    """Expand selected inclusive page ranges in model order, keeping first pages."""
    by_id = {node["node_id"]: node for node in flatten_pageindex_tree(tree)}
    ranking: list[int] = []
    seen: set[int] = set()
    for node_id in node_ids:
        node = by_id.get(node_id)
        if node is None:
            raise ValueError(f"Selected node ID is not in the PageIndex tree: {node_id!r}")
        for page in range(node["start_index"], node["end_index"] + 1):
            if page not in seen:
                ranking.append(page)
                seen.add(page)
    return ranking


def _ollama_tree_search(
    model: str,
    base_url: str,
    query: str,
    tree: list[dict[str, Any]],
    node_ids: set[str],
) -> tuple[str, dict[str, int | None]]:
    schema = {
        "type": "object",
        "properties": {
            "thinking": {"type": "string"},
            "node_list": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(node_ids)},
            },
        },
        "required": ["thinking", "node_list"],
        "additionalProperties": False,
    }
    tree_json = json.dumps(tree, ensure_ascii=False, separators=(",", ":"))
    prompt = (
        f"Question: {query}\n\n"
        "Document tree structure (node IDs and inclusive one-based physical page ranges):\n"
        f"{tree_json}\n\n"
        "Find all nodes likely to contain the answer. List node IDs in relevance order, "
        "most relevant first. Use only IDs in the tree. The thinking field should give "
        "a brief high-level selection rationale. Return the requested JSON object."
    )
    from .ollama_native import chat, usage
    payload = chat({
        "model": model, "format": schema,
        "messages": [{"role": "user", "content": prompt}],
    }, base_url)
    content = payload["message"].get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama /api/chat returned no message content")
    return content, usage(payload)


def derive_pageindex_retrieval(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive a page ranking from successful PageIndex page-content reads.

    PageIndex's agent does not return scored page rankings. Each successful
    get_page_content result exposes returned_pages; calls are ordered by the
    agent's tool-call sequence, pages within a call by that returned list,
    and duplicates retain their first position.
    """
    calls: dict[str, dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    call_order = 0
    for event in events:
        if event.get("type") == "tool_call" and event.get("name") == "get_page_content":
            call_order += 1
            arguments = event.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            call = {
                "order": call_order,
                "requested_pages": _expand_page_spec(arguments.get("pages")),
                "returned_pages": None,
                "error": None,
            }
            call_id = event.get("call_id")
            if call_id is None:
                unkeyed.append(call)
            else:
                calls[str(call_id)] = call
        elif event.get("type") == "tool_result" and event.get("name") == "get_page_content":
            call_id = event.get("call_id")
            call = calls.get(str(call_id)) if call_id is not None else None
            if call is None:
                continue
            returned, error = _tool_result_details(event.get("output"))
            call["returned_pages"] = returned
            call["error"] = error
            if error:
                errors.append({"tool": "get_page_content", "error": error})

    ordered_calls = sorted([*calls.values(), *unkeyed], key=lambda call: call["order"])
    ranking: list[int] = []
    seen: set[int] = set()
    trace = []
    for call in ordered_calls:
        returned = call["returned_pages"]
        if call["error"] is None and returned is None:
            call["error"] = "No tool result was recorded."
            errors.append({"tool": "get_page_content", "error": call["error"]})
        if call["error"] is None and returned is not None:
            for page in returned:
                if page not in seen:
                    ranking.append(page)
                    seen.add(page)
        trace.append({
            "tool": "get_page_content",
            "requested_pages": call["requested_pages"],
            "returned_pages": returned or [],
            "error": call["error"],
        })
    tool_events = [event for event in events if event.get("type") == "tool_call"]
    return {
        "ranked_pages": ranking,
        "retrieval_trace": trace,
        "tool_calls": [
            {
                "name": event.get("name"),
                "arguments": event.get("arguments") or {},
            }
            for event in tool_events
        ],
        "tool_errors": errors,
        "tool_call_count": len(tool_events),
        "page_content_call_count": call_order,
    }
