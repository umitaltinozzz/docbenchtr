from __future__ import annotations

import re
import shutil
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
import snowballstemmer
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION, OLLAMA_MODEL


TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+", re.IGNORECASE)
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
        request = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            import json

            tags = json.loads(response.read().decode("utf-8"))
        available = {entry.get("name") for entry in tags.get("models", [])}
        if model in available or any(name and name.startswith(model.split(":")[0] + ":") for name in available):
            return True, f"Ollama is serving {model}"
        return False, f"Ollama is reachable but model {model!r} is not installed; found {sorted(available)}"
    except Exception as error:
        cli_state = "Ollama executable was not found on PATH; " if not executable else ""
        return False, f"{cli_state}Ollama API http://127.0.0.1:11434/api/tags is unreachable: {type(error).__name__}: {error}"


def _new_pageindex_client(storage_path: str):
    from pageindex import PageIndexClient

    ollama_model = OLLAMA_MODEL
    backend = {"api_base": "http://127.0.0.1:11434"}
    return PageIndexClient(
        index={
            "model": f"ollama_chat/{ollama_model}",
            "backend": backend,
            "storage_path": storage_path,
            "optimize": "merge",
            "summary_concurrency": 1,
        },
        chat={
            "model": f"ollama_chat/{ollama_model}",
            "backend": backend,
        },
    )


def pageindex_local_preflight(storage_path: str) -> str:
    """Initialize the SDK's local config without submitting documents or making model calls."""
    client = _new_pageindex_client(storage_path)
    return f"Local SDK config initialized with index model {client.index_model!r} and chat model {client.chat_model!r}."


class PageIndexRetriever(Retriever):
    """PageIndex local SDK adapter; page citations are parsed from local chat output."""

    def __init__(self, pdf_paths: dict[str, str], storage_path: str):
        self.name = "PageIndex (local Ollama)"
        self.client = _new_pageindex_client(storage_path)
        self.doc_ids: dict[str, str] = {}
        for doc_id, pdf_path in pdf_paths.items():
            submitted = self.client.submit_document(pdf_path, wait=True)
            self.doc_ids[doc_id] = submitted["doc_id"]

    def retrieve(self, query: str, document_id: str, k: int = 5) -> list[int]:
        # PageIndex chat returns citations with physical page numbers. We do
        # not infer retrieval pages from answer prose if no citation is present.
        doc_id = self.doc_ids[document_id]
        answer = self.client.chat(query, doc_id=doc_id, citations=True)
        citations = self.client.get_citations(answer, doc_id=doc_id)
        return list(dict.fromkeys(int(citation["page"]) for citation in citations if citation.get("page")))[:k]
