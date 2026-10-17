from __future__ import annotations

import os
import hashlib
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
QUESTIONS_PATH = ROOT / "data" / "pilot_questions.json"
SOURCES_PATH = ROOT / "data" / "sources.json"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
# The published pilot uses qwen2.5:7b-instruct; other runs override it per process.
OLLAMA_MODEL = os.environ.get("DOCBENCHTR_OLLAMA_MODEL", "qwen2.5:7b-instruct")
PAGEINDEX_INDEX_MODEL = os.environ.get("DOCBENCHTR_INDEX_MODEL", "qwen2.5:7b-instruct")
MODEL_DIGESTS = json.loads(os.environ.get("DOCBENCHTR_MODEL_DIGESTS", "{}"))
OLLAMA_BASE_URL = os.environ.get("DOCBENCHTR_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# Context window requested per call (the published pilot uses 32,768).
OLLAMA_NUM_CTX = int(os.environ.get("DOCBENCHTR_NUM_CTX", "32768"))
# Shared per-response cap for indexing, agentic search, tree search and reranking.
OLLAMA_NUM_PREDICT = int(os.environ.get("DOCBENCHTR_NUM_PREDICT", "4096"))


def model_slug(model: str) -> str:
    """Portable, collision-resistant folder name (including Windows/Drive)."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", model)[:80] + "-" + hashlib.sha256(model.encode()).hexdigest()[:10]


def pageindex_cache_dir(index_model: str = PAGEINDEX_INDEX_MODEL) -> Path:
    root = Path(os.environ.get("DOCBENCHTR_CACHE_ROOT", str(ROOT / ".pageindex" / "models")))
    return root / model_slug(index_model) / f"ctx{OLLAMA_NUM_CTX}_out{OLLAMA_NUM_PREDICT}"

DOCUMENTS = [
    {
        "document_id": "kvkk_6698",
        "title": "Kişisel Verilerin Korunması Kanunu",
        "filename": "6698.pdf",
        "source_url": "https://upload.wikimedia.org/wikipedia/commons/7/79/Ki%C5%9Fisel_Verilerin_Korunmas%C4%B1_Kanunu.pdf",
        "expected_sha256": "cf96728294134fa067cdff0f214aa63fbc9966901ee5700aad612c87c32adea6",
        "canonical_source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6698.pdf",
        "source_page_url": "https://tr.wikisource.org/wiki/Dosya:Kişisel_Verilerin_Korunması_Kanunu.pdf",
        "license_note": "Officially promulgated law; free use under Turkish Law No. 5846, Article 31. The archived PDF identifies Mevzuat Bilgi Sistemi as its source and reflects the text available in 2020.",
    },
    {
        "document_id": "tuketicinin_korunmasi_6502",
        "title": "Tüketicinin Korunması Hakkında Kanun ve İkincil Mevzuatı (Ocak 2021)",
        "filename": "6502.pdf",
        "source_url": "https://tuketici.ticaret.gov.tr/data/5e81982d13b876a1b04c7a42/Tuketicinin_Korunmasi_Hakkinda_Kanun_6502_Ocak_2021.pdf",
        "expected_sha256": "2a833defee359f117e4f994233627a65f3e51b1dbde63977e2ede105c64f45aa",
        "canonical_source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6502.pdf",
        "source_page_url": "https://ticaret.gov.tr/tuketici/mevzuat/6502-sayili-tuketicinin-korunmasi-mevzuati",
        "license_note": "Officially published Turkish legislation; free use under Turkish Law No. 5846, Article 31. Agency-hosted January 2021 PDF.",
    },
    {
        "document_id": "yuksekogretim_2547",
        "title": "Yükseköğretim Kanunu",
        "filename": "2547.pdf",
        "source_url": "https://www.yokak.gov.tr/documents/mevzuatlar/1.5.2547.pdf",
        "expected_sha256": "47de7428388fa0146bf6c5b056b52380379af59b8f62b4f5940b9053ca013c31",
        "canonical_source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.2547.pdf",
        "source_page_url": "https://www.yokak.gov.tr/documents/mevzuatlar/1.5.2547.pdf",
        "license_note": "Officially published Turkish legislation; free use under Turkish Law No. 5846, Article 31. Agency-hosted PDF.",
    },
]
