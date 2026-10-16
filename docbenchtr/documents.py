from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from .config import DOCUMENTS, PROCESSED_DIR, RAW_DIR, SOURCES_PATH


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_documents(retries: int = 3, timeout: int = 90) -> list[dict[str, Any]]:
    """Download source PDFs and write an integrity manifest; never commits PDFs."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in DOCUMENTS:
        path = RAW_DIR / source["filename"]
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    source["source_url"],
                    headers={"User-Agent": "DocBenchTR/0.1 research pilot"},
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                    content_type = response.headers.get("Content-Type", "")
                if not payload.startswith(b"%PDF-") or "pdf" not in content_type.lower():
                    raise ValueError(
                        f"Expected a PDF from {source['source_url']}; received {content_type!r}"
                    )
                path.write_bytes(payload)
                last_error = None
                break
            except Exception as error:  # recorded in the raised error after retries
                last_error = error
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
        if last_error:
            raise RuntimeError(f"Could not download {source['document_id']}: {last_error}")

        digest = sha256_file(path)
        if digest != source.get("expected_sha256"):
            raise ValueError(
                f"SHA-256 mismatch for {source['document_id']}: expected "
                f"{source.get('expected_sha256')}, received {digest}; "
                "check the source snapshot before updating its pin."
            )
        record = {
            **source,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "downloaded_from": source["source_url"],
        }
        records.append(record)

    SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


def extract_pdf_pages(records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Extract searchable text per physical PDF page (one-based page numbers)."""
    if records is None:
        records = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    executable = shutil.which("pdftotext")
    if not executable:
        raise RuntimeError("Poppler's pdftotext executable is required and was not found on PATH")
    version = subprocess.run([executable, "-v"], capture_output=True, text=True, check=False)
    extractor_version = (version.stderr or version.stdout).splitlines()[0].strip()
    pages: list[dict[str, Any]] = []
    for record in records:
        path = RAW_DIR / record["filename"]
        extracted = subprocess.run(
            [executable, "-enc", "UTF-8", "-layout", str(path), "-"],
            capture_output=True,
            check=False,
        )
        if extracted.returncode != 0:
            diagnostic = extracted.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"pdftotext failed for {record['filename']}: {diagnostic}")
        full_text = extracted.stdout.decode("utf-8", errors="strict")
        text_pages = full_text.split("\f")
        while text_pages and not text_pages[-1].strip():
            text_pages.pop()
        text_pages = [page.strip() for page in text_pages]
        if not text_pages:
            raise ValueError(f"PDF has no pages: {record['filename']}")
        if any("\ufffd" in text for text in text_pages):
            raise ValueError(f"Unicode replacement character found while extracting {record['filename']}")
        page_count = len(text_pages)
        if page_count == 0:
            raise ValueError(f"PDF has no pages: {record['filename']}")
        nonempty = sum(bool(text.strip()) for text in text_pages)
        if nonempty < max(1, int(page_count * 0.8)):
            raise ValueError(
                f"{record['filename']} appears scanned or has sparse text: "
                f"{nonempty} text pages out of {page_count}"
            )
        for page_number, text in enumerate(text_pages, start=1):
            pages.append(
                {
                    "document_id": record["document_id"],
                    "page_number": page_number,
                    "text": text,
                }
            )
        record["pdf_page_count"] = page_count
        record["text_page_count"] = nonempty
        record["text_extractor"] = extractor_version
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (PROCESSED_DIR / "pages.jsonl").open("w", encoding="utf-8") as output:
        for page in pages:
            output.write(json.dumps(page, ensure_ascii=False) + "\n")
    SOURCES_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pages
