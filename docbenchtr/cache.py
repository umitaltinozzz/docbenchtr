"""Copy the historical 7B cache into its own namespace without editing it."""
from __future__ import annotations

import json
import shutil

from .config import (
    ROOT, PAGEINDEX_INDEX_MODEL, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, pageindex_cache_dir,
)


def prepare_cache():
    target = pageindex_cache_dir()
    legacy = ROOT / ".pageindex" / "docbenchtr-pilot"
    if (PAGEINDEX_INDEX_MODEL == "qwen2.5:7b-instruct"
            and (OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT) == (32768, 4096)
            and not target.exists() and (legacy / "manifest.json").exists()):
        shutil.copytree(legacy, target)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        for doc in manifest["docs"].values():
            doc.setdefault("metadata", {}).update(
                index_model=PAGEINDEX_INDEX_MODEL, num_ctx=32768, num_predict=4096,
                temperature=0, provenance="copied from published 7B pilot; see README correction history",
            )
            (target / "docs" / doc["id"] / "doc.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return target
