"""Keep unit tests CPU-only; no test constructs a dense embedding model."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


torch_stub = ModuleType("torch")
torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch_stub

sentence_transformers_stub = ModuleType("sentence_transformers")


class _NoModelForUnitTests:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("Dense model construction is outside the unit test suite")


sentence_transformers_stub.SentenceTransformer = _NoModelForUnitTests
sys.modules["sentence_transformers"] = sentence_transformers_stub
