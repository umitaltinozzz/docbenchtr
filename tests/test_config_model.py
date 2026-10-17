import importlib

from docbenchtr import config


def test_ollama_model_defaults_to_published_pilot_model(monkeypatch):
    monkeypatch.delenv("DOCBENCHTR_OLLAMA_MODEL", raising=False)
    assert importlib.reload(config).OLLAMA_MODEL == "qwen2.5:7b-instruct"


def test_ollama_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("DOCBENCHTR_OLLAMA_MODEL", "qwen3.8:27b")
    try:
        assert importlib.reload(config).OLLAMA_MODEL == "qwen3.8:27b"
    finally:
        monkeypatch.delenv("DOCBENCHTR_OLLAMA_MODEL")
        importlib.reload(config)
