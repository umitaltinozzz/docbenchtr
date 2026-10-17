import ast
import json
from pathlib import Path

import pytest

from docbenchtr.ollama_native import (
    OllamaResponseError, TRANSPORT_EVENTS, _response_parts, decode_response,
    native_messages, protect_query,
)

ROOT = Path(__file__).resolve().parents[1]


def recorded_errors(fragment):
    return [row for path in (ROOT / "results/colab_a100_models").glob("*/per_query.json")
            for row in json.loads(path.read_text(encoding="utf-8")) if fragment in (row.get("error") or "")]


def test_recorded_missing_message_payloads_keep_provider_error():
    rows = recorded_errors("KeyError: 'message'")
    assert len(rows) == 3
    for row in rows:
        payload = ast.literal_eval(row["error"].split("Got unexpected response from Ollama: ", 1)[1])
        with pytest.raises(OllamaResponseError, match="error parsing tool call") as error:
            decode_response(json.dumps(payload).encode())
        assert "KeyError" not in str(error.value)


def test_recorded_extra_data_uses_whole_native_tool_objects():
    rows = recorded_errors("Extra data")
    assert len(rows) == 13
    assert all('json.loads(typed_tool["function"]["arguments"])' in r["error"] for r in rows)
    # The saved errors contain stack traces, not the offending response bytes.
    # Reconstruct tool argument shapes from their recorded tool-call traces.
    seen = 0
    for row in rows:
        for call in row["tool_calls"]:
            args = call.get("arguments")
            if not isinstance(args, dict):
                continue
            message, finish, _ = _response_parts({"message": {
                "content": "", "thinking": "Choose pages", "tool_calls": [
                    {"function": {"name": call["name"], "arguments": args}}]}})
            assert finish == "tool_calls"
            native = native_messages([message])[0]
            assert native["tool_calls"][0]["function"]["arguments"] == args
            assert native["thinking"] == "Choose pages"
            seen += 1
    assert seen > 0


def test_recorded_no_user_query_is_guarded_before_sending():
    rows = recorded_errors("no user query found")
    assert len(rows) == 21
    for row in rows:
        raw = row["error"].split("Ollama_chatException - ", 1)[1].encode()
        with pytest.raises(OllamaResponseError, match="no user query found"):
            decode_response(raw)
    TRANSPORT_EVENTS.clear()
    messages = [{"role": "system", "content": "instructions"},
                {"role": "user", "content": "original question"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": {}}}]},
                {"role": "tool", "content": "large document " * 20000}]
    guarded = protect_query(messages, [], 65536, 32768)
    assert guarded[1] == messages[1]
    assert len(json.dumps({"messages": guarded, "tools": []}, ensure_ascii=False).encode()) <= 65536 - 32768 - 2048
    assert "shortened" not in messages[-1]["content"]
    assert "shortened" in guarded[-1]["content"]
    assert TRANSPORT_EVENTS
    with pytest.raises(OllamaResponseError, match="no user question"):
        protect_query(messages[2:], [], 65536, 32768)


def test_guard_rejects_oversized_fixed_prompt_without_dropping_question():
    with pytest.raises(OllamaResponseError, match="preserved"):
        protect_query([{"role": "user", "content": "x" * 100000}], [], 65536, 32768)


def test_rejects_incomplete_response():
    with pytest.raises(OllamaResponseError, match="num_predict"):
        decode_response(b'{"message":{"content":"partial"},"done_reason":"length"}')


def test_native_http_options_and_non_streaming(monkeypatch):
    from docbenchtr import ollama_native as native
    captured = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"message":{"content":"ok"},"done":true}'
    def urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()
    monkeypatch.setattr(native.urllib.request, "urlopen", urlopen)
    native.chat({"model": "test", "messages": [], "stream": True, "options": {"temperature": 1}})
    assert captured["stream"] is False
    assert captured["options"] == {"temperature": 0, "num_ctx": native.OLLAMA_NUM_CTX, "num_predict": native.OLLAMA_NUM_PREDICT}


def test_native_socket_timeouts_use_existing_idle_retry_policy():
    from docbenchtr.retrievers import _is_http_timeout
    import urllib.error
    class OtherTimeout(Exception):
        pass
    error = urllib.error.URLError("provider unavailable")
    error.__cause__ = TimeoutError("native read timed out")
    assert _is_http_timeout(error, OtherTimeout)
    assert not _is_http_timeout(ValueError("invalid JSON"), OtherTimeout)
