"""Native Ollama wire transport, with a small LiteLLM response-type bridge.

PageIndex 0.4.0 still owns agent instructions, tools and execution. Only its
Ollama HTTP route is replaced. Non-streaming native calls avoid concatenating
complete tool arguments as streaming deltas (the recorded 'Extra data' bug).
https://docs.ollama.com/api/chat
https://docs.ollama.com/capabilities/tool-calling
"""
from __future__ import annotations

import asyncio
import copy
import json
import urllib.error
import urllib.request
import uuid
from typing import Any

from .config import OLLAMA_BASE_URL, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT

TRANSPORT_EVENTS: list[dict[str, Any]] = []
_THINKING: dict[str, str] = {}


class OllamaResponseError(RuntimeError):
    pass


def decode_response(raw: bytes) -> dict[str, Any]:
    """Errors are provider errors, never a misleading KeyError('message')."""
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise OllamaResponseError(f"Invalid non-streaming Ollama response: {raw[:500]!r}") from error
    if not isinstance(payload, dict):
        raise OllamaResponseError("Ollama response must be an object")
    if payload.get("error"):
        raise OllamaResponseError(str(payload["error"]))
    if not isinstance(payload.get("message"), dict):
        raise OllamaResponseError(f"Ollama returned no message: {payload!r}")
    if payload.get("done_reason") == "length":
        raise OllamaResponseError("Ollama exhausted num_predict before completing the response")
    return payload


def chat(body: dict[str, Any], base_url: str = OLLAMA_BASE_URL) -> dict[str, Any]:
    body = {**body, "stream": False,
            "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX,
                        "num_predict": OLLAMA_NUM_PREDICT}}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return decode_response(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        # Ollama commonly returns a JSON error body with HTTP 500.
        try:
            decode_response(raw)
        except OllamaResponseError as detail:
            raise OllamaResponseError(f"HTTP {error.code}: {detail}") from error
        raise OllamaResponseError(f"HTTP {error.code}: {raw[:500]!r}") from error


def usage(payload: dict[str, Any]) -> dict[str, int | None]:
    inp, out = payload.get("prompt_eval_count"), payload.get("eval_count")
    return {"input_tokens": inp, "output_tokens": out,
            "total_tokens": inp + out if isinstance(inp, int) and isinstance(out, int) else None}


def native_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    names = {}
    for original in messages:
        message = copy.deepcopy(original)
        role = message["role"]
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content)
        item = {"role": "system" if role == "developer" else role, "content": content}
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call["function"]
                args = function["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError as error:
                        # Do not guess which of multiple JSON objects the agent meant.
                        raise OllamaResponseError(f"Invalid arguments for {function['name']}: {args[:500]}") from error
                if not isinstance(args, dict):
                    raise OllamaResponseError("Tool arguments must be a JSON object")
                names[call.get("id")] = function["name"]
                calls.append({"function": {"name": function["name"], "arguments": args}})
                if call.get("id") in _THINKING:
                    item["thinking"] = _THINKING[call["id"]]
            item["tool_calls"] = calls
        if role == "tool":
            item["tool_name"] = names.get(message.get("tool_call_id"), message.get("name", ""))
        converted.append(item)
    return converted


def protect_query(messages: list[dict[str, Any]], tools: list, num_ctx: int,
                  num_predict: int) -> list[dict[str, Any]]:
    """Keep the original user question and bound history before Ollama truncates it.

    UTF-8 bytes provide a deliberately conservative token upper-bound proxy.
    Reserve the output cap and 2k template tokens; shorten tool-result text only,
    oldest first. If fixed instructions/question/tool schemas do not fit, fail
    explicitly. Every shortening is recorded; no hidden prompt loss.
    """
    result = copy.deepcopy(messages)
    if not any(m["role"] == "user" and m.get("content") for m in result):
        raise OllamaResponseError("Agent request contains no user question")
    budget = num_ctx - num_predict - 2048
    def size():
        return len(json.dumps({"messages": result, "tools": tools}, ensure_ascii=False).encode())
    for message in result:
        if size() <= budget:
            break
        if message["role"] != "tool":
            continue
        content = message["content"].encode("utf-8")
        keep = max(0, len(content) - (size() - budget) - 128)
        message["content"] = content[:keep].decode("utf-8", errors="ignore") + "\n[Tool result shortened to fit context]"
        TRANSPORT_EVENTS.append({"event": "tool_result_shortened", "original_bytes": len(content), "retained_bytes": keep})
    if size() > budget:
        raise OllamaResponseError("Fixed agent prompt exceeds context budget; user question was preserved")
    return result


def _response_parts(payload: dict[str, Any]) -> tuple[dict, str, dict]:
    message = payload["message"]
    converted = {"role": "assistant", "content": message.get("content", "")}
    calls = []
    for call in message.get("tool_calls") or []:
        function = call["function"]
        args = function["arguments"]
        if not isinstance(args, dict):
            raise OllamaResponseError("Native tool arguments must be an object")
        call_id = "call_" + uuid.uuid4().hex
        if message.get("thinking"):
            _THINKING[call_id] = message["thinking"]
        calls.append({"id": call_id, "type": "function", "function": {
            "name": function["name"], "arguments": json.dumps(args, ensure_ascii=False)}})
    if calls:
        converted["tool_calls"] = calls
    if message.get("thinking"):
        converted["reasoning_content"] = message["thinking"]
    counts = usage(payload)
    counts = {"prompt_tokens": counts["input_tokens"] or 0,
              "completion_tokens": counts["output_tokens"] or 0,
              "total_tokens": counts["total_tokens"] or 0}
    return converted, "tool_calls" if calls else "stop", counts


def _request(kwargs: dict) -> dict:
    messages = native_messages(kwargs["messages"])
    tool_defs = kwargs.get("tools") or []
    if tool_defs:
        messages = protect_query(messages, tool_defs, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT)
    body = {"model": kwargs["model"].removeprefix("ollama_chat/"), "messages": messages}
    if tool_defs:
        body["tools"] = tool_defs
    response_format = kwargs.get("response_format")
    if isinstance(response_format, dict):
        body["format"] = (response_format.get("json_schema") or {}).get("schema", "json")
    return chat(body, kwargs.get("api_base") or kwargs.get("base_url") or OLLAMA_BASE_URL)


def install() -> None:
    """Install once inside the isolated PageIndex worker, never for other routes."""
    import litellm
    from litellm.types.utils import ModelResponseStream
    if getattr(litellm, "_docbenchtr_native", False):
        return
    original_sync, original_async = litellm.completion, litellm.acompletion

    def completion(*args, **kwargs):
        if not kwargs.get("model", "").startswith("ollama_chat/"):
            return original_sync(*args, **kwargs)
        if args or kwargs.get("stream"):
            raise ValueError("Native synchronous indexing expects keyword arguments and stream=False")
        message, finish, counts = _response_parts(_request(kwargs))
        return litellm.ModelResponse(model=kwargs["model"], choices=[{
            "index": 0, "message": message, "finish_reason": finish}], usage=counts)

    async def acompletion(*args, **kwargs):
        if not kwargs.get("model", "").startswith("ollama_chat/"):
            return await original_async(*args, **kwargs)
        message, finish, counts = _response_parts(await asyncio.to_thread(_request, kwargs))
        if not kwargs.get("stream"):
            return litellm.ModelResponse(model=kwargs["model"], choices=[{
                "index": 0, "message": message, "finish_reason": finish}], usage=counts)
        async def chunks():
            # Emit each tool argument object exactly once; never concatenate objects.
            delta = copy.deepcopy(message)
            for i, call in enumerate(delta.get("tool_calls", [])):
                call["index"] = i
            yield ModelResponseStream(model=kwargs["model"], choices=[{
                "index": 0, "delta": delta, "finish_reason": None}])
            yield ModelResponseStream(model=kwargs["model"], choices=[{
                "index": 0, "delta": {}, "finish_reason": finish}], usage=counts)
        return chunks()

    litellm.completion, litellm.acompletion = completion, acompletion
    litellm._docbenchtr_native = True
