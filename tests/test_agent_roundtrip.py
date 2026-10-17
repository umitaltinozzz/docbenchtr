"""Exercise the pinned SDK stream consumer with native HTTP replaced by a stub.

Run in a child interpreter so heavyweight package imports cannot bypass the
unit suite's guard against constructing embedding models. No model is loaded.
"""
import os
from pathlib import Path
import subprocess
import sys


def test_native_bridge_agent_tool_roundtrip():
    script = r'''
import asyncio
from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from docbenchtr import ollama_native as native
set_tracing_disabled(True)
seen = []
def fake_chat(body, base_url=None):
    seen.append(body)
    if len(seen) == 1:
        return {"message": {"role": "assistant", "content": "", "thinking": "Need pages",
                "tool_calls": [{"function": {"name": "get_page_content", "arguments": {"pages": "3"}}}]},
                "prompt_eval_count": 10, "eval_count": 20}
    assert body["messages"][1]["role"] == "user", body
    assert body["messages"][-1]["role"] == "tool", body
    assert body["messages"][-2]["thinking"] == "Need pages", body
    return {"message": {"role": "assistant", "content": "Answer"},
            "prompt_eval_count": 30, "eval_count": 40}
native.chat = fake_chat
native.install()
@function_tool
def get_page_content(pages: str) -> str:
    assert pages == "3"
    return '{"returned_pages":"3","success":true}'
async def run():
    agent = Agent(name="test", instructions="Find pages",
                  model=LitellmModel(model="ollama_chat/test", base_url="http://127.0.0.1:11434"),
                  tools=[get_page_content])
    stream = Runner.run_streamed(agent, input="Original question")
    async for _ in stream.stream_events():
        pass
    assert stream.final_output == "Answer"
    assert len(seen) == 2
    assert stream.context_wrapper.usage.input_tokens == 40
    assert stream.context_wrapper.usage.output_tokens == 60
asyncio.run(run())

# Also exercise the actual pinned PageIndex tools/default instructions and trace
# consumer, not only a generic SDK Agent. The tiny synthetic cache needs no PDF
# indexing, embeddings, Ollama server or model weights.
import runpy
import tempfile
runpy.run_path("tests/conftest.py")  # prevent embedding construction in this child
from docbenchtr.retrievers import _new_pageindex_client, _run_pageindex_query
from pageindex.agent_tools import AGENT_INSTRUCTIONS
seen.clear()
def pageindex_chat(body, base_url=None):
    seen.append(body)
    if len(seen) == 1:
        assert AGENT_INSTRUCTIONS in body["messages"][0]["content"]
        return {"message": {"content": "", "thinking": "Read page three",
                "tool_calls": [{"function": {"name": "get_page_content",
                  "arguments": {"doc_name": "fixture.pdf", "pages": "3"}}}]},
                "prompt_eval_count": 10, "eval_count": 20}
    assert len(seen) == 2, body
    assert any(m["role"] == "user" and "Original question" in m["content"] for m in body["messages"])
    assert body["messages"][-2]["thinking"] == "Read page three"
    assert body["messages"][-1]["tool_name"] == "get_page_content"
    assert "fixture content 3" in body["messages"][-1]["content"]
    return {"message": {"content": "Answer [fixture.pdf, p. 3]"},
            "prompt_eval_count": 30, "eval_count": 40}
native.chat = pageindex_chat
with tempfile.TemporaryDirectory() as cache:
    client = _new_pageindex_client(cache, "default")
    client._api._store.save_document("fixture", {
        "id": "fixture", "name": "fixture.pdf", "status": "completed",
        "pageNum": 3, "createdAt": "2026-10-18T00:00:00Z", "mode": "local",
    }, [{"title": "Fixture", "node_id": "0001", "start_index": 1, "end_index": 3}],
       [{"page_index": n, "markdown": f"fixture content {n}"} for n in [1, 2, 3]])
    details = _run_pageindex_query(client, "Original question", "fixture", 3)
    assert details["ranked_pages"] == [3], details
    assert details["tool_call_count"] == 1, details
    assert details["input_tokens"] == 40 and details["output_tokens"] == 60, details
    assert not details["tool_errors"], details

# PageIndex uses both sync and async completion paths during indexing.
from pageindex.utils import llm_completion, llm_acompletion
def index_chat(body, base_url=None):
    assert body["model"] == "index:test"
    assert body["messages"] == [{"role": "user", "content": "Summarize fixture"}]
    return {"message": {"content": "Fixture summary"}}
native.chat = index_chat
assert llm_completion("ollama_chat/index:test", "Summarize fixture") == "Fixture summary"
assert asyncio.run(llm_acompletion("ollama_chat/index:test", "Summarize fixture")) == "Fixture summary"
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
