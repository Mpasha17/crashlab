"""``client.embeddings.create`` through a wrapped client must record and replay offline
(RAG agents: replay used to fail with "not made through a crashlab-wrapped client")."""

from __future__ import annotations

import asyncio

import pytest

import crashlab

from ._transport import httpx_module

openai = pytest.importorskip("openai")
httpx = httpx_module()

BODY = {
    "object": "list", "model": "emb-1",
    "data": [{"object": "embedding", "index": i, "embedding": [0.25 * i, -1.5, 3.0]}
             for i in range(2)],
    "usage": {"prompt_tokens": 7, "total_tokens": 7},
}  # fmt: skip
LIVE = {"on": True}


def _handle(request):
    assert LIVE["on"], "replay hit the network"
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(200, json=BODY)
    return httpx.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "Paris"}}]})  # fmt: skip


def _client(is_async=False):
    kw = {"api_key": "sk-x", "base_url": "http://t/v1"}
    if is_async:
        http = httpx.AsyncClient(transport=httpx.MockTransport(_handle))
        return crashlab.wrap(openai.AsyncOpenAI(http_client=http, **kw))
    http = httpx.Client(transport=httpx.MockTransport(_handle))
    return crashlab.wrap(openai.OpenAI(http_client=http, **kw))


@crashlab.agent(name="rag")
def rag(q: str) -> list:
    client = _client()
    emb = client.embeddings.create(model="emb-1", input=[q, "doc"], encoding_format="float")
    assert isinstance(emb, openai.types.CreateEmbeddingResponse)
    ans = client.chat.completions.create(model="m", messages=[{"role": "user", "content": q}])
    return [emb.data[1].embedding, ans.choices[0].message.content]


@crashlab.agent(name="rag-async")
async def rag_async(q: str) -> list:
    emb = await _client(True).embeddings.create(model="emb-1", input=q)
    return [d.embedding for d in emb.data]


def _replay(run):
    LIVE["on"] = False
    try:
        return crashlab.replay(run)
    finally:
        LIVE["on"] = True


def test_embeddings_record_and_replay_offline(crashlab_dir):
    out = rag("where is Paris?")
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["embeddings", "chat"]
    step = run.steps[0]
    assert step.name == "emb-1" and step.request["input"] == ["where is Paris?", "doc"]
    assert step.response["embeddings"] == [{"index": 0, "dimensions": 3},
                                           {"index": 1, "dimensions": 3}]  # fmt: skip
    assert run.usage.input_tokens >= 7
    replayed = _replay(run)
    assert replayed.status == "ok" and replayed.output == out


def test_async_embeddings_record_and_replay(crashlab_dir):
    out = asyncio.run(rag_async("q"))
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["embeddings"]
    assert _replay(run).output == out


def test_changed_embedding_input_diverges(crashlab_dir):
    rag("where is Paris?")
    run = crashlab.load("last")
    with pytest.raises(crashlab.DivergenceError):
        LIVE["on"] = False
        try:
            crashlab.replay(run, overrides={"q": "where is Rome?"})
        finally:
            LIVE["on"] = True
