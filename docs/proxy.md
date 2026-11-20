# Localhost recording and replay proxy

`crashlab proxy` is a zero-dependency HTTP proxy that records live LLM traffic
from *any* language or framework into `.crashlab/` and can later serve that
cassette offline. Point the process at this proxy with `OPENAI_BASE_URL` or
`ANTHROPIC_BASE_URL`. Supported routes are OpenAI `/v1/chat/completions` and
`/v1/responses`, and Anthropic `/v1/messages`, including `stream: true` SSE.

The proxy binds to localhost only (`127.0.0.1`, `::1`, or `localhost`).
`Authorization` and `x-api-key` are forwarded to the upstream in record mode
and are never written to the store.

## Record

```bash
crashlab proxy --upstream https://api.openai.com --port 8787 --agent my-bot
```

Then in another shell:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
```

Each request is forwarded to `--upstream`, streamed back to the client in real
time, and stored as a step on a session run. Runtime checks run as the steps
arrive, and findings are saved on the run. Stop the proxy with Ctrl-C to
finalize the session.

Group several requests into one run by sending `X-Crashlab-Run: <name>`.
Requests without that header share one per-server session. `--once` handles a
single LLM request and exits (useful in scripts).

```bash
crashlab proxy --upstream https://api.openai.com --once
```

## Replay

```bash
crashlab proxy --replay last --port 8787
```

The proxy serves recorded responses from the cassette with no upstream network
calls. It matches the next unused chat step by request hash, or by model and
messages. A request that does not match returns HTTP 409 JSON
(`cassette_divergence`); a cassette with no remaining steps returns HTTP 404
(`cassette_exhausted`). JSON and SSE clients are both supported: a streamed
recording can be replayed as JSON, and a JSON recording can be replayed as SSE.

## In Python

```python
from crashlab.proxy import ProxyConfig, create_proxy_server
from crashlab.store import Store

config = ProxyConfig(
    host="127.0.0.1",
    port=0,
    upstream_url="http://127.0.0.1:9",
    agent="proxy",
)
server = create_proxy_server(config)
host, port = server.server_address[:2]
# Point OPENAI_BASE_URL at http://127.0.0.1:{port}/v1, then:
# server.serve_forever()
server.server_close()

# Offline replay of a recorded run (no upstream_url):
store = Store()
replay = ProxyConfig(host="127.0.0.1", port=0, replay_run_id="last", store_dir=str(store.root))
```

`create_proxy_server` returns a `ThreadingHTTPServer`. Use `port=0` to pick an
ephemeral port and read it back from `server.server_address`. Call
`server.shutdown()` then `server.server_close()` for a clean stop; open session
runs are finalized on close.

`ProxyConfig.allow_remote` must stay false unless you intentionally bind a
non-localhost address. `fault` can inject a delay or an HTTP error (an int
status, a name like `"empty"`, or a dict with `status` / `delay_ms`).
