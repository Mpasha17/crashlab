"""``crashlab proxy``: localhost HTTP recording and offline-replay proxy."""

from __future__ import annotations

import contextlib

import typer

from .cli import _console, _e, app
from .store import RunNotFoundError, Store


@app.command("proxy")
def proxy_cmd(
    upstream: str | None = typer.Option(
        None,
        "--upstream",
        help="Upstream origin (required unless --replay), e.g. https://api.openai.com.",
    ),
    port: int = typer.Option(8787, "--port", help="Local port (0 picks an ephemeral port)."),
    agent: str = typer.Option("proxy", "--agent", help="Agent name stored on recorded runs."),
    replay: str | None = typer.Option(
        None, "--replay", help="Run id, unique prefix, or 'last' to serve offline."
    ),
    once: bool = typer.Option(False, "--once", help="Handle one LLM request and exit."),
) -> None:
    """Record or replay LLM HTTP traffic on localhost.

    Point OPENAI_BASE_URL or ANTHROPIC_BASE_URL at this process. Requests are recorded
    into the local store, including SSE streams. --replay serves a cassette with no
    upstream calls. Binds to 127.0.0.1 only. Stop with Ctrl-C.
    """
    from .proxy import ProxyConfig, create_proxy_server

    console = _console()
    if port < 0:
        console.print("[red]--port must be 0 or a positive port[/]")
        raise typer.Exit(code=2)
    if not replay and not upstream:
        console.print("[red]--upstream is required unless --replay is set[/]")
        raise typer.Exit(code=2)
    if replay:
        store = Store()
        try:
            run = store.load_run(replay)
        except RunNotFoundError as exc:
            console.print(f"[red]{_e(exc)}[/]")
            raise typer.Exit(code=2) from exc
        replay = run.run_id
    config = ProxyConfig(
        host="127.0.0.1",
        port=port,
        upstream_url=upstream,
        agent=agent,
        replay_run_id=replay,
        once=once,
    )
    try:
        server = create_proxy_server(config)
    except (OSError, ValueError, RunNotFoundError) as exc:
        console.print(f"[red]cannot start proxy: {_e(exc)}[/]")
        raise typer.Exit(code=2) from exc
    bound = server.server_address[1]
    url = f"http://127.0.0.1:{bound}"
    store_root = Store().root
    mode = f"replay run {replay}; no upstream" if replay else f"record → {upstream}"
    console.print(f"crashlab proxy  {url}  ({mode}; agent {_e(agent)}; store {_e(store_root)})")
    console.print(f"  OPENAI_BASE_URL={url}/v1")
    console.print(f"  ANTHROPIC_BASE_URL={url}")
    if once:
        console.print("  --once: one LLM request, then exit")
    else:
        console.print("  Ctrl-C to stop")
    try:
        if once:
            server.timeout = 300
            server.handle_request()
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        raise typer.Exit(code=0) from None
    finally:
        with contextlib.suppress(Exception):
            server.shutdown()
        server.server_close()
