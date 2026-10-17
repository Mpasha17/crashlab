# crashlab

**Crash-test your AI agents.**

crashlab is an open-source, framework-agnostic Python library that turns agent failures into fixes you can trust:

- **Record** every LLM and tool call from your existing agent loop (OpenAI, Anthropic, LiteLLM), with no rewrite.
- **Replay** any failed run exactly, offline, and fork it at any step.
- **Diagnose** the failing step and its root cause, backed by verbatim evidence from the trace.
- **Verify** a proposed fix by rerunning the failure and your past good runs.
- **Export** the failure as a pytest regression test.

> Status: early development. v0.1 is coming soon.

## License

Apache-2.0
