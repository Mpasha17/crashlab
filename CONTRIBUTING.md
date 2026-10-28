# Contributing

crashlab is pre-alpha. Useful contributions right now:

1. **Integration reports** — wrap a real agent, add `@crashlab.check` oracles, open an
   issue with the cassette id and what broke (or worked). See `examples/integrations/`.
2. **Offline tests** — pure unit tests under `tests/`, no live keys. Run
   `pytest` and `ruff check .` before a PR.
3. **Docs** — fix quickstart pain; keep claims honest (CIs, small-n, quotas). Case
   studies under `docs/case-studies/` must not invent results.

Please do **not** commit API keys, `.crashlab/` live stores, or `crashlab-live/`.
Do not publish to PyPI from a fork. Do not force-push `main`.

Dev setup: `python -m venv .venv && .venv/bin/pip install -e ".[dev,openai,anthropic,litellm]"`.
Pre-push: `./scripts/check.sh` (ruff + offline pytest + secret-scan), or at least
`./scripts/scan.sh` before every push.
