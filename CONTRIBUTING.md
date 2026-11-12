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

The test suite refuses to start if `import crashlab` would load a copy other than this
checkout's `src/` (another worktree's editable install, an older wheel), because the
results would describe other code. Use `pip install -e .` in this checkout, or run
`PYTHONPATH=src python -m pytest` from its root. To test a different install on purpose,
set `CRASHLAB_TEST_ANY_INSTALL=1`.
