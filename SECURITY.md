# Security

crashlab is a local-first debugging tool. Treat recorded runs as sensitive.

## Secrets
- API keys and auth headers are redacted before anything is written under `.crashlab/`.
- Never commit `.crashlab/`, live stores, or env files. Run `scripts/scan.sh` (or an equivalent) before every push. It flags real
  `AQ.Ab8…` / `gsk_…` *values*; naming those prefixes in docs is fine. Never commit
  full key values.
- Do not paste keys into issues, PRs, case studies, or demo transcripts.

## Supply chain
- Pre-alpha: pin dependencies in your own venv; we are not on PyPI yet.
- Review typed patches before applying them to production prompts.

## Reporting
Email or open a private GitHub security advisory on
[Mpasha17/crashlab](https://github.com/Mpasha17/crashlab). Do not file public issues
that include live cassettes with user data.
