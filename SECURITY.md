# Security

crashlab is a local-first debugging tool. Treat recorded runs as sensitive.

## Secrets
- API keys and auth headers are redacted before anything is written under `.crashlab/`.
- Never commit `.crashlab/`, live stores, or env files. The repo's `scan.sh` (or
  equivalent) should be run before every push; look for `AQ.Ab8`, `gsk_`, and full
  key values.
- Do not paste keys into issues, PRs, case studies, or demo transcripts.

## Supply chain
- Pre-alpha: pin dependencies in your own venv; we are not on PyPI yet.
- Review typed patches before applying them to production prompts.

## Reporting
Email or open a private GitHub security advisory on
[Mpasha17/crashlab](https://github.com/Mpasha17/crashlab). Do not file public issues
that include live cassettes with user data.
