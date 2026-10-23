#!/usr/bin/env bash
# Minimal secret-scan for contributors without the private live helper.
# Looks for real key *values*, not the documentation that names the prefixes.
set -euo pipefail
cd "$(dirname "$0")/.."
# gsk_ followed by a long token-looking string; AQ.Ab8 followed by more base64-ish chars
hits=$(git grep -nE 'AQ\.Ab8[A-Za-z0-9_-]{8,}|gsk_[A-Za-z0-9]{20,}' \
  -- ':!docs/demo/*.gif' ':!*.gif' ':!scripts/scan.sh' || true)
if [[ -n "$hits" ]]; then
  echo "$hits"
  echo "FAIL: key-like patterns found"
  exit 1
fi
echo "pattern hits: 0"
echo "keys check: ok"
