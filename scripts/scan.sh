#!/usr/bin/env bash
# Minimal secret-scan for contributors without the private live helper.
set -euo pipefail
cd "$(dirname "$0")/.."
hits=$(git grep -nE 'AQ\.Ab8|gsk_[A-Za-z0-9]{20,}' -- ':!docs/demo/*.gif' ':!*.gif' || true)
if [[ -n "$hits" ]]; then
  echo "$hits"
  echo "FAIL: key-like patterns found"
  exit 1
fi
echo "pattern hits: 0"
echo "keys check: ok"
