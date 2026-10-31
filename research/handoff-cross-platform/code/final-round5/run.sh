#!/usr/bin/env bash
# Fixed run command for every node: recreate the locked env, run all seeds, print the summary.
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
  uv run --locked python run_seeds.py "$@"
else
  # No uv on this host: build a venv from the requirements exported from uv.lock.
  python3 -m venv .venv
  .venv/bin/pip install -q --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
  .venv/bin/python run_seeds.py "$@"
fi
