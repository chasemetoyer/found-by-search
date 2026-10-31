#!/bin/sh
# Fixed tracked command. Approved Mac controller; all training remains Modal.
set -eu
test -n "${ORX_RUN_ID:-}" || { echo 'Use tracked project runner' >&2; exit 2; }
test -n "${ORX_COMMIT:-}" || { echo 'Recorded source required' >&2; exit 2; }
unset STAGE2_CLOUD_CONTROLLER
uv run --locked --default-index https://pypi.org/simple python -m stage2.mac
