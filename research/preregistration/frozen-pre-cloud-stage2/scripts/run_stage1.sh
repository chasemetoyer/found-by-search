#!/bin/sh
# Fixed tracked run contract (project run command: bash scripts/run_stage1.sh).
# On this Stage 2 branch it starts the Stage 2 controller; the committed
# stage2/mode.json selects the plan. The controller is local orchestration
# only; every run is a separate Modal Function input. Credentials must already
# be configured externally.
set -eu
test -n "${ORX_RUN_ID:-}" || { echo 'Start through the project experiment runner.' >&2; exit 2; }
test -n "${ORX_COMMIT:-}" || { echo 'A recorded source commit is required.' >&2; exit 2; }
uv run --locked python -m stage2.controller
