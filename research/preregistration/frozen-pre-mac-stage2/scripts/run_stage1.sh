#!/bin/sh
# Fixed tracked run contract (project run command: bash scripts/run_stage1.sh).
# This Stage2 entry selects verification or remote orchestration from the
# approved cloud profile. Every training input is a separate Modal Function.
# Credentials are injected externally and are never stored in this script.
set -eu
test -n "${ORX_RUN_ID:-}" || { echo 'Start through the project experiment runner.' >&2; exit 2; }
test -n "${ORX_COMMIT:-}" || { echo 'A recorded source commit is required.' >&2; exit 2; }
# Remote controller is a Sandbox; training remains in separate Modal Functions.
if [ "$(uname -s)" = Linux ]; then export STAGE2_CLOUD_CONTROLLER=1; fi
uv run --locked --default-index https://pypi.org/simple python -m stage2.cloud
