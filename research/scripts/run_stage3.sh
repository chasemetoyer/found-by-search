#!/bin/sh
# New Stage3 fixed entry point; all Stage2 locked entry files are unchanged.
set -eu
test -n "${ORX_RUN_ID:-}" || { echo 'Use the tracked project runner' >&2; exit 2; }
test -n "${ORX_COMMIT:-}" || { echo 'Recorded source required' >&2; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run --locked --default-index https://pypi.org/simple python -m stage3.controller
