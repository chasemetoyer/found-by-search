"""Publish non-secret controller state so any authorized agent can reconnect."""
import os
from pathlib import Path
from stage1.budget import atomic_json, utc_now

SHARED = Path('/Users/chasemetoyer/Desktop/evolved-intrinsic-rewards/handoff-cross-platform')


def publish_runtime(ledger, results):
    if not SHARED.parent.is_dir():
        return
    atomic_json(SHARED / 'runtime-state.json', {
        'updated_utc': utc_now(), 'tracked_run_id': os.environ.get('ORX_RUN_ID'),
        'source_commit': os.environ.get('ORX_COMMIT'), 'results_directory': str(results),
        'budget_ledger_path': str(results / 'budget-ledger.json'), 'budget_ledger': ledger,
        'running_calls': {k: v for k, v in ledger.get('calls', {}).items()
                          if v.get('status') != 'complete'},
        'resume_instruction': 'Reconcile this tracked controller and its existing Modal call IDs before launching any replacement.',
        'stage_boundary': 'Stage 1 only; stop after report or USD 15 guard.'})
