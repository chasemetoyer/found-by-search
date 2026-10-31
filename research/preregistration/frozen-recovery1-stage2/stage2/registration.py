"""Pre-lock search-path rehearsal and full-search authorization share one path.

The smoke profile changes only the plan data. It never authorizes scientific
seed panels. Full search requires a separately committed, publicly corroborated
registration and a matching receipt; neither is created by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stage2.common import ROOT

TESTED_IMPLEMENTATION = (
    "generic_reward.py", "stage1/corrected_ppo.py", "stage1/score.py", "stage1/statistics.py",
    "stage1/run_training.py", "stage1/budget.py",
    "stage2/__init__.py", "stage2/budget.py", "stage2/common.py", "stage2/controller.py",
    "stage2/modal_app.py", "stage2/registration.py", "stage2/run_training.py",
    "stage2/scheduler.py", "stage2/search.py", "stage2/specs.py", "stage2/isa_probe.py",
    "scripts/run_stage1.sh", "pyproject.toml", "uv.lock", ".python-version",
    "handoff-cross-platform/code/final-round5/ppo.py",
    "handoff-cross-platform/code/final-round5/intrinsic.py",
    "handoff-cross-platform/code/final-round5/config.toml",
)
REQUIRED_FROZEN_INPUTS = set(TESTED_IMPLEMENTATION) | {"preregistration/stage2-draft.md", "stage2/mode.json",
                                                     "stage2/search-path-smoke-receipt.json"}

def mode_record(root: Path = ROOT) -> dict:
    return json.loads((root / "stage2/mode.json").read_text())


def authorize(record: dict, root: Path = ROOT) -> None:
    if record.get("mode") != "search":
        return
    profile = record.get("profile")
    if profile == "smoke":
        if record != {"mode": "search", "profile": "smoke"}:
            raise ValueError("Unexpected search smoke overrides")
        return
    if profile != "registered" or record != {"mode": "search", "profile": "registered"}:
        raise ValueError("Search profile must be explicitly smoke or registered")
    lock_path = root / "preregistration/stage2-lock.json"
    receipt_path = root / "preregistration/stage2-github-receipt.json"
    if not lock_path.exists() or not receipt_path.exists():
        raise ValueError("Full search requires the public registration lock and GitHub receipt")
    lock = json.loads(lock_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("repository_visibility") != "PUBLIC":
        raise ValueError("The Stage 2 lock must be publicly corroborated")
    if receipt.get("lock_commit") != receipt.get("github_ref_commit") or not receipt.get("lock_commit"):
        raise ValueError("Stage 2 receipt does not corroborate the lock commit")
    if receipt.get("lock_manifest_sha256") != hashlib.sha256(lock_path.read_bytes()).hexdigest():
        raise ValueError("Stage 2 lock differs from its receipt")
    if not REQUIRED_FROZEN_INPUTS <= set(lock.get("sha256", {})):
        raise ValueError("Stage 2 registration omits mandatory frozen inputs")
    from stage2.recovery import authorized_hashes
    effective_hashes = authorized_hashes(lock, root)
    for name, expected in effective_hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid registration input path")
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen Stage 2 input changed: " + name)
    smoke = json.loads((root / "stage2/search-path-smoke-receipt.json").read_text())
    if smoke.get("status") != "PASS" or smoke.get("mode") != {"mode": "search", "profile": "smoke"}:
        raise ValueError("Exact search-path smoke has not passed")
    if not smoke.get("run_id") or not smoke.get("tested_commit"):
        raise ValueError("Search smoke receipt has no tracked run provenance")
    for name in TESTED_IMPLEMENTATION:
        if smoke.get("implementation_sha256", {}).get(name) != lock["sha256"][name]:
            raise ValueError("Stage 2 lock would freeze untested implementation: " + name)


def configured_search(record: dict | None = None, root: Path = ROOT):
    record = mode_record(root) if record is None else record
    authorize(record, root)
    if record.get("mode") != "search":
        raise ValueError("Search plan requested outside search mode")
    from stage2.search import REGISTERED_DRAFT, SEARCH_PATH_SMOKE
    return SEARCH_PATH_SMOKE if record["profile"] == "smoke" else REGISTERED_DRAFT
