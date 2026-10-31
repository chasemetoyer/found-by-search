"""Adversarial result/registration fixtures; no training or provider calls."""
from __future__ import annotations

import hashlib
import json

import pytest

from stage2.common import canonical_sha
from stage2.registration import REQUIRED_FROZEN_INPUTS, TESTED_IMPLEMENTATION, authorize
from stage2.search import SEARCH_PATH_SMOKE, SearchPlan, candidate_spec


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def recorded_result(tmp_path):
    plan = SearchPlan(SEARCH_PATH_SMOKE)
    cfg = plan.cfg
    spec = candidate_spec(cfg, "random", 0, 0, cfg.tasks[0], cfg.rs_seeds(0)[0], plan.rs[0])
    directory = tmp_path / "attempt"
    directory.mkdir()
    write_json(directory / "spec.json", spec)
    write_json(directory / "summary.json", {
        "seed": spec["seed"], "total_timesteps": cfg.budget_steps,
        "first_solved_step": None, "first_goal_step": None,
    })
    (directory / "metrics.jsonl").write_text(json.dumps({
        "global_step": cfg.budget_steps, "train/success": 0.25,
    }) + "\n")
    result = {"id": spec["id"], "spec_sha256": canonical_sha(spec), "outcome": "success"}
    return plan, spec, result, directory


def test_matching_result_is_scored(recorded_result):
    plan, spec, result, directory = recorded_result
    plan.record(spec["id"], result, directory)
    assert plan.areas == {spec["id"]: 0.25}


@pytest.mark.parametrize("outcome", ["success", "candidate_failure"])
@pytest.mark.parametrize("mutation", ["returned_id", "returned_hash", "missing_hash", "saved_spec"])
def test_wrong_result_identity_is_rejected_before_scoring(recorded_result, mutation, outcome):
    plan, spec, result, directory = recorded_result
    result["outcome"] = outcome
    if mutation == "returned_id":
        result["id"] = "another-logical-run"
    elif mutation == "returned_hash":
        result["spec_sha256"] = "0" * 64
    elif mutation == "missing_hash":
        result.pop("spec_sha256")
    else:
        saved = json.loads((directory / "spec.json").read_text())
        saved["seed"] += 1
        write_json(directory / "spec.json", saved)
    with pytest.raises(ValueError, match="identity|specification"):
        plan.record(spec["id"], result, directory)
    assert plan.areas == {} and plan.failed_runs == set()


def test_wrong_learner_seed_is_rejected_before_scoring(recorded_result):
    plan, spec, result, directory = recorded_result
    summary = json.loads((directory / "summary.json").read_text())
    summary["seed"] += 1
    write_json(directory / "summary.json", summary)
    with pytest.raises(ValueError, match="summary seed"):
        plan.record(spec["id"], result, directory)
    assert plan.areas == {} and plan.failed_runs == set()


@pytest.fixture
def full_gate(tmp_path):
    """A synthetic public lock exists only under pytest's temporary directory."""
    record = {"mode": "search", "profile": "registered"}
    for name in REQUIRED_FROZEN_INPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture source\n")
    write_json(tmp_path / "stage2/mode.json", record)
    smoke_path = tmp_path / "stage2/search-path-smoke-receipt.json"
    smoke = {
        "status": "PASS", "mode": {"mode": "search", "profile": "smoke"},
        "run_id": "fixture-smoke-run", "tested_commit": "fixture-tested-commit",
        "implementation_sha256": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in TESTED_IMPLEMENTATION
        },
    }
    write_json(smoke_path, smoke)
    lock_path = tmp_path / "preregistration/stage2-lock.json"
    receipt_path = tmp_path / "preregistration/stage2-github-receipt.json"

    def reseal(omit=None):
        names = REQUIRED_FROZEN_INPUTS - ({omit} if omit else set())
        write_json(lock_path, {"sha256": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names
        }})
        write_json(receipt_path, {
            "repository_visibility": "PUBLIC", "lock_commit": "fixture-lock-commit",
            "github_ref_commit": "fixture-lock-commit",
            "lock_manifest_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        })

    reseal()
    authorize(record, tmp_path)
    return tmp_path, record, lock_path, receipt_path, smoke_path, reseal


@pytest.mark.parametrize("which", ["lock", "receipt", "smoke"])
def test_missing_registration_evidence_blocks_full_search(full_gate, which):
    root, record, lock, receipt, smoke, _ = full_gate
    {"lock": lock, "receipt": receipt, "smoke": smoke}[which].unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        authorize(record, root)


@pytest.mark.parametrize(("field", "value"), [
    ("repository_visibility", "PRIVATE"),
    ("lock_commit", ""),
    ("github_ref_commit", "another-commit"),
    ("lock_manifest_sha256", "0" * 64),
])
def test_unverified_or_mismatched_public_receipt_blocks_full_search(full_gate, field, value):
    root, record, _, receipt_path, _, _ = full_gate
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        authorize(record, root)


@pytest.mark.parametrize("omitted", sorted(REQUIRED_FROZEN_INPUTS))
def test_every_required_input_must_be_frozen(full_gate, omitted):
    root, record, _, _, _, reseal = full_gate
    # Receipt hashes match the incomplete lock; the mandatory-set check must
    # reject it independently of ordinary digest mismatch detection.
    reseal(omit=omitted)
    with pytest.raises(ValueError, match="mandatory frozen inputs"):
        authorize(record, root)


def test_code_change_after_lock_blocks_full_search(full_gate):
    root, record, _, _, _, _ = full_gate
    (root / "stage2/search.py").write_text("changed after lock\n")
    with pytest.raises(ValueError, match="Frozen Stage 2 input changed"):
        authorize(record, root)


def test_new_lock_cannot_authorize_code_changed_since_smoke(full_gate):
    root, record, _, _, _, reseal = full_gate
    (root / "stage2/controller.py").write_text("changed after passing smoke\n")
    reseal()  # internally consistent new lock/receipt, but stale smoke proof
    with pytest.raises(ValueError, match="untested implementation"):
        authorize(record, root)


@pytest.mark.parametrize("mutation", [
    "failed", "missing_status", "wrong_mode", "missing_run", "missing_commit", "missing_hash", "wrong_hash",
])
def test_smoke_proof_must_be_passing_exact_path_and_same_code(full_gate, mutation):
    root, record, _, _, smoke_path, reseal = full_gate
    smoke = json.loads(smoke_path.read_text())
    if mutation == "failed":
        smoke["status"] = "FAIL"
    elif mutation == "missing_status":
        smoke.pop("status")
    elif mutation == "wrong_mode":
        smoke["mode"] = {"mode": "smoke"}
    elif mutation == "missing_run":
        smoke.pop("run_id")
    elif mutation == "missing_commit":
        smoke.pop("tested_commit")
    elif mutation == "missing_hash":
        smoke["implementation_sha256"].pop("stage2/scheduler.py")
    else:
        smoke["implementation_sha256"]["stage2/scheduler.py"] = "0" * 64
    write_json(smoke_path, smoke)
    reseal()  # verify smoke semantics, not merely its outer file hash
    with pytest.raises(ValueError, match="smoke|untested implementation"):
        authorize(record, root)
