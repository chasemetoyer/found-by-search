"""Exact search-mode wiring and safety gates; synthetic data, no learning."""
import copy
import hashlib
import json

import pytest

from stage2.budget import Stage2Ledger
from stage2.controller import make_plan
from stage2.registration import TESTED_IMPLEMENTATION, REQUIRED_FROZEN_INPUTS, authorize, configured_search, mode_record
from stage2.scheduler import run_plan
from stage2.search import REGISTERED_DRAFT, SEARCH_PATH_SMOKE, candidate_spec, random_candidates
from stage2.specs import validate
from test_stage2_search import FakeBackend, hooks_for


@pytest.fixture(autouse=True)
def rehearsal_configuration(monkeypatch):
    # The native smoke run verifies the committed mode record independently.
    # Unit fixtures remain usable after its data-only switch to registered.
    import stage2.registration
    monkeypatch.setattr(stage2.registration, 'mode_record', lambda *args, **kwargs: {"mode": "search", "profile": "smoke"})


def test_actual_search_controller_validator_and_plan_path(tmp_path):
    assert mode_record() == {"mode": "search", "profile": "smoke"}
    plan = make_plan("search")
    assert plan.cfg == SEARCH_PATH_SMOKE

    class ValidatingBackend(FakeBackend):
        def spawn(self, spec, deadline):
            validate(spec)  # same entry point as the Modal worker
            assert spec["seed"] >= 5000 and spec["total_timesteps"] == 8192
            return super().spawn(spec, deadline)

    backend, store = ValidatingBackend(), {}
    ledger = Stage2Ledger(tmp_path / "ledger.json")
    run_plan(plan, ledger, backend, hooks_for(tmp_path, backend, store), concurrency=32)
    assert len(backend.spawned) == len(set(backend.spawned)) == 72
    assert plan.summary()["runs_expected"] == plan.summary()["runs_recorded"] == 72
    assert plan.summary()["replay_reproduces_cma_candidates"]
    selection = plan.selection_record()
    assert set(selection["methods"]) == {"cma", "random"}
    assert all(len(row["finalists"]) == 3 for row in selection["methods"].values())


def test_rehearsal_rejects_scientific_seed_panels_and_overrides():
    cfg = REGISTERED_DRAFT
    spec = candidate_spec(cfg, "cma", 0, 0, cfg.tasks[0], cfg.cma_seeds(0)[0], random_candidates(cfg)[0])
    with pytest.raises(ValueError):
        validate(spec)
    with pytest.raises(ValueError):
        authorize({"mode": "search", "profile": "smoke", "budget_steps": 2000000})
    bad = candidate_spec(SEARCH_PATH_SMOKE, "cma", 99, 0, cfg.tasks[0], 5400, random_candidates(cfg)[0])
    with pytest.raises(ValueError):
        validate(bad)
    spec = candidate_spec(SEARCH_PATH_SMOKE, "cma", 0, 0, cfg.tasks[0], 5400, random_candidates(cfg)[0])
    for changes in ({"seed": 1000}, {"total_timesteps": 2000000}, {"task": "KeyCorridorS3R2"},
                    {"isa_pins": {"MKL_CBWR": "AVX2"}}):
        bad = copy.deepcopy(spec)
        bad.update(changes)
        with pytest.raises(ValueError):
            validate(bad)
    bad = copy.deepcopy(spec)
    bad["config"]["intrinsic"]["coef"] = .5
    with pytest.raises(ValueError):
        validate(bad)


def test_full_registration_gate_is_tested_without_creating_a_project_lock(tmp_path):
    # These files exist only in pytest's temporary fixture, never the project.
    record = {"mode": "search", "profile": "registered"}
    with pytest.raises(ValueError):
        authorize(record, tmp_path)
    (tmp_path / "preregistration").mkdir()
    digest = hashlib.sha256(b"fixture").hexdigest()
    for name in REQUIRED_FROZEN_INPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    smoke_path = tmp_path / "stage2/search-path-smoke-receipt.json"
    smoke_path.write_text(json.dumps({"status": "PASS", "mode": {"mode": "search", "profile": "smoke"},
                                    "run_id": "fixture-run", "tested_commit": "fixture-commit",
                                    "implementation_sha256": {name: digest for name in TESTED_IMPLEMENTATION}}))
    hashes = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in REQUIRED_FROZEN_INPUTS}
    lock = tmp_path / "preregistration/stage2-lock.json"
    lock.write_text(json.dumps({"sha256": hashes}))
    receipt = dict(repository_visibility="PUBLIC", lock_commit="fixture-commit", github_ref_commit="fixture-commit",
                   lock_manifest_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
    receipt_path = tmp_path / "preregistration/stage2-github-receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    assert configured_search(record, tmp_path) == REGISTERED_DRAFT
    receipt["repository_visibility"] = "PRIVATE"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        authorize(record, tmp_path)
    receipt["repository_visibility"] = "PUBLIC"
    receipt_path.write_text(json.dumps(receipt))
    (tmp_path / "stage2/search.py").write_text("tampered")
    with pytest.raises(ValueError):
        authorize(record, tmp_path)
    (tmp_path / "stage2/search.py").write_bytes(b"fixture")
    omitted = dict(hashes)
    omitted.pop("stage2/search.py")
    lock.write_text(json.dumps({"sha256": omitted}))
    receipt["lock_manifest_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        authorize(record, tmp_path)


def test_cap_increase_retains_existing_calls_and_costs(tmp_path):
    ledger = Stage2Ledger(tmp_path / "ledger.json")
    ledger.reserve("prior-input", "sha", 900)
    ledger.associate("prior-input", "fc-fixture", "ap-fixture")
    ledger.complete("prior-input", 60, {"wall_seconds": 60})
    calls = copy.deepcopy(ledger.data["calls"])
    guard = ledger.guard_total()
    ledger.data["limit_usd"] = "30"
    ledger.save()
    migrated = Stage2Ledger(ledger.path)
    assert migrated.data["limit_usd"] == "35"
    assert migrated.data["calls"] == calls and migrated.guard_total() == guard
    assert len(migrated.data["limit_amendments"]) == 1
