"""Failure contracts using tiny fake records; no training, environments, or provider calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stage2.budget import BudgetStop, Stage2Ledger
from stage2.common import canonical_sha
from stage2.run_training import (
    CANDIDATE_FAILURE_EXIT, CandidateNumericalFailure, check_finite_outputs,
    nonfinite_exception_evidence, validated_outcome, write_candidate_failure,
)
from stage2.scheduler import CandidateFailureRateStop, Hooks, run_plan


def test_only_observed_nonfinite_distribution_tensor_qualifies():
    for exception in (ValueError("logits contain nan"), RuntimeError("probability tensor contains nan"),
                      ImportError("nan"), FileNotFoundError("checkpoint")):
        assert nonfinite_exception_evidence(exception) is None
    with pytest.raises(ValueError) as caught:
        torch.distributions.Categorical(logits=torch.tensor([float("nan"), 1.]))
    evidence = nonfinite_exception_evidence(caught.value)
    assert evidence["phase"] == "categorical_parameter_validation" and evidence["nonfinite_elements"] == 2
    with pytest.raises(ValueError) as caught:
        torch.distributions.Categorical(probs=torch.tensor([-1., 2.]))
    assert nonfinite_exception_evidence(caught.value) is None


def test_candidate_outcome_requires_spec_hash_reason_and_exit(tmp_path):
    spec = {"config": {"intrinsic": {"kind": "generic_homeostatic"}}, "id": "fake"}
    failure = CandidateNumericalFailure("nonfinite_checkpoint_tensor", {"tensor": "actor.weight", "nonfinite_elements": 1})
    write_candidate_failure(tmp_path, spec, failure)
    assert validated_outcome(tmp_path, spec, CANDIDATE_FAILURE_EXIT)["outcome"] == "candidate_failure"
    with pytest.raises(RuntimeError):
        validated_outcome(tmp_path, spec, 1)
    with pytest.raises(ValueError):
        validated_outcome(tmp_path, {**spec, "id": "other"}, CANDIDATE_FAILURE_EXIT)
    with pytest.raises(RuntimeError):
        write_candidate_failure(tmp_path, {"config": {"intrinsic": {"kind": "none"}}}, failure)


def test_postrun_nonfinite_scan_and_missing_data_remain_distinct(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"v_loss": 0.5}\n')
    (tmp_path / "summary.json").write_text('{"first_solved_step": null}\n')
    torch.save({"actor": torch.tensor([1.])}, tmp_path / "agent.pt")
    check_finite_outputs(tmp_path)
    torch.save({"actor": torch.tensor([float("inf")])}, tmp_path / "agent.pt")
    with pytest.raises(CandidateNumericalFailure, match="nonfinite_checkpoint_tensor"):
        check_finite_outputs(tmp_path)
    (tmp_path / "agent.pt").unlink()
    with pytest.raises(FileNotFoundError):
        check_finite_outputs(tmp_path)
    (tmp_path / "metrics.jsonl").write_text('not-json\n')
    with pytest.raises(json.JSONDecodeError):
        check_finite_outputs(tmp_path)


class FakePlan:
    cfg = SimpleNamespace(name="failure-contract")

    def __init__(self, count):
        self.count, self.areas = count, {}

    def owns(self, key):
        return key.startswith("failure-contract-")

    def spec_for_key(self, key):
        return {"id": key}

    def deadline_seconds(self, spec):
        return 900

    def ready_specs(self, ledger):
        return [{"id": f"failure-contract-{i:03d}"} for i in range(self.count)]

    def record(self, key, result, directory):
        self.areas[key] = 0. if result.get("outcome") == "candidate_failure" else 1.

    def done(self, ledger):
        return len(self.areas) == self.count


class FakeBackend:
    app_id = "ap-fake"

    def __init__(self, failing=(), exhausted=()):
        self.failing, self.exhausted = set(failing), set(exhausted)
        self.spawned, self.reattached, self.cancelled = [], [], []
        self.calls = {}

    def spawn(self, spec, deadline):
        self.spawned.append(spec["id"])
        call = SimpleNamespace(object_id="fc-" + spec["id"], spec=spec)
        self.calls[call.object_id] = call
        return call

    def reattach(self, call_id):
        self.reattached.append(call_id)
        return self.calls[call_id]

    def poll(self, call):
        key = call.spec["id"]
        if key in self.exhausted:
            raise RuntimeError("SDK retries exhausted")
        failed = key in self.failing
        return {"id": key, "spec_sha256": canonical_sha(call.spec),
                "outcome": "candidate_failure" if failed else "success", "attempt": key,
                "failure": {"classification": "candidate_numerical"} if failed else None,
                "usage": {"wall_seconds": 1., "compute_cost_estimate_usd": "0.0001"}, "files": {}}

    def cancel(self, call):
        self.cancelled.append(call.object_id)


def hooks(tmp_path, store):
    def sync(result):
        store[result["id"]] = result
        return tmp_path
    return Hooks(persist=lambda: None, sync=sync, sync_existing=lambda key: (store[key], tmp_path), sleep=lambda seconds: None)


def prefill(ledger, backend, store, count):
    for index in range(count):
        spec = {"id": f"failure-contract-{index:03d}"}
        ledger.reserve(spec["id"], canonical_sha(spec), 900)
        call = backend.spawn(spec, 0)
        ledger.associate(spec["id"], call.object_id, backend.app_id)
        result = backend.poll(call)
        ledger.data["calls"][spec["id"]]["outcome"] = result["outcome"]
        ledger.complete(spec["id"], 1., result["usage"])
        store[spec["id"]] = result


def test_failure_stops_submissions_drains_and_persists_zero_without_replacement(tmp_path):
    backend = FakeBackend(failing={"failure-contract-000"})
    ledger, plan, store = Stage2Ledger(tmp_path / "ledger.json"), FakePlan(10), {}
    with pytest.raises(CandidateFailureRateStop):
        run_plan(plan, ledger, backend, hooks(tmp_path, store), concurrency=4)
    assert len(backend.spawned) == 4 and not backend.cancelled
    assert len(plan.areas) == 4 and plan.areas["failure-contract-000"] == 0.
    assert all(record["status"] == "complete" for record in ledger.data["calls"].values())
    guard = ledger.data["failure_rate_guards"][plan.cfg.name]
    assert guard["first_crossing_completed_runs"] == 1 and guard["completed_runs"] == 4
    restored, restored_plan = Stage2Ledger(ledger.path), FakePlan(10)
    with pytest.raises(CandidateFailureRateStop):
        run_plan(restored_plan, restored, backend, hooks(tmp_path, store), concurrency=4)
    assert restored_plan.areas == plan.areas and len(backend.spawned) == 4


def test_exact_five_percent_continues_and_prior_unowned_runs_are_excluded(tmp_path):
    backend = FakeBackend(failing={"failure-contract-019"})
    ledger, store = Stage2Ledger(tmp_path / "ledger.json"), {}
    prefill(ledger, backend, store, 20)
    ledger.data["calls"]["old-smoke-failure"] = {"status": "complete", "outcome": "candidate_failure", "guard_charge_usd": "0"}
    plan = FakePlan(21)
    run_plan(plan, ledger, backend, hooks(tmp_path, store), concurrency=1)
    assert len(backend.spawned) == 21 and plan.areas["failure-contract-019"] == 0
    assert ledger.data["failure_rate_guards"][plan.cfg.name]["candidate_failures"] == 1
    assert ledger.data["failure_rate_guards"][plan.cfg.name]["completed_runs"] == 21


def test_failure_rate_latch_remains_after_drain_reduces_fraction(tmp_path):
    backend = FakeBackend(failing={"failure-contract-000"})
    ledger, store, plan = Stage2Ledger(tmp_path / "ledger.json"), {}, FakePlan(40)
    with pytest.raises(CandidateFailureRateStop):
        run_plan(plan, ledger, backend, hooks(tmp_path, store), concurrency=32)
    guard = ledger.data["failure_rate_guards"][plan.cfg.name]
    assert guard["failure_fraction"] < .05 and guard["stopped"] and len(backend.spawned) == 32
    with pytest.raises(CandidateFailureRateStop):
        run_plan(FakePlan(40), Stage2Ledger(ledger.path), backend, hooks(tmp_path, store), concurrency=32)
    assert len(backend.spawned) == 32


def test_exhausted_infrastructure_call_stops_without_zero_or_resubmission(tmp_path):
    backend = FakeBackend(exhausted={"failure-contract-000"})
    ledger, store, plan = Stage2Ledger(tmp_path / "ledger.json"), {}, FakePlan(4)
    with pytest.raises(RuntimeError, match="SDK retries exhausted"):
        run_plan(plan, ledger, backend, hooks(tmp_path, store), concurrency=2)
    assert plan.areas == {} and len(backend.spawned) == 2
    assert all(record["status"] == "uncertain" for record in ledger.data["calls"].values())
    with pytest.raises(BudgetStop):
        run_plan(FakePlan(4), Stage2Ledger(ledger.path), backend, hooks(tmp_path, store), concurrency=2)
    assert len(backend.spawned) == 2


@pytest.mark.parametrize("field,value", [("id", "wrong-run"), ("spec_sha256", "wrong-sha")])
def test_mismatched_result_identity_never_completes_or_scores(tmp_path, field, value):
    backend = FakeBackend()
    original_poll = backend.poll
    backend.poll = lambda call: {**original_poll(call), field: value}
    ledger, plan = Stage2Ledger(tmp_path / "ledger.json"), FakePlan(1)
    with pytest.raises(ValueError, match="identity/specification"):
        run_plan(plan, ledger, backend, hooks(tmp_path, {}), concurrency=1)
    assert plan.areas == {} and ledger.data["calls"]["failure-contract-000"]["status"] == "uncertain"
