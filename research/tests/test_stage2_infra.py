"""Stage 2 orchestration checks with a fake provider; no Modal calls, no training."""
from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

from stage2.budget import BudgetStop, LIMIT_USD, Stage2Ledger
from stage2.common import ISA_PINS, canonical_sha, cpu_class, parse_cpuinfo
from stage2.determinism import MAX_PINNED_REPLICATES, WAVE_SIZE, PinnedDeterminismPlan
from stage2.scheduler import Hooks, run_plan
from stage2.specs import pinned_determinism_spec, stage1_determinism_spec, validate

@pytest.fixture(autouse=True)
def pinned_mode(monkeypatch):
    """These checks exercise the pinned-determinism node's validator and plan."""
    import stage2.specs
    monkeypatch.setattr(stage2.specs, "committed_mode", lambda: "pinned_determinism")


CPUS = {"zen3": ("AuthenticAMD", 25, 1), "zen4": ("AuthenticAMD", 25, 17),
        "intel": ("GenuineIntel", 6, 85), "other": ("AuthenticAMD", 23, 49)}


def cpuinfo(kind: str) -> str:
    vendor, family, model = CPUS[kind]
    block = f"processor\t: 0\nvendor_id\t: {vendor}\ncpu family\t: {family}\nmodel\t\t: {model}\nmodel name\t: unknown\nflags\t\t: fpu avx2 fma\n"
    return block + "\n" + block.replace("processor\t: 0", "processor\t: 1")


def test_cpu_classes():
    for kind in CPUS:
        assert cpu_class(parse_cpuinfo(cpuinfo(kind))) == kind
    assert parse_cpuinfo(cpuinfo("zen3"))["logical_processors"] == 2


def test_pinned_spec_is_stage1_spec_plus_pins_only():
    base = stage1_determinism_spec()
    spec = pinned_determinism_spec(7)
    assert spec["config"] == base["config"] and spec["seed"] == 910 and spec["total_timesteps"] == 131_072
    assert {k for k in spec if spec.get(k) != base.get(k)} == {"id", "purpose", "replicate", "isa_pins"}
    assert spec["isa_pins"] == ISA_PINS
    validate(spec)
    for mutate in (lambda s: s["config"]["ppo"].update(learning_rate=0.001),
                   lambda s: s["isa_pins"].update(MKL_CBWR="AVX512"),
                   lambda s: s.update(seed=911), lambda s: s.update(purpose="search")):
        bad = copy.deepcopy(spec)
        mutate(bad)
        with pytest.raises(ValueError):
            validate(bad)
    with pytest.raises(ValueError):
        pinned_determinism_spec(MAX_PINNED_REPLICATES)


def test_ledger_cap_concurrency_and_reservations(tmp_path):
    ledger = Stage2Ledger(tmp_path / "ledger.json", max_concurrent=32)
    assert LIMIT_USD == Decimal("35")
    first = ledger.reserve("k0", "sha", 1200)
    assert Decimal(first["reservation_usd"]) == Decimal("1260") * Decimal("0.00001754")
    for i in range(1, 32):
        ledger.reserve(f"k{i}", "sha", 1200)
    with pytest.raises(BudgetStop):
        ledger.reserve("k32", "sha", 1200)  # 33rd concurrent call
    with pytest.raises(BudgetStop):
        ledger.reserve("k0", "sha", 1200)  # never twice
    ledger.data["stage"] = 1
    ledger.save()
    with pytest.raises(BudgetStop):
        Stage2Ledger(tmp_path / "ledger.json")


class FakeCall:
    def __init__(self, object_id):
        self.object_id = object_id
        self.polls = 0


class FakeBackend:
    app_id = "ap-fake"

    def __init__(self, classes, polls_needed=2, fail_spawn_for=None):
        self.classes, self.polls_needed, self.fail_spawn_for = classes, polls_needed, fail_spawn_for
        self.spawned, self.reattached, self.cancelled, self.calls = [], [], [], {}

    def spawn(self, spec, deadline):
        if spec["id"] == self.fail_spawn_for:
            raise ConnectionError("network")
        self.spawned.append(spec["id"])
        call = FakeCall("fc-" + spec["id"])
        self.calls[call.object_id] = (call, spec)
        return call

    def reattach(self, call_id):
        self.reattached.append(call_id)
        return self.calls[call_id][0]

    def poll(self, call):
        call.polls += 1
        if call.polls < self.polls_needed:
            return None
        spec = self.calls[call.object_id][1]
        return {"id": spec["id"], "attempt": "a-" + spec["id"], "files": {},
                "spec_sha256": canonical_sha(spec),
                "fingerprints": {"metrics_without_sps_sha256": "m", "summary_sha256": "s",
                                 "checkpoint_tensors_sha256": "t"},
                "usage": {"wall_seconds": 60.0, "compute_cost_estimate_usd": "0.001"}}

    def cancel(self, call):
        self.cancelled.append(call.object_id)


def make_hooks(tmp_path, backend, store):
    def sync(result):
        directory = tmp_path / "results" / result["id"] / "attempts" / result["attempt"]
        directory.mkdir(parents=True, exist_ok=True)
        replicate = int(result["id"].rsplit("-r", 1)[1])
        (directory / "environment.json").write_text(json.dumps({"cpuinfo": cpuinfo(backend.classes(replicate))}))
        store[result["id"]] = result
        return directory

    def sync_existing(key):
        result = store[key]
        return result, sync(result)

    return Hooks(persist=lambda: None, sync=sync, sync_existing=sync_existing, sleep=lambda s: None, now=lambda: 0.0)


def test_waves_stop_when_each_required_class_has_two(tmp_path):
    # Wave 1 has Zen 3 and Zen 4 only; wave 2 adds Intel at r32 and r33.
    classes = lambda r: "zen4" if r in (0, 1) else ("intel" if r in (32, 33) else "zen3")
    backend = FakeBackend(classes)
    ledger = Stage2Ledger(tmp_path / "ledger.json")
    plan = PinnedDeterminismPlan()
    run_plan(plan, ledger, backend, make_hooks(tmp_path, backend, {}), concurrency=32)
    assert len(backend.spawned) == 2 * WAVE_SIZE and len(set(backend.spawned)) == 2 * WAVE_SIZE
    assert plan.rule_satisfied() and plan.done(ledger)
    assert all(c["status"] == "complete" for c in ledger.data["calls"].values())
    assert plan.summary()["bit_identical_all_replicates"] is True


def test_replicate_cap_without_required_classes(tmp_path):
    backend = FakeBackend(lambda r: "zen3")
    ledger = Stage2Ledger(tmp_path / "ledger.json")
    plan = PinnedDeterminismPlan()
    run_plan(plan, ledger, backend, make_hooks(tmp_path, backend, {}), concurrency=32)
    assert len(backend.spawned) == MAX_PINNED_REPLICATES and not plan.rule_satisfied() and plan.done(ledger)


def test_resume_reattaches_and_never_resubmits(tmp_path):
    classes = lambda r: ("zen3", "zen4", "intel")[r % 3]
    backend = FakeBackend(classes)
    ledger = Stage2Ledger(tmp_path / "ledger.json")
    store = {}
    hooks = make_hooks(tmp_path, backend, store)
    for r in range(10):  # r0-r9 complete before the restart
        spec = pinned_determinism_spec(r)
        ledger.reserve(spec["id"], canonical_sha(spec), 1200)
        call = backend.spawn(spec, 0)
        ledger.associate(spec["id"], call.object_id, "ap-fake")
        ledger.complete(spec["id"], 60.0, {"wall_seconds": 60.0})
        store[spec["id"]] = backend.poll(call) or backend.poll(call)
    for r in range(10, 20):  # r10-r19 still running with recorded call IDs
        spec = pinned_determinism_spec(r)
        ledger.reserve(spec["id"], canonical_sha(spec), 1200)
        call = backend.spawn(spec, 0)
        ledger.associate(spec["id"], call.object_id, "ap-fake")
    before = list(backend.spawned)
    plan = PinnedDeterminismPlan()
    run_plan(plan, ledger, backend, hooks, concurrency=32)
    new = backend.spawned[len(before):]
    assert sorted(backend.reattached) == sorted(f"fc-{pinned_determinism_spec(r)['id']}" for r in range(10, 20))
    assert not set(new) & {pinned_determinism_spec(r)["id"] for r in range(20)}
    assert sorted(new) == sorted(pinned_determinism_spec(r)["id"] for r in range(20, 32))
    assert plan.done(ledger)


def test_unknown_submission_outcome_marks_uncertain_and_cancels(tmp_path):
    victim = pinned_determinism_spec(5)["id"]
    backend = FakeBackend(lambda r: "zen3", fail_spawn_for=victim)
    ledger = Stage2Ledger(tmp_path / "ledger.json")
    with pytest.raises(ConnectionError):
        run_plan(PinnedDeterminismPlan(), ledger, backend, make_hooks(tmp_path, backend, {}), concurrency=32)
    statuses = {k: c["status"] for k, c in ledger.data["calls"].items()}
    assert statuses[victim] == "uncertain" and len(backend.cancelled) == 5
    assert all(s == "uncertain" for s in statuses.values())
    with pytest.raises(BudgetStop):
        ledger.assert_can_launch()
