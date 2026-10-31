"""Provider-independent scheduling loop shared by every Stage 2 plan.

Rules carried over from Stage 1: reserve and persist before each paid submission;
record the provider call ID before any waiting; reattach recorded calls on resume;
never respawn a key the ledger already knows; cancel and mark in-flight calls
uncertain if the loop stops unexpectedly. A plan decides which specs are ready.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from stage2.budget import BudgetStop, Stage2Ledger
from stage2.common import canonical_sha


@dataclass
class Hooks:
    persist: Callable[[], None]
    sync: Callable[[dict], Path]                 # download a result; return its local attempt directory
    sync_existing: Callable[[str], tuple[dict, Path]]  # re-sync an already completed key on resume
    submitted: Callable[[dict, object], None] = lambda spec, call: None
    completed: Callable[[str, dict], None] = lambda key, result: None
    after_completions: Callable[[], None] = lambda: None
    heartbeat: Callable[[dict], None] = lambda active: None
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.time


@dataclass
class RunState:
    active: dict = field(default_factory=dict)
    submitted: list = field(default_factory=list)


class CandidateFailureRateStop(BudgetStop):
    """Latched failure-rate stop: submitted calls are drained, never replaced."""


def result_outcome(result: dict) -> str:
    outcome = result.get("outcome", "success")  # historical successful pre-check results
    if outcome not in {"success", "candidate_failure"}:
        raise ValueError("Unrecognized terminal training outcome: " + str(outcome))
    if outcome == "candidate_failure" and (result.get("failure") or {}).get("classification") != "candidate_numerical":
        raise ValueError("Candidate failure requires explicit numerical classification")
    return outcome


def verify_result_identity(result: dict, key: str, spec_hash: str) -> None:
    if result.get("id") != key or result.get("spec_sha256") != spec_hash:
        raise ValueError("Returned result identity/specification differs from the reserved input: " + key)


def failure_rate_state(plan, ledger: Stage2Ledger) -> dict:
    """Per-plan terminal logical runs; infrastructure retries are not observations."""
    scope = getattr(getattr(plan, "cfg", None), "name", type(plan).__name__)
    state = ledger.data.setdefault("failure_rate_guards", {}).setdefault(scope, {"stopped": False})
    complete = [r for key, r in ledger.data["calls"].items() if plan.owns(key) and r["status"] == "complete"]
    failures = sum(r.get("outcome", "success") == "candidate_failure" for r in complete)
    state.update(completed_runs=len(complete), candidate_failures=failures,
                 failure_fraction=failures / len(complete) if complete else 0.0,
                 threshold=0.05, denominator="completed successful plus candidate-failed logical runs owned by this plan")
    # Integer arithmetic makes exactly 5% inclusive. Once crossed, later drain
    # successes cannot silently reopen submissions, including after a restart.
    if failures * 20 > len(complete) and not state["stopped"]:
        state.update(stopped=True, first_crossing_completed_runs=len(complete), first_crossing_candidate_failures=failures)
    return state


def run_plan(plan, ledger: Stage2Ledger, backend, hooks: Hooks, concurrency: int) -> RunState:
    state = RunState()
    try:
        order = getattr(plan, "reconcile_sort_key", lambda key: key)
        owned = sorted((k for k in ledger.data["calls"] if plan.owns(k)), key=order)
        for key in owned:
            record = ledger.data["calls"][key]
            spec = plan.spec_for_key(key)
            if record["spec_sha256"] != canonical_sha(spec):
                raise BudgetStop("Existing run specification differs: " + key)
            if record["status"] == "complete":
                result, directory = hooks.sync_existing(key)
                verify_result_identity(result, key, record["spec_sha256"])
                outcome = result_outcome(result)
                if record.get("outcome", outcome) != outcome:
                    raise ValueError("Stored and re-synced outcomes differ: " + key)
                record.update(outcome=outcome, failure=result.get("failure"))
                plan.record(key, result, directory)
            elif record["status"] in {"reserved", "running"}:
                if not record.get("function_call_id"):
                    ledger.uncertain(key, "Pending submission lacks a recoverable provider handle")
                    hooks.persist()
                    raise BudgetStop("Inspect provider state for interrupted submission: " + key)
                state.active[key] = backend.reattach(record["function_call_id"])
            else:
                raise BudgetStop("Uncertain call requires provider billing reconciliation: " + key)
        failure_guard = failure_rate_state(plan, ledger)
        hooks.persist()
        while True:
            blocked = None
            queue = ([] if failure_guard["stopped"] else
                     [s for s in plan.ready_specs(ledger) if s["id"] not in ledger.data["calls"]])
            while queue and len(state.active) < concurrency:
                spec = queue.pop(0)
                try:
                    reservation = ledger.reserve(spec["id"], canonical_sha(spec), plan.deadline_seconds(spec))
                except BudgetStop as exc:
                    blocked = exc
                    break
                hooks.persist()  # durability boundary BEFORE the paid invocation
                try:
                    call = backend.spawn(spec, reservation["absolute_deadline"])
                except BaseException as exc:
                    ledger.uncertain(spec["id"], f"Submission outcome unknown: {type(exc).__name__}: {exc}")
                    hooks.persist()
                    raise
                state.active[spec["id"]] = call
                state.submitted.append(spec["id"])
                ledger.associate(spec["id"], call.object_id, backend.app_id)
                hooks.persist()
                hooks.submitted(spec, call)
            if not state.active:
                if failure_guard["stopped"]:
                    raise CandidateFailureRateStop(
                        "Candidate numerical failure rate exceeded 5%; no replacement runs; submitted calls drained. "
                        f"Observed {failure_guard['candidate_failures']}/{failure_guard['completed_runs']} terminal runs.")
                if blocked is not None:
                    raise blocked
                if plan.done(ledger):
                    return state
                raise RuntimeError("Plan is not done but has nothing in flight and nothing ready to submit")
            completed = False
            for key, call in list(state.active.items()):
                record = ledger.data["calls"][key]
                result = backend.poll(call)
                if result is None:
                    if hooks.now() >= record["absolute_deadline"]:
                        raise BudgetStop("Shared call deadline reached: " + key)
                    continue
                # Charge submit-to-observed-result wall time, including queueing,
                # startup and any automatic preemption restarts.
                verify_result_identity(result, key, record["spec_sha256"])
                outcome = result_outcome(result)
                record.update(outcome=outcome, failure=result.get("failure"))
                ledger.complete(key, hooks.now() - record["submitted_unix_seconds"], result["usage"])
                del state.active[key]
                failure_guard = failure_rate_state(plan, ledger)
                hooks.persist()
                directory = hooks.sync(result)
                plan.record(key, result, directory)
                hooks.completed(key, result)
                completed = True
            if completed:
                hooks.after_completions()
                hooks.persist()
                ledger.assert_can_launch()
            hooks.heartbeat(state.active)
            if state.active and not completed:
                hooks.sleep(5)
    except BaseException:
        for key, call in list(state.active.items()):
            error = None
            try:
                backend.cancel(call)
            except Exception as exc:
                error = f"; cancellation failed: {type(exc).__name__}: {exc}"
            ledger.uncertain(key, "Controller stopped before recording a completed result" + (error or ""))
        hooks.persist()
        raise
