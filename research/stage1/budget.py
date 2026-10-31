"""Conservative, persistent spending guard; estimates are never invoice claims.

Rates retrieved 2026-10-27 from https://modal.com/pricing.
Function CPU is physical cores. Startup and idle time are billable.
Provider budgets: https://modal.com/docs/guide/budgets (dashboard configuration;
no supported public SDK setter at implementation time).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

LIMIT_USD = Decimal("15")
CPU_USD_CORE_SECOND = Decimal("0.0000131")
RAM_USD_GIB_SECOND = Decimal("0.00000222")
CPU_LIMIT = 1
RAM_LIMIT_GIB = 2
RATE = CPU_USD_CORE_SECOND * CPU_LIMIT + RAM_USD_GIB_SECOND * RAM_LIMIT_GIB
# Kept reserved for image setup, volume retention, egress and unmeasured lifecycle
# overhead. This is a protective allowance, not a proven bound on those charges.
OVERHEAD_ALLOWANCE_USD = Decimal("0.50")
TERMINATION_ALLOWANCE_SECONDS = 60
CALL_DEADLINE_SECONDS = 7200
MAX_CONCURRENT_CALLS = 4
# The user explicitly approved further infrastructure repairs within USD15
# after the first two image-build failures. Scientific settings remain locked.
MAX_SETUP_FAILURES = None
SETUP_ATTEMPT_GUARD_USD = Decimal("0.25")


class BudgetStop(RuntimeError):
    """Do not launch another paid call until the cause is reconciled."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_cost(seconds: float) -> Decimal:
    return Decimal(str(max(0.0, seconds))) * RATE


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class BudgetLedger:
    """One controller; all in-flight reservations count toward the same cap.

    Pending calls with recorded provider handles can be resumed without spawning
    them again. Uncertain calls block new launches; do not erase their entries.
    Conservative estimates include complete submit-to-result wall time, plus
    termination allowance at the CPU/RAM limits, across automatic preemptions.
    This cannot certify an invoice cap if provider accounting is delayed.
    """

    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text())
        else:
            self.data = {
                "schema": 2, "created_utc": utc_now(), "limit_usd": str(LIMIT_USD),
                "overhead_allowance_usd": str(OVERHEAD_ALLOWANCE_USD),
                "price_source": "https://modal.com/pricing",
                "price_checked_utc_date": "2026-10-27",
                "cpu_usd_core_second": str(CPU_USD_CORE_SECOND),
                "ram_usd_gib_second": str(RAM_USD_GIB_SECOND),
                "cpu_limit": CPU_LIMIT, "ram_limit_gib": RAM_LIMIT_GIB,
                "provider_invoice_cost_usd": None, "calls": {}, "setup_failures": [],
                "limitations": "Estimates include a protective overhead allowance. "
                    "Provider accounting is delayed; this is not a guaranteed invoice cap.",
            }
        if self.data["limit_usd"] != str(LIMIT_USD):
            raise BudgetStop("Ledger spending limit differs from the registered $15 cap")

    def guard_total(self) -> Decimal:
        estimates = sum((Decimal(c["guard_charge_usd"]) for c in self.data["calls"].values()), Decimal(0))
        # Keep failed builds in the spending guard even when their invoice is
        # not yet available. This is a conservative reservation, not billed cost.
        setup_count = len(self.data.get("setup_failures", [])) + int(self.data.get("setup_build_inflight", False))
        estimates += setup_count * SETUP_ATTEMPT_GUARD_USD
        reported = Decimal(self.data.get("provider_reported_complete_intervals_cost_usd", "0"))
        return max(estimates, reported) + Decimal(self.data["overhead_allowance_usd"])

    def save(self) -> None:
        self.data["updated_utc"] = utc_now()
        self.data["guard_total_usd"] = str(self.guard_total())
        atomic_json(self.path, self.data)

    def assert_can_launch(self) -> None:
        uncertain = [k for k, c in self.data["calls"].items() if c["status"] == "uncertain"]
        if uncertain:
            raise BudgetStop("Uncertain charges require provider-state reconciliation: " + ", ".join(uncertain))
        if MAX_SETUP_FAILURES is not None and len(self.data.get("setup_failures", [])) >= MAX_SETUP_FAILURES:
            raise BudgetStop("Two setup failures reached; inspect provider billing before retrying")
        if self.guard_total() >= LIMIT_USD:
            raise BudgetStop("Stage 1 spending guard reached $15")

    def assert_reconciled(self) -> None:
        self.assert_can_launch()
        pending = [k for k, c in self.data["calls"].items() if c["status"] != "complete"]
        if pending:
            raise BudgetStop("Provider calls are still pending: " + ", ".join(pending))

    def setup_failed(self, reason: str) -> None:
        self.data["setup_build_inflight"] = False
        self.data.setdefault("setup_failures", []).append({"utc": utc_now(), "reason": reason})
        self.save()

    def start_setup(self) -> None:
        self.assert_can_launch()
        if self.data.get("setup_build_inflight"):
            raise BudgetStop("Previous image-build state must be reconciled before another setup")
        if self.guard_total() + SETUP_ATTEMPT_GUARD_USD > LIMIT_USD:
            raise BudgetStop("Setup reservation would cross the Stage 1 USD15 guard")
        self.data["setup_build_inflight"] = True
        self.data["setup_attempt_guard_usd"] = str(SETUP_ATTEMPT_GUARD_USD)
        self.data["setup_guard_basis"] = "Protective reservation per failed or in-flight build, not invoice cost"
        self.save()

    def finish_setup(self) -> None:
        self.data["setup_build_inflight"] = False
        self.save()

    def reserve(self, key: str, spec_sha256: str, deadline_seconds: int = CALL_DEADLINE_SECONDS) -> dict:
        self.assert_can_launch()
        if key in self.data["calls"]:
            raise BudgetStop("Call already exists in spending ledger: " + key)
        pending = sum(c["status"] in {"reserved", "running"} for c in self.data["calls"].values())
        if pending >= MAX_CONCURRENT_CALLS:
            raise BudgetStop("All four concurrent-call reservations are occupied")
        amount = estimate_cost(deadline_seconds + TERMINATION_ALLOWANCE_SECONDS)
        if amount > 1:
            raise BudgetStop("Per-call reservation exceeds $1")
        if self.guard_total() + amount > LIMIT_USD:
            raise BudgetStop("Next call would cross the Stage 1 $15 spending guard")
        started = time.time()
        record = {
            "status": "reserved", "reserved_utc": utc_now(),
            "spec_sha256": spec_sha256, "function_call_id": None,
            "reservation_usd": str(amount), "guard_charge_usd": str(amount),
            "deadline_seconds": deadline_seconds, "invoice_cost_usd": None,
            "submitted_unix_seconds": started, "absolute_deadline": started + deadline_seconds,
        }
        self.data["calls"][key] = record
        self.save()
        return record

    def associate(self, key: str, call_id: str, app_id: str) -> None:
        self.data["calls"][key].update(status="running", function_call_id=call_id, app_id=app_id)
        self.save()

    def complete(self, key: str, elapsed_seconds: float, result: dict) -> None:
        record = self.data["calls"][key]
        charged = estimate_cost(elapsed_seconds + TERMINATION_ALLOWANCE_SECONDS)
        if charged > Decimal(record["reservation_usd"]):
            record.update(status="uncertain", reason="Call exceeded its reserved wall envelope")
            self.save()
            raise BudgetStop(record["reason"])
        record.update(status="complete", completed_utc=utc_now(),
                      controller_elapsed_seconds=elapsed_seconds,
                      guard_charge_usd=str(charged), measured=result)
        self.save()

    def uncertain(self, key: str, reason: str) -> None:
        record = self.data["calls"][key]
        record.update(status="uncertain", reason=reason, stopped_utc=utc_now(),
                      guard_charge_usd=max(record["guard_charge_usd"], record["reservation_usd"], key=Decimal))
        # Keep the entire reservation; do not pretend failed compute was free.
        self.save()
