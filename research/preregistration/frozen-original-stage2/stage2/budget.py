"""Stage 2 spending guard: same conservative rules as Stage 1, user-approved $35 cap.

One ledger is shared by every Stage 2 node (pre-lock checks, smoke tests and the
search), so all of them count toward the same cap. Rates rechecked 2026-10-28 at
https://modal.com/pricing: $0.0000131 per physical core-second and $0.00000222
per GiB-second. Estimates are never invoice claims.
"""
from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import Path

from stage1.budget import (
    CPU_LIMIT, CPU_USD_CORE_SECOND, RAM_LIMIT_GIB, RAM_USD_GIB_SECOND, RATE,
    TERMINATION_ALLOWANCE_SECONDS, BudgetStop, atomic_json, estimate_cost, utc_now,
)
from stage2.common import MAX_CONCURRENT_CALLS

LIMIT_USD = Decimal("35")
OVERHEAD_ALLOWANCE_USD = Decimal("0.50")
SETUP_ATTEMPT_GUARD_USD = Decimal("0.25")
MAX_CALL_RESERVATION_USD = Decimal("1")

__all__ = ["Stage2Ledger", "BudgetStop", "LIMIT_USD", "RATE", "estimate_cost", "utc_now", "atomic_json"]


class Stage2Ledger:
    """Reserve before submission; never respawn a recorded key; block on uncertainty."""

    def __init__(self, path: Path, max_concurrent: int = MAX_CONCURRENT_CALLS):
        self.path = Path(path)
        self.max_concurrent = max_concurrent
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {
                "schema": 3, "stage": 2, "created_utc": utc_now(), "limit_usd": str(LIMIT_USD),
                "overhead_allowance_usd": str(OVERHEAD_ALLOWANCE_USD),
                "price_source": "https://modal.com/pricing", "price_checked_utc_date": "2026-10-28",
                "cpu_usd_core_second": str(CPU_USD_CORE_SECOND), "ram_usd_gib_second": str(RAM_USD_GIB_SECOND),
                "cpu_limit": CPU_LIMIT, "ram_limit_gib": RAM_LIMIT_GIB, "max_concurrent_calls": max_concurrent,
                "provider_invoice_cost_usd": None, "calls": {}, "setup_failures": [],
                "limitations": "Estimates include a protective overhead allowance. Provider accounting is "
                               "delayed; this is not a guaranteed invoice cap.",
            }
        # Exact one-time increase explicitly approved by the user. Retain every
        # prior charge/call and an append-only cap amendment; never reset a ledger.
        if self.data.get("stage") == 2 and self.data.get("limit_usd") == "30":
            self.data.setdefault("limit_amendments", []).append({
                "utc": utc_now(), "previous_limit_usd": "30", "limit_usd": "35",
                "authorization": "User approval: Stage-2 cap $35. Keep the 3,600 s deadline and 10 finalist seeds.",
            })
            self.data["limit_usd"] = "35"
            self.save()
        if self.data.get("stage") != 2 or self.data["limit_usd"] != str(LIMIT_USD):
            raise BudgetStop("Ledger is not the Stage 2 ledger with the approved $35 cap")

    def guard_total(self) -> Decimal:
        estimates = sum((Decimal(c["guard_charge_usd"]) for c in self.data["calls"].values()), Decimal(0))
        setup_count = len(self.data.get("setup_failures", [])) + int(self.data.get("setup_build_inflight", False))
        estimates += setup_count * SETUP_ATTEMPT_GUARD_USD
        reported = Decimal(self.data.get("provider_reported_complete_intervals_cost_usd", "0"))
        return max(estimates, reported) + Decimal(self.data["overhead_allowance_usd"])

    def in_flight(self) -> int:
        return sum(c["status"] in {"reserved", "running"} for c in self.data["calls"].values())

    def save(self) -> None:
        self.data["updated_utc"] = utc_now()
        self.data["guard_total_usd"] = str(self.guard_total())
        atomic_json(self.path, self.data)

    def assert_can_launch(self) -> None:
        uncertain = [k for k, c in self.data["calls"].items() if c["status"] == "uncertain"]
        if uncertain:
            raise BudgetStop("Uncertain charges require provider-state reconciliation: " + ", ".join(uncertain))
        if self.guard_total() >= LIMIT_USD:
            raise BudgetStop("Stage 2 spending guard reached $35")

    def setup_failed(self, reason: str) -> None:
        self.data["setup_build_inflight"] = False
        self.data.setdefault("setup_failures", []).append({"utc": utc_now(), "reason": reason})
        self.save()

    def start_setup(self) -> None:
        self.assert_can_launch()
        if self.data.get("setup_build_inflight"):
            raise BudgetStop("Previous image-build state must be reconciled before another setup")
        if self.guard_total() + SETUP_ATTEMPT_GUARD_USD > LIMIT_USD:
            raise BudgetStop("Setup reservation would cross the Stage 2 $35 guard")
        self.data["setup_build_inflight"] = True
        self.data["setup_attempt_guard_usd"] = str(SETUP_ATTEMPT_GUARD_USD)
        self.data["setup_guard_basis"] = "Protective reservation per failed or in-flight build, not invoice cost"
        self.save()

    def finish_setup(self) -> None:
        self.data["setup_build_inflight"] = False
        self.save()

    def reserve(self, key: str, spec_sha256: str, deadline_seconds: int) -> dict:
        self.assert_can_launch()
        if key in self.data["calls"]:
            raise BudgetStop("Call already exists in spending ledger: " + key)
        if self.in_flight() >= self.max_concurrent:
            raise BudgetStop(f"All {self.max_concurrent} concurrent-call reservations are occupied")
        amount = estimate_cost(deadline_seconds + TERMINATION_ALLOWANCE_SECONDS)
        if amount > MAX_CALL_RESERVATION_USD:
            raise BudgetStop("Per-call reservation exceeds $1")
        if self.guard_total() + amount > LIMIT_USD:
            raise BudgetStop("Next call would cross the Stage 2 $35 spending guard")
        started = time.time()
        record = {"status": "reserved", "reserved_utc": utc_now(), "spec_sha256": spec_sha256,
                  "function_call_id": None, "reservation_usd": str(amount), "guard_charge_usd": str(amount),
                  "deadline_seconds": deadline_seconds, "invoice_cost_usd": None,
                  "submitted_unix_seconds": started, "absolute_deadline": started + deadline_seconds}
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
        record.update(status="complete", completed_utc=utc_now(), controller_elapsed_seconds=elapsed_seconds,
                      guard_charge_usd=str(charged), measured=result)
        self.save()

    def uncertain(self, key: str, reason: str) -> None:
        record = self.data["calls"][key]
        record.update(status="uncertain", reason=reason, stopped_utc=utc_now(),
                      guard_charge_usd=max(record["guard_charge_usd"], record["reservation_usd"], key=Decimal))
        self.save()
