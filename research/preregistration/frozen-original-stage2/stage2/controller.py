"""Tracked local Stage 2 controller: orchestration only; every run is a Modal Function input.

Start only through the project's fixed run command (scripts/run_stage1.sh), which
requires ORX_RUN_ID and ORX_COMMIT. The committed stage2/mode.json selects the plan.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath

import modal

from stage2.budget import BudgetStop, Stage2Ledger, LIMIT_USD
from stage2.common import (
    CONTROLLER_LOCK_NAME, MAX_CONCURRENT_CALLS, PROJECT_ID, ROOT, SHARED, VOLUME_NAME,
    atomic_json, cpu_class, parse_cpuinfo, sha256_file, utc_now,
)
from stage2.modal_app import app, train_one, volume
from stage2.scheduler import Hooks, run_plan

# Both profiles execute the search branch now; full search stays gated by the
# public registration, while the approved rehearsal uses only tiny smoke data.
MODES = ("smoke", "search")
HEARTBEAT_SECONDS = 30
VOLUME_SWEEP_SECONDS = 600
BILLING_MIN_INTERVAL_SECONDS = 120
CPU_LOG = SHARED / "stage2-container-cpu-log.jsonl"


def make_plan(mode: str):
    if mode == "smoke":
        from stage2.search import SMOKE, SearchPlan
        return SearchPlan(SMOKE)
    if mode == "search":
        from stage2.registration import configured_search
        from stage2.search import SearchPlan
        return SearchPlan(configured_search())
    raise SystemExit(f"Mode not available on this node: {mode}")


class ModalBackend:
    def __init__(self, commit: str, run_id: str):
        self.commit, self.run_id = commit, run_id

    @property
    def app_id(self) -> str:
        return app.app_id

    def spawn(self, spec: dict, absolute_deadline: float):
        return train_one.spawn(spec, absolute_deadline, self.commit, self.run_id)

    def reattach(self, call_id: str):
        return modal.FunctionCall.from_id(call_id)

    def poll(self, call):
        try:
            return call.get(timeout=0)
        except TimeoutError:
            return None

    def cancel(self, call) -> None:
        call.cancel(terminate_containers=True)


def download(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(local.name + ".part")
    with temporary.open("wb") as stream:
        for chunk in volume.read_file(remote):
            stream.write(chunk)
    temporary.replace(local)


class Controller:
    def __init__(self, mode: str, results: Path, ledger: Stage2Ledger, billing_start: datetime):
        self.mode, self.results, self.ledger, self.billing_start = mode, results, ledger, billing_start
        from stage2.registration import mode_record
        self.is_registered_search = mode == "search" and mode_record().get("profile") == "registered"
        self.peak_running = 0
        self.last_heartbeat = self.last_sweep = self.last_billing = 0.0
        self.logged = set()
        if CPU_LOG.exists():
            self.logged = {json.loads(l)["attempt"] for l in CPU_LOG.read_text().splitlines() if l.strip()}

    # -- durable state -----------------------------------------------------------------
    def persist(self) -> None:
        self.ledger.save()
        with volume.batch_upload(force=True) as batch:
            batch.put_file(self.ledger.path, "/budget-ledger.json")
        self.publish()

    def publish(self) -> None:
        if not SHARED.is_dir():
            return
        calls = self.ledger.data["calls"]
        atomic_json(SHARED / "stage2-runtime-state.json", {
            "updated_utc": utc_now(), "tracked_run_id": os.environ.get("ORX_RUN_ID"),
            "source_commit": os.environ.get("ORX_COMMIT"), "mode": self.mode, "results_directory": str(self.results),
            "ledger_path": str(self.ledger.path), "guard_total_usd": str(self.ledger.guard_total()),
            "limit_usd": str(LIMIT_USD), "peak_running_containers": self.peak_running,
            "running_calls": {k: {"function_call_id": v.get("function_call_id"), "status": v["status"]}
                              for k, v in calls.items() if v["status"] != "complete"},
            "resume_instruction": "Reconnect this controller's recorded Modal call IDs; never resubmit a recorded key."})
        registry = []
        for key, call in sorted(calls.items()):
            measured = call.get("measured") or {}
            registry.append({"run_key": key, "status": call["status"], "function_call_id": call.get("function_call_id"),
                             "app_id": call.get("app_id"), "reserved_utc": call.get("reserved_utc"),
                             "completed_utc": call.get("completed_utc"), "reservation_usd": call.get("reservation_usd"),
                             "guard_charge_usd": call.get("guard_charge_usd"),
                             "measured_compute_cost_estimate_usd": measured.get("compute_cost_estimate_usd")})
        atomic_json(SHARED / "stage2-run-registry.json", {
            "schema": "stage2-run-registry-v1", "updated_utc": utc_now(),
            "write_ahead_record": "Each call is reserved and persisted before submission and its Modal call ID is "
                                  "recorded immediately after submission, before any waiting.",
            "calls": registry, "cpu_log": str(CPU_LOG)})
        self.publish_budget()

    def publish_budget(self) -> None:
        path = SHARED / "budget.json"
        budget = json.loads(path.read_text()) if path.exists() else {"schema": "stage-budget-v1", "stages": {}}
        calls = self.ledger.data["calls"].values()
        measured = sum((Decimal(c["measured"]["compute_cost_estimate_usd"]) for c in calls
                        if c["status"] == "complete" and c.get("measured")), Decimal(0))
        budget["stages"]["stage2"] = {
            "cap_usd": str(LIMIT_USD), "status": ("search in progress" if self.mode == "search" and self.is_registered_search else "pre-lock checks (no scientific search launched)"), "mode": self.mode,
            "guard_total_usd": str(self.ledger.guard_total()),
            "measured_completed_function_compute_usd": str(measured),
            "modal_billed_complete_intervals_usd": self.ledger.data.get("provider_reported_complete_intervals_cost_usd"),
            "calls": {s: sum(c["status"] == s for c in calls) for s in ("complete", "running", "reserved", "uncertain")},
            "ledger_path": str(self.ledger.path), "updated_utc": utc_now(),
            "rule": "Report guard, measured compute and Modal-billed amounts separately; the guard is not an invoice."}
        budget["updated_utc"] = utc_now()
        atomic_json(path, budget)

    # -- results ----------------------------------------------------------------------
    def sync(self, result: dict) -> Path:
        for remote_path, expected in result["files"].items():
            path = PurePosixPath(remote_path)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "runs":
                raise ValueError("Invalid remote output path")
            local = self.results.joinpath(*path.parts[1:])
            download(remote_path, local)
            if sha256_file(local) != expected:
                raise ValueError("Downloaded result hash mismatch: " + remote_path)
        atomic_json(self.results / result["id"] / "complete.json", result)
        directory = self.results / result["id"] / "attempts" / result["attempt"]
        self.log_cpu(result["id"], directory, "synced-results")
        return directory

    def sync_existing(self, key: str):
        local = self.results / key / "complete.json"
        download(f"/runs/{key}/complete.json", local)
        result = json.loads(local.read_text())
        return result, self.sync(result)

    def log_cpu(self, key: str, directory: Path, source: str) -> None:
        attempt = directory.name
        if attempt in self.logged or not (directory / "environment.json").exists():
            return
        environment = json.loads((directory / "environment.json").read_text())
        meta = json.loads((directory / "attempt.json").read_text()) if (directory / "attempt.json").exists() else {}
        probe = json.loads((directory / "isa_probe.json").read_text()) if (directory / "isa_probe.json").exists() else {}
        info = parse_cpuinfo(environment.get("cpuinfo"))
        row = {"logged_utc": utc_now(), "run_key": key, "attempt": attempt, "source_at_logging": source,
               "function_call_id": meta.get("modal_function_call_id"), "attempt_started_utc": meta.get("started_utc"),
               "environment_json_sha256": sha256_file(directory / "environment.json"), "cpu": info,
               "cpu_class": cpu_class(info), "isa_pins": environment.get("isa_pins"),
               "aten_cpu_capability": probe.get("aten_cpu_capability")}
        CPU_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CPU_LOG.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        with (self.results / "container-cpu-log.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.logged.add(attempt)

    def sweep_volume(self) -> None:
        """Copy metadata of attempts that never synced (for example preempted ones)."""
        found = {}
        for entry in volume.listdir("/runs", recursive=True):
            parts = PurePosixPath(entry.path.lstrip("/")).parts
            if parts and parts[0] != "runs":
                parts = ("runs",) + parts
            if len(parts) == 5 and parts[2] == "attempts":
                found.setdefault((parts[1], parts[3]), set()).add(parts[4])
        for (key, attempt), files in sorted(found.items()):
            if attempt in self.logged or "environment.json" not in files:
                continue
            destination = self.results / "volume-attempts" / key / attempt
            for name in ("environment.json", "attempt.json", "usage.json", "isa_probe.json"):
                if name in files:
                    download(f"/runs/{key}/attempts/{attempt}/{name}", destination / name)
            self.log_cpu(key, destination, "modal-volume-copy")

    # -- periodic -----------------------------------------------------------------------
    def heartbeat(self, active: dict) -> None:
        now = time.monotonic()
        if now - self.last_heartbeat < HEARTBEAT_SECONDS:
            return
        self.last_heartbeat = now
        running = None
        try:
            proc = subprocess.run(["modal", "container", "list", "--app-id", app.app_id, "--json"],
                                  capture_output=True, text=True, timeout=60)
            running = len(json.loads(proc.stdout or "[]"))
            self.peak_running = max(self.peak_running, running)
        except Exception:
            pass
        if now - self.last_sweep >= VOLUME_SWEEP_SECONDS:
            self.last_sweep = now
            try:
                self.sweep_volume()
            except Exception as exc:
                print(json.dumps({"event": "volume_sweep_error", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        print(json.dumps({"event": "waiting", "active": len(active), "running_containers": running,
                          "peak_running_containers": self.peak_running,
                          "guard_total_usd": str(self.ledger.guard_total())}), flush=True)
        self.publish()

    def billing(self, destination: Path, force: bool = False) -> None:
        if not force and time.monotonic() - self.last_billing < BILLING_MIN_INTERVAL_SECONDS:
            return
        self.last_billing = time.monotonic()
        try:
            rows = modal.Workspace.from_context().billing.report(start=self.billing_start, resolution="h",
                                                                 tag_names=["project_id", "stage"])
            rows = [dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r for r in rows]
            selected = [r for r in rows if (r.get("tags") or {}).get("project_id") == PROJECT_ID
                        and (r.get("tags") or {}).get("stage") == "2"]
            amount = sum((Decimal(str(r.get("cost", "0"))) for r in selected), Decimal(0))
            self.ledger.data["provider_reported_complete_intervals_cost_usd"] = str(amount)
            atomic_json(destination, json.loads(json.dumps({"queried_utc": utc_now(), "rows": selected,
                        "coverage": "Complete hourly intervals only; recent costs may be absent"}, default=str)))
        except Exception as exc:
            atomic_json(destination, {"queried_utc": utc_now(), "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    if not os.environ.get("ORX_RUN_ID") or not os.environ.get("ORX_COMMIT"):
        raise SystemExit("Use the tracked fixed project run command")
    mode = json.loads((ROOT / "stage2/mode.json").read_text())["mode"]
    if mode not in MODES:
        raise SystemExit(f"Unknown Stage 2 mode: {mode}")
    plan = make_plan(mode)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    lock = modal.Dict.from_name(CONTROLLER_LOCK_NAME, create_if_missing=True)
    owner = os.environ["ORX_RUN_ID"]
    if not lock.put("owner", owner, skip_if_exists=True):
        raise BudgetStop("Another Stage 2 controller owns the budget; inspect it before resuming")
    ledger = controller = None
    app_started = False
    try:
        ledger_path = results / "budget-ledger.json"
        try:
            download("/budget-ledger.json", ledger_path)
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass  # first Stage 2 run only; any other error must stop rather than reset spending history
        ledger = Stage2Ledger(ledger_path)
        ledger.assert_can_launch()
        billing_start = datetime.fromisoformat(ledger.data["created_utc"])
        controller = Controller(mode, results, ledger, billing_start)
        controller.persist()
        controller.billing(results / "billing-before.json", force=True)
        ledger.start_setup()
        controller.persist()
        with modal.enable_output(), app.run():
            app_started = True
            ledger.finish_setup()
            controller.persist()
            print(json.dumps({"event": "stage2_start", "mode": mode, "app_id": app.app_id,
                              "max_concurrent_calls": MAX_CONCURRENT_CALLS, "volume": VOLUME_NAME}), flush=True)
            backend = ModalBackend(os.environ["ORX_COMMIT"], owner)
            hooks = Hooks(persist=controller.persist, sync=controller.sync, sync_existing=controller.sync_existing,
                          submitted=lambda spec, call: print(json.dumps({"event": "submitted", "id": spec["id"],
                                                             "function_call_id": call.object_id,
                                                             "guard_total_usd": str(ledger.guard_total())}), flush=True),
                          completed=lambda key, result: print(json.dumps({"event": "run_recorded", "id": key,
                                                              "attempt": result["attempt"],
                                                              "guard_total_usd": str(ledger.guard_total())}), flush=True),
                          after_completions=lambda: controller.billing(results / "billing-latest.json"),
                          heartbeat=controller.heartbeat)
            run_plan(plan, ledger, backend, hooks, MAX_CONCURRENT_CALLS)
            controller.sweep_volume()
            if hasattr(plan, "selection_record"):
                selection_path = results / f"{plan.cfg.name}-selection.json"
                atomic_json(selection_path, plan.selection_record())
                with volume.batch_upload(force=True) as batch:
                    batch.put_file(selection_path, f"/plans/{plan.cfg.name}-selection.json")
        controller.billing(results / "billing-after.json", force=True)
        controller.persist()
        summary = plan.summary()
        summary.update(peak_running_containers=controller.peak_running, max_concurrent_calls=MAX_CONCURRENT_CALLS,
                       guard_total_usd=str(ledger.guard_total()))
        atomic_json(results / f"{mode}-summary.json", summary)
        if SHARED.is_dir():
            atomic_json(SHARED / f"stage2-{mode}-summary.json", summary)
        print(json.dumps({"event": "stage2_plan_complete", "mode": mode,
                          "summary": {k: summary[k] for k in summary if not isinstance(summary[k], (list, dict))}}),
              flush=True)
    except BaseException as exc:
        if ledger is not None:
            if not app_started and not isinstance(exc, BudgetStop):
                ledger.setup_failed(f"{type(exc).__name__}: {exc}")
            if controller is not None:
                controller.persist()
            atomic_json(results / "controller-stop.json", {"utc": utc_now(), "error": f"{type(exc).__name__}: {exc}",
                        "guard_total_usd": str(ledger.guard_total()), "invoice_cost_usd": None,
                        "search_started": controller.is_registered_search if controller is not None else False})
        raise
    finally:
        lock.pop("owner")


if __name__ == "__main__":
    main()
