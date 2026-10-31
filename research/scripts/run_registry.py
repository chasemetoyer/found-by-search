"""Shared run registry, per-stage budget record and append-only container CPU log.

Never submits, cancels or waits on Modal calls (no FunctionCall.get). Reads the
tracked controller's budget ledger and synced outputs, lists the Modal volume and
copies attempt metadata that was never synced (for example a preempted attempt).
Writes only into the shared handoff folder, mirrored into this worktree.

The controller ledger remains the write-ahead record: it reserves and persists
each call before submission and records its Modal call ID before any waiting
(stage1/modal_app.py run_phase). This registry is a readable, CPU-annotated view
regenerated from that ledger. Agents refresh it before every wait or wake.

Modal's gVisor sandbox reports "model name: unknown", so the CPU model is
identified from vendor, family, model and instruction-set flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "64555583-12e3-4641-82a0-b8841b663e70"
SHARED = Path("/Users/chasemetoyer/Desktop/evolved-intrinsic-rewards/handoff-cross-platform")
LOCAL = ROOT / "handoff-cross-platform"
RUN_ROOT = Path("/Users/chasemetoyer/.local/share/openresearch/local-runs")
VOLUME_NAME = "evolved-intrinsic-rewards-stage1"
COPIES = Path("stage1-container-cpu/volume-attempts")
COPIED_FILES = ("environment.json", "attempt.json", "usage.json")
CPU_LOG = "container-cpu-log.jsonl"
STAGE1_CAP_USD = Decimal("15")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def microarchitecture(vendor, family, model, flags):
    """Range-based labels checked against CPUID references; anything else stays unmapped."""
    if vendor == "AuthenticAMD" and family == 0x19 and model is not None:
        if 0x00 <= model <= 0x0F:
            return "AMD Zen 3, EPYC 7003 Milan range (family 19h, models 00h-0Fh)"
        if 0x10 <= model <= 0x1F:
            return "AMD Zen 4, EPYC 9004 Genoa range (family 19h, models 10h-1Fh)"
    if vendor == "GenuineIntel" and family == 6 and model == 0x55:
        label = "Intel Xeon Scalable, family 6 model 55h (Skylake-SP/Cascade Lake/Cooper Lake; stepping hidden)"
        if "avx512_vnni" in flags and "avx512_bf16" not in flags:
            label += "; AVX512_VNNI without AVX512_BF16 indicates Cascade Lake class"
        return label
    return f"unmapped ({vendor} family {family} model {model})"


def aten_capability(flags):
    """Inferred PyTorch ATen CPU kernel tier: AVX512 needs avx512vl/bw/dq and FMA3."""
    if {"avx512vl", "avx512bw", "avx512dq", "fma"} <= flags:
        return "AVX512"
    if {"avx2", "fma"} <= flags:
        return "AVX2"
    return "DEFAULT"


def parse_cpuinfo(text):
    text = text or ""
    fields = {}
    for line in text.split("\n\n")[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()

    def integer(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    flags = set(fields.get("flags", "").split())
    vendor, family, model = fields.get("vendor_id"), integer(fields.get("cpu family")), integer(fields.get("model"))
    return {
        "vendor_id": vendor, "cpu_family": family, "model": model,
        "stepping": fields.get("stepping"), "model_name_reported": fields.get("model name"),
        "cpu_mhz_reported": fields.get("cpu MHz"), "logical_processors": text.count("processor\t:"),
        **{flag: flag in flags for flag in ("avx2", "fma", "avx512f", "avx512bw", "avx512dq",
                                             "avx512vl", "avx512_vnni", "avx512_bf16")},
        "microarchitecture": microarchitecture(vendor, family, model, flags),
        "inferred_aten_cpu_capability": aten_capability(flags),
    }


def tracked_runs():
    proc = subprocess.run(["orx", "runs", PROJECT], capture_output=True, text=True)
    runs = []
    for line in proc.stdout.splitlines():
        if " reason:" in line:
            continue
        match = re.match(r"^([0-9a-f-]{36})\s+(\S+)\s+(.+?)\s+([0-9a-f]{7,40})\s+(\S+)\s+(.+?)\s*$", line)
        if match:
            runs.append(dict(run_id=match[1], status=match[2], experiment=match[3].strip(),
                             commit=match[4], duration=match[5], updated=match[6]))
    reasons = dict(re.findall(r"(?m)^([0-9a-f-]{36})\s+reason:\s*(.+?)\s*$", proc.stdout))
    for run in runs:
        run["reason"] = reasons.get(run["run_id"])
        active = RUN_ROOT / run["run_id"] / "repo/results"
        record = SHARED / "stage1-run-records" / run["run_id"]
        run["results_directory"] = str(active) if active.is_dir() else None
        ledger = active / "budget-ledger.json" if (active / "budget-ledger.json").exists() else record / "budget-ledger.json"
        run["ledger_path"] = str(ledger) if ledger.exists() else None
        apps = set()
        if run["ledger_path"]:
            apps |= {c.get("app_id") for c in read_json(ledger).get("calls", {}).values() if c.get("app_id")}
        if (record / "controller.log").exists():
            apps |= set(re.findall(r"ap-[A-Za-z0-9]{22}", (record / "controller.log").read_text(errors="replace")))
        run["modal_app_ids"] = sorted(apps)
        stop = read_json(record / "controller-stop.json") or read_json(active / "controller-stop.json")
        run["controller_stop"] = stop
    return runs, proc.returncode


def active_results(runs, explicit):
    if explicit:
        return Path(explicit)
    candidates = [Path(r["results_directory"]) for r in runs
                  if r["results_directory"] and (Path(r["results_directory"]) / "budget-ledger.json").exists()]
    if not candidates:
        raise SystemExit("No tracked controller results directory with a budget ledger was found")
    return max(candidates, key=lambda p: (p / "budget-ledger.json").stat().st_mtime)


def volume_listing():
    import modal
    volume = modal.Volume.from_name(VOLUME_NAME)
    found = {}
    for entry in volume.listdir("/runs", recursive=True):
        parts = Path(entry.path.lstrip("/")).parts
        if parts and parts[0] != "runs":
            parts = ("runs",) + parts
        if len(parts) == 5 and parts[2] == "attempts":
            found.setdefault((parts[1], parts[3]), set()).add(parts[4])
    return volume, found


def collect_attempts(results, volume, found):
    attempts = {}
    for environment in sorted(results.glob("*/attempts/*/environment.json")):
        attempts[environment.parent.name] = dict(run_key=environment.parents[2].name, attempt=environment.parent.name,
                                                 source="synced-results", directory=str(environment.parent))
    for (key, attempt), files in sorted(found.items()):
        if attempt in attempts or "environment.json" not in files:
            continue
        destination = SHARED / COPIES / key / attempt
        for name in COPIED_FILES:
            if name in files and not (destination / name).exists():
                raw = b"".join(volume.read_file(f"/runs/{key}/attempts/{attempt}/{name}"))
                destination.mkdir(parents=True, exist_ok=True)
                temporary = destination / (name + ".part")
                temporary.write_bytes(raw)
                temporary.replace(destination / name)
        attempts[attempt] = dict(run_key=key, attempt=attempt, source="modal-volume-copy", directory=str(destination))
    for record in attempts.values():
        directory = Path(record["directory"])
        raw = (directory / "environment.json").read_bytes()
        environment = json.loads(raw)
        meta = read_json(directory / "attempt.json") or {}
        usage = read_json(directory / "usage.json") or {}
        record.update(environment_json_sha256=hashlib.sha256(raw).hexdigest(),
                      started_utc=meta.get("started_utc"), function_call_id=meta.get("modal_function_call_id"),
                      modal_input_id=meta.get("modal_input_id"), platform=environment.get("platform"),
                      cpu=parse_cpuinfo(environment.get("cpuinfo")),
                      wall_seconds=usage.get("wall_seconds"), training_core_seconds=usage.get("training_core_seconds"),
                      attempt_error=usage.get("error"))
    return attempts


def append_cpu_log(attempts):
    path = SHARED / CPU_LOG
    logged = set()
    if path.exists():
        logged = {json.loads(line)["attempt"] for line in path.read_text().splitlines() if line.strip()}
    new = [a for a in sorted(attempts.values(), key=lambda a: (a["started_utc"] or "", a["attempt"]))
           if a["attempt"] not in logged]
    with path.open("a") as stream:
        for a in new:
            stream.write(json.dumps({
                "logged_utc": now(), "run_key": a["run_key"], "attempt": a["attempt"],
                "function_call_id": a["function_call_id"], "modal_input_id": a["modal_input_id"],
                "attempt_started_utc": a["started_utc"], "source_at_logging": a["source"],
                "environment_json_sha256": a["environment_json_sha256"], "platform": a["platform"], "cpu": a["cpu"],
            }, sort_keys=True) + "\n")
    return len(new)


def billing_total(results):
    snapshot = read_json(results / "billing-latest.json") or {}
    rows = snapshot.get("rows") or []
    return (str(sum((Decimal(str(r.get("cost", "0"))) for r in rows), Decimal(0))),
            snapshot.get("queried_utc"), snapshot.get("coverage"),
            {r.get("object_id"): r.get("cost") for r in rows})


def build(results, runs, attempts):
    ledger = read_json(results / "budget-ledger.json")
    manifest = read_json(ROOT / "stage1/manifest.json")
    calls = ledger.get("calls", {})
    by_run = {}
    for a in attempts.values():
        by_run.setdefault(a["run_key"], []).append(a)
    rows, counts = [], {}
    for spec in manifest["runs"]:
        key = spec["id"]
        call = calls.get(key)
        status = call["status"] if call else "queued (not yet submitted)"
        counts[status] = counts.get(status, 0) + 1
        complete = read_json(results / key / "complete.json")
        final_attempt = complete.get("attempt") if complete else None
        listed = sorted(by_run.get(key, []), key=lambda a: (a["started_utc"] or "", a["attempt"]))
        container_rows = []
        for index, a in enumerate(listed):
            if final_attempt:
                role = "final" if a["attempt"] == final_attempt else "superseded (preempted or failed)"
            else:
                role = "current (in progress)" if index == len(listed) - 1 else "superseded (preempted or failed)"
            cpu = a["cpu"]
            container_rows.append(dict(attempt=a["attempt"], role=role, source=a["source"], started_utc=a["started_utc"],
                                       vendor_id=cpu["vendor_id"], cpu_family=cpu["cpu_family"], model=cpu["model"],
                                       microarchitecture=cpu["microarchitecture"], avx512f=cpu["avx512f"],
                                       inferred_aten_cpu_capability=cpu["inferred_aten_cpu_capability"],
                                       training_core_seconds=a["training_core_seconds"], attempt_error=a["attempt_error"]))
        measured = (call or {}).get("measured") or {}
        rows.append(dict(run_key=key, purpose=spec["purpose"], task=spec["task"], method=spec["method"],
                         seed=spec["seed"], budget_steps=spec["total_timesteps"], replicate=spec["replicate"],
                         detector=spec.get("detector"), status=status,
                         function_call_id=(call or {}).get("function_call_id"), app_id=(call or {}).get("app_id"),
                         reserved_utc=(call or {}).get("reserved_utc"), completed_utc=(call or {}).get("completed_utc"),
                         reservation_usd=(call or {}).get("reservation_usd"),
                         guard_charge_usd=(call or {}).get("guard_charge_usd"),
                         measured_compute_cost_estimate_usd=measured.get("compute_cost_estimate_usd"),
                         wall_seconds=measured.get("wall_seconds"), containers=container_rows,
                         container_count=len(container_rows)))
    cpu_types = {}
    for a in attempts.values():
        cpu_types[a["cpu"]["microarchitecture"]] = cpu_types.get(a["cpu"]["microarchitecture"], 0) + 1
    registry = {
        "schema": "stage1-run-registry-v1", "updated_utc": now(),
        "write_ahead_record": "The controller budget ledger reserves and persists each call before submission and records "
                              "its Modal call ID immediately after submission, before any waiting (stage1/modal_app.py "
                              "run_phase). This registry is regenerated from that ledger plus synced outputs and the Modal volume.",
        "ledger_path": str(results / "budget-ledger.json"), "ledger_updated_utc": ledger.get("updated_utc"),
        "tracked_runs": runs, "manifest_entries": len(rows), "status_counts": counts,
        "containers_logged": len(attempts), "containers_by_cpu_type": cpu_types, "calls": rows,
        "cpu_identification": "gVisor reports 'model name: unknown'; CPU type is derived from /proc/cpuinfo vendor, "
                              "family, model and flags saved by every worker attempt. ATen capability is inferred from flags.",
    }
    in_flight = [c for c in calls.values() if c["status"] in {"reserved", "running"}]
    completed = [c for c in calls.values() if c["status"] == "complete"]
    uncertain = [c for c in calls.values() if c["status"] == "uncertain"]
    setup = Decimal(ledger.get("setup_attempt_guard_usd", "0")) * len(ledger.get("setup_failures", []))
    components = {
        "overhead_allowance_usd": str(Decimal(ledger.get("overhead_allowance_usd", "0"))),
        "setup_failure_reservations_usd": str(setup),
        "in_flight_reservations_usd": str(sum((Decimal(c["reservation_usd"]) for c in in_flight), Decimal(0))),
        "completed_guard_charges_usd": str(sum((Decimal(c["guard_charge_usd"]) for c in completed), Decimal(0))),
        "uncertain_charges_usd": str(sum((Decimal(c.get("guard_charge_usd") or c["reservation_usd"]) for c in uncertain), Decimal(0))),
    }
    component_total = sum((Decimal(v) for v in components.values()), Decimal(0))
    billed, billed_utc, coverage, billed_by_app = billing_total(results)
    guard = Decimal(ledger["guard_total_usd"])
    budget = {
        "schema": "stage-budget-v1", "updated_utc": now(),
        "rule": "Report these numbers separately; never add them. The guard is protective, not an invoice.",
        "stages": {
            "paper_impact_audit": {"cap_usd": None, "status": "complete",
                                   "spend_note": "Local read-only audit of archived records; no Modal calls recorded."},
            "stage1": {
                "cap_usd": str(STAGE1_CAP_USD), "status": "in progress", "ledger_path": str(results / "budget-ledger.json"),
                "ledger_updated_utc": ledger.get("updated_utc"),
                "guard_total_usd": str(guard), "guard_fraction_of_cap": str((guard / STAGE1_CAP_USD).quantize(Decimal("0.0001"))),
                "guard_components": components, "guard_components_match_ledger": component_total == guard,
                "measured_completed_function_compute_usd": str(sum((Decimal(c["measured"]["compute_cost_estimate_usd"])
                                                                    for c in completed if c.get("measured")), Decimal(0))),
                "modal_billed_complete_intervals_usd": billed, "modal_billed_queried_utc": billed_utc,
                "modal_billed_coverage": coverage, "modal_billed_by_app": billed_by_app,
                "provider_invoice_usd": ledger.get("provider_invoice_cost_usd"),
                "calls": {"manifest": len(rows), "complete": len(completed), "in_flight": len(in_flight),
                          "uncertain": len(uncertain), "not_yet_submitted": counts.get("queued (not yet submitted)", 0)},
                "setup_failures": ledger.get("setup_failures"), "limitations": ledger.get("limitations"),
            },
            "stage2": {"cap_usd": None, "status": "not authorized"},
            "stage3": {"cap_usd": None, "status": "not authorized"},
        },
    }
    return registry, budget


def mirror():
    if LOCAL.resolve() == SHARED.resolve():
        return
    for name in ("run-registry.json", "budget.json", CPU_LOG):
        if (SHARED / name).exists():
            shutil.copy2(SHARED / name, LOCAL / name)
    if (SHARED / COPIES).exists():
        shutil.copytree(SHARED / COPIES, LOCAL / COPIES, dirs_exist_ok=True)


def sync(args):
    runs, code = tracked_runs()
    results = active_results(runs, args.results)
    volume, found = (None, {}) if args.no_volume else volume_listing()
    attempts = collect_attempts(results, volume, found)
    added = append_cpu_log(attempts)
    registry, budget = build(results, runs, attempts)
    registry["orx_runs_exit_code"] = code
    registry["volume_listed"] = not args.no_volume
    atomic(SHARED / "run-registry.json", registry)
    atomic(SHARED / "budget.json", budget)
    mirror()
    print(json.dumps({"updated_utc": registry["updated_utc"], "status_counts": registry["status_counts"],
                      "containers_logged": registry["containers_logged"], "new_cpu_log_rows": added,
                      "guard_total_usd": budget["stages"]["stage1"]["guard_total_usd"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", help="Controller results directory; default is the newest tracked run ledger")
    parser.add_argument("--no-volume", action="store_true", help="Skip the Modal volume listing")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    if not args.watch:
        sync(args)
        return
    while True:
        try:
            sync(args)
        except Exception:
            print(json.dumps({"utc": now(), "error": traceback.format_exc()}), flush=True)
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    main()
