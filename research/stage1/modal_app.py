"""Tracked local controller, one isolated Modal Function input per training run.

Do not invoke this module directly. scripts/run_stage1.sh is the fixed project
run contract and rejects untracked invocation. Training executes only on Modal.

Official API references checked 2026-10-27:
https://modal.com/docs/sdk/py/latest/App
https://modal.com/docs/sdk/py/latest/Image
https://modal.com/docs/sdk/py/latest/Volume
https://modal.com/docs/sdk/py/latest/FunctionCall
https://modal.com/docs/guide/preemption
https://modal.com/docs/guide/timeouts
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath

import modal

from stage1.budget import (
    BudgetLedger, BudgetStop, CALL_DEADLINE_SECONDS, CPU_LIMIT, RAM_LIMIT_GIB,
    MAX_CONCURRENT_CALLS, atomic_json, estimate_cost, utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "evolved-intrinsic-rewards-stage1"
PROJECT_ID = "64555583-12e3-4641-82a0-b8841b663e70"
# Docker Registry manifest fetched by authenticated public registry API, no
# user credential, 2026-10-28. The full manifest is retained under the handoff.
BASE_IMAGE = "python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"
UV_VERSION = "0.9.7"
STARTUP_TIMEOUT_SECONDS = 120
EXCLUDED_PARTS = {".git", ".venv", ".agents", ".openresearch", ".orx", "__pycache__", ".pytest_cache", "results"}


def excluded(path: Path) -> bool:
    path = Path(path)
    if any(p in EXCLUDED_PARTS for p in path.parts) or path.name.startswith(".env"):
        return True
    relative = path.resolve().relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()
    files = {".python-version", "generic_reward.py", "pyproject.toml", "uv.lock"}
    directories = ("stage1/", "tests/", "scripts/", "preregistration/",
                   "handoff-cross-platform/code/final-round5/",
                   "handoff-cross-platform/code/a1b-run-commit-1533734/")
    return not (relative in files or relative.startswith(directories)
                or any(d.startswith(relative.rstrip("/") + "/") for d in directories))


image = (
    modal.Image.from_registry(BASE_IMAGE)
    .uv_sync(uv_project_dir=str(ROOT), uv_version=UV_VERSION,
             frozen=False, extra_options="--locked --default-index=https://pypi.org/simple --verbose",
             env={"UV_INDEX_URL": "https://pypi.org/simple", "UV_DEFAULT_INDEX": "https://pypi.org/simple"})
    .workdir("/workspace")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
          "PYTHONUNBUFFERED": "1", "MODAL_IS_REMOTE": "1"})
    .add_local_dir(ROOT, "/workspace", ignore=excluded)
)
app = modal.App("evolved-intrinsic-rewards-stage1",
                tags={"project_id": PROJECT_ID, "stage": "1"}, include_source=False)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def package_inventory() -> dict:
    return {
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "packages": dict(sorted((d.metadata["Name"], d.version)
                                 for d in importlib.metadata.distributions() if d.metadata["Name"])),
        "cpuinfo": read_optional("/proc/cpuinfo"),
        "cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
        "memory_max": read_optional("/sys/fs/cgroup/memory.max"),
        "base_image": BASE_IMAGE, "uv_version": UV_VERSION,
    }


def deterministic_fingerprints(out: Path) -> dict:
    """Numerical record and tensor bytes; exclude only timing from metrics.

    Torch archives contain serializer metadata, so checkpoint bit identity is
    defined on sorted parameter name, dtype, shape and contiguous tensor bytes.
    Full raw files are retained alongside these hashes.
    """
    answer = {}
    metrics = out / "metrics.jsonl"
    if metrics.exists():
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        for row in rows:
            row.pop("sps", None)
        answer["metrics_without_sps_sha256"] = canonical_sha(rows)
    summary = out / "summary.json"
    if summary.exists():
        answer["summary_sha256"] = canonical_sha(json.loads(summary.read_text()))
    checkpoint = out / "agent.pt"
    if checkpoint.exists():
        import torch
        tensors = torch.load(checkpoint, map_location="cpu", weights_only=True)
        digest = hashlib.sha256()
        for name, tensor in sorted(tensors.items()):
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        answer["checkpoint_tensors_sha256"] = digest.hexdigest()
    return answer


@app.function(image=image, volumes={"/outputs": volume},
              cpu=(CPU_LIMIT, CPU_LIMIT), memory=(RAM_LIMIT_GIB * 1024, RAM_LIMIT_GIB * 1024),
              timeout=CALL_DEADLINE_SECONDS, startup_timeout=STARTUP_TIMEOUT_SECONDS,
              retries=0, max_containers=MAX_CONCURRENT_CALLS, min_containers=0, buffer_containers=0,
              scaledown_window=2, single_use_containers=True)
def train_one(spec: dict, absolute_deadline: float, source_commit: str, tracked_run_id: str) -> dict:
    """Exactly one (configuration, task, seed, budget, replicate) input.

    Provider preemption restarts only this input. Every restart gets a new
    attempt folder; the absolute deadline is shared across all attempts.
    """
    spec_hash = canonical_sha(spec)
    run_key = spec["id"]
    if not run_key or PurePosixPath(run_key).name != run_key or run_key in {".", ".."}:
        raise ValueError("Run id must be a simple path component")
    volume.reload()
    run_dir = Path("/outputs/runs") / run_key
    previous = run_dir / "complete.json"
    if previous.exists():
        result = json.loads(previous.read_text())
        if result["spec_sha256"] != spec_hash:
            raise ValueError("Existing completed run has a different specification")
        return result
    remaining = absolute_deadline - time.time()
    if remaining <= 15:
        raise TimeoutError("No training starts after the shared call deadline")
    attempt = uuid.uuid4().hex
    out = run_dir / "attempts" / attempt
    out.mkdir(parents=True)
    started = time.monotonic()
    before_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    inventory = package_inventory()
    atomic_json(out / "environment.json", inventory)
    atomic_json(out / "spec.json", spec)
    atomic_json(out / "attempt.json", {
        "attempt": attempt, "started_utc": utc_now(), "spec_sha256": spec_hash,
        "source_commit": source_commit, "modal_function_call_id": modal.current_function_call_id(),
        "modal_input_id": modal.current_input_id(), "absolute_deadline": absolute_deadline,
    })
    volume.commit()
    command = [sys.executable, "-m", "stage1.run_training", "--spec", str(out / "spec.json"), "--out", str(out)]
    print(json.dumps({"event": "training_start", "spec": spec, "source_commit": source_commit,
                      "attempt": attempt, "packages": inventory["packages"]}), flush=True)
    returncode = None
    error = None
    try:
        with (out / "stdout.log").open("w") as log:
            child_env = dict(os.environ, ORX_COMMIT=source_commit, ORX_RUN_ID=tracked_run_id)
            child = subprocess.Popen(command, cwd="/workspace", stdout=log, stderr=subprocess.STDOUT,
                                     env=child_env, start_new_session=True)
            try:
                returncode = child.wait(timeout=max(1, absolute_deadline - time.time() - 10))
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise TimeoutError("Training stopped at shared wall deadline")
        if returncode:
            raise RuntimeError(f"Training subprocess exited {returncode}")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        after_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall = time.monotonic() - started
        usage = {
            "wall_seconds": wall,
            "training_core_seconds": after_cpu.ru_utime + after_cpu.ru_stime - before_cpu.ru_utime - before_cpu.ru_stime,
            "training_peak_rss_bytes": int(after_cpu.ru_maxrss * 1024),
            "container_memory_peak_bytes": read_optional("/sys/fs/cgroup/memory.peak"),
            "container_cpu_stat": read_optional("/sys/fs/cgroup/cpu.stat"),
            "compute_cost_estimate_usd": str(estimate_cost(wall)),
            "provider_invoice_cost_usd": None,
            "cost_basis": "Requested resource limits times measured function wall; excludes startup/teardown. "
                          "Controller guard separately includes whole-call wall and allowance.",
            "returncode": returncode, "error": error,
        }
        atomic_json(out / "usage.json", usage)
        volume.commit()
        print((out / "stdout.log").read_text(), flush=True)
    fingerprints = deterministic_fingerprints(out)
    atomic_json(out / "fingerprints.json", fingerprints)
    result = {"id": run_key, "spec_sha256": spec_hash, "source_commit": source_commit,
              "attempt": attempt, "usage": usage, "fingerprints": fingerprints,
              "files": {str(p.relative_to(Path('/outputs'))): sha256_file(p)
                        for p in sorted(out.rglob("*")) if p.is_file()}}
    atomic_json(previous, result)
    volume.commit()
    print(json.dumps({"event": "training_complete", **result}), flush=True)
    return result


def download_file(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_name(local_path.name + ".part")
    with temporary.open("wb") as stream:
        for chunk in volume.read_file(remote_path):
            stream.write(chunk)
    temporary.replace(local_path)


def persist_ledger(ledger: BudgetLedger) -> None:
    ledger.save()
    from stage1.continuity import publish_runtime
    publish_runtime(ledger.data, ledger.path.parent)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(ledger.path, "/budget-ledger.json")


def sync_result(result: dict, destination: Path) -> None:
    for remote_path, expected_hash in result["files"].items():
        path = PurePosixPath(remote_path)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "runs":
            raise ValueError("Invalid remote output path")
        local = destination.joinpath(*path.parts[1:])
        download_file(remote_path, local)
        if sha256_file(local) != expected_hash:
            raise ValueError("Downloaded result hash mismatch: " + remote_path)
    atomic_json(destination / result["id"] / "complete.json", result)


def billing_snapshot(destination: Path, start: datetime, ledger: BudgetLedger) -> None:
    """Keep provider evidence separate from estimates; partial hours absent."""
    try:
        rows = modal.Workspace.from_context().billing.report(
            start=start, resolution="h", tag_names=["project_id", "stage"])
        converted = [dataclasses.asdict(row) if dataclasses.is_dataclass(row) else row for row in rows]
        selected = [row for row in converted if row.get("tags", {}).get("project_id") == PROJECT_ID]
        amount = sum((Decimal(str(row.get("cost", "0"))) for row in selected), Decimal(0))
        ledger.data["provider_reported_complete_intervals_cost_usd"] = str(amount)
        ledger.save()
        atomic_json(destination, json.loads(json.dumps({
            "queried_utc": utc_now(), "coverage": "Complete hourly intervals only; recent costs may be absent",
            "rows": selected,
        }, default=str)))
    except Exception as exc:
        atomic_json(destination, {"queried_utc": utc_now(), "error": f"{type(exc).__name__}: {exc}",
                                  "provider_invoice_cost_usd": None})


def stop_active(active: dict, ledger: BudgetLedger) -> None:
    """Cancel known handles and retain every reservation after an uncertain stop."""
    for key, call in list(active.items()):
        error = None
        try:
            call.cancel(terminate_containers=True)
        except Exception as exc:
            error = f"; cancellation failed: {type(exc).__name__}: {exc}"
        ledger.uncertain(key, "Controller stopped before recording a completed result" + (error or ""))
        print(json.dumps({"event": "call_cancelled", "id": key, "error": error}), flush=True)
    persist_ledger(ledger)


def run_phase(specs: list[dict], ledger: BudgetLedger, results: Path, owner: str,
              billing_start: datetime, concurrency: int = MAX_CONCURRENT_CALLS) -> None:
    """Reserve before submission, poll independently, and never respawn a known id."""
    pending = []
    active = {}
    try:
        for spec in specs:
            key = spec["id"]
            old = ledger.data["calls"].get(key)
            if old is None:
                pending.append(spec)
                continue
            if old["spec_sha256"] != canonical_sha(spec):
                raise BudgetStop("Existing run specification differs: " + key)
            if old["status"] == "complete":
                destination = results / key / "complete.json"
                download_file(f"/runs/{key}/complete.json", destination)
                sync_result(json.loads(destination.read_text()), results)
            elif old["status"] in {"reserved", "running"}:
                if not old.get("function_call_id") or not old.get("absolute_deadline"):
                    ledger.uncertain(key, "Pending submission lacks a recoverable provider handle/deadline")
                    persist_ledger(ledger)
                    raise BudgetStop("Inspect provider state for interrupted submission: " + key)
                active[key] = modal.FunctionCall.from_id(old["function_call_id"])
            else:
                raise BudgetStop("Uncertain call requires provider billing reconciliation: " + key)
        last_heartbeat = 0.0
        while pending or active:
            blocked = None
            while pending and len(active) < concurrency:
                spec = pending[0]
                key = spec["id"]
                try:
                    reservation = ledger.reserve(key, canonical_sha(spec))
                except BudgetStop as exc:
                    blocked = exc
                    break
                # This durability boundary is BEFORE the paid invocation. A crash
                # between spawn and associate is deliberately fail-closed on resume.
                persist_ledger(ledger)
                try:
                    call = train_one.spawn(spec, reservation["absolute_deadline"],
                                           os.environ["ORX_COMMIT"], owner)
                except BaseException as exc:
                    ledger.uncertain(key, f"Submission outcome unknown: {type(exc).__name__}: {exc}")
                    persist_ledger(ledger)
                    raise
                active[key] = call
                ledger.associate(key, call.object_id, app.app_id)
                persist_ledger(ledger)
                pending.pop(0)
                print(json.dumps({"event": "submitted", "id": key, "function_call_id": call.object_id,
                                  "guard_total_usd": str(ledger.guard_total())}), flush=True)
            if not active:
                if blocked is not None:
                    raise blocked
                break
            completed = False
            for key, call in list(active.items()):
                record = ledger.data["calls"][key]
                try:
                    result = call.get(timeout=0)
                except TimeoutError:
                    if time.time() >= record["absolute_deadline"]:
                        raise BudgetStop("Shared call deadline reached: " + key)
                    continue
                # Charge complete submit-to-observed-result wall time, including
                # waiting, startup and any automatic preemption/retry attempts.
                ledger.complete(key, time.time() - record["submitted_unix_seconds"], result["usage"])
                del active[key]
                persist_ledger(ledger)
                sync_result(result, results)
                completed = True
                print(json.dumps({"event": "run_recorded", "id": key,
                                  "guard_total_usd": str(ledger.guard_total()), "result": result}), flush=True)
            if completed:
                billing_snapshot(results / "billing-latest.json", billing_start, ledger)
                persist_ledger(ledger)
                # Crossing the guard or learning of uncertain charges cancels all
                # remaining work. Ordinary full reservations only delay refills.
                ledger.assert_can_launch()
            if time.monotonic() - last_heartbeat >= 30:
                print(json.dumps({"event": "waiting", "active": list(active), "queued": len(pending),
                                  "guard_total_usd": str(ledger.guard_total())}), flush=True)
                last_heartbeat = time.monotonic()
            if active and not completed:
                time.sleep(5)
    except BaseException:
        stop_active(active, ledger)
        raise


def main() -> None:
    if not os.environ.get("ORX_RUN_ID") or not os.environ.get("ORX_COMMIT"):
        raise SystemExit("Use the tracked fixed project run command")
    from scripts.verify_lock import verify
    verify()
    manifest = json.loads((ROOT / "stage1/manifest.json").read_text())
    if manifest.get("stage") != 1 or manifest.get("no_evolution_or_random_search") is not True:
        raise SystemExit("This controller only executes Stage 1")
    specs = manifest["runs"]
    if len({s["id"] for s in specs}) != len(specs):
        raise ValueError("Duplicate run ids")
    for spec in specs:
        if spec["purpose"] not in {"validation", "determinism", "cross_platform", "cross_platform_corrected", "calibration"}:
            raise ValueError("Unauthorized Stage 1 purpose: " + spec["purpose"])
        key = spec["id"]
        if not key or PurePosixPath(key).name != key or key in {".", ".."}:
            raise ValueError("Run ids must be simple path components")
    if not any(s["purpose"] == "validation" for s in specs):
        raise ValueError("A successful Modal validation barrier is required before training")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    # A distributed atomic controller lock prevents concurrent budget ledgers.
    controller_lock = modal.Dict.from_name(VOLUME_NAME + "-controller", create_if_missing=True)
    owner = os.environ["ORX_RUN_ID"]
    if not controller_lock.put("owner", owner, skip_if_exists=True):
        raise BudgetStop("Another controller owns the Stage 1 budget; inspect it before resuming")
    ledger_path = results / "budget-ledger.json"
    ledger = None
    app_started = False
    try:
        try:
            download_file("/budget-ledger.json", ledger_path)
        except FileNotFoundError:
            pass
        ledger = BudgetLedger(ledger_path)
        ledger.assert_can_launch()
        unknown = set(ledger.data["calls"]) - {s["id"] for s in specs}
        if unknown:
            raise BudgetStop("Ledger contains runs outside the locked manifest: " + ", ".join(sorted(unknown)))
        persist_ledger(ledger)
        billing_start = datetime.fromisoformat(ledger.data["created_utc"])
        billing_snapshot(results / "billing-before.json", billing_start, ledger)
        ledger.assert_can_launch()
        ledger.start_setup()
        persist_ledger(ledger)
        with modal.enable_output(), app.run():
            app_started = True
            ledger.finish_setup()
            persist_ledger(ledger)
            # Phase barriers prevent scientific training before unit validation,
            # and finish the platform checks before the calibration grid.
            phases = ({"validation"}, {"determinism"},
                      {"cross_platform", "cross_platform_corrected"}, {"calibration"})
            for purposes in phases:
                run_phase([s for s in specs if s["purpose"] in purposes], ledger, results, owner,
                          billing_start, concurrency=1 if purposes == {"validation"} else MAX_CONCURRENT_CALLS)
        billing_snapshot(results / "billing-after.json", billing_start, ledger)
        persist_ledger(ledger)
        print(json.dumps({"event": "stage1_manifest_complete", "runs": len(specs),
                          "guard_total_usd": str(ledger.guard_total())}), flush=True)
    except BaseException as exc:
        if ledger is not None:
            if not app_started and not isinstance(exc, BudgetStop):
                ledger.setup_failed(f"{type(exc).__name__}: {exc}")
                persist_ledger(ledger)
            atomic_json(results / "controller-stop.json", {
                "utc": utc_now(), "error": f"{type(exc).__name__}: {exc}",
                "guard_total_usd": str(ledger.guard_total()),
                "invoice_cost_usd": None, "stage_2_started": False,
            })
        raise
    finally:
        # An uncertain call remains durable in the ledger, blocking launches.
        controller_lock.pop("owner")


if __name__ == "__main__":
    main()
