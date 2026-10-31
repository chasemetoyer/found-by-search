"""Stage 3 Modal app: one isolated Function input per run, at most 32 at once.

Same pinned image recipe as Stage 1 (base digest, uv version, locked install from
the public PyPI index, single-threaded math libraries), plus the stage3 package.
Outputs go to a Modal Volumes v2 volume, which supports many concurrent writers.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import uuid
from decimal import Decimal
from pathlib import Path, PurePosixPath

import modal

from stage1.budget import CPU_LIMIT, RAM_LIMIT_GIB, estimate_cost
from stage3.common import (
    APP_NAME, FUNCTION_TIMEOUT_SECONDS, MAX_CONCURRENT_CALLS, PROJECT_ID, ROOT, STARTUP_TIMEOUT_SECONDS,
    VOLUME_NAME, VOLUME_VERSION, atomic_json, canonical_sha, parse_cpuinfo, sha256_file, utc_now,
)

BASE_IMAGE = "python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"
UV_VERSION = "0.9.7"
EXCLUDED_PARTS = {".git", ".venv", ".agents", ".openresearch", ".orx", "__pycache__", ".pytest_cache", "results"}
MAX_INFRASTRUCTURE_RETRIES = 3


def excluded(path: Path) -> bool:
    path = Path(path)
    if any(p in EXCLUDED_PARTS for p in path.parts) or path.name.startswith(".env"):
        return True
    relative = path.resolve().relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()
    files = {".python-version", "generic_reward.py", "pyproject.toml", "uv.lock"}
    directories = ("stage1/", "stage2/", "stage3/", "preregistration/",
                   "handoff-cross-platform/code/final-round5/",
                   "handoff-cross-platform/code/a1b-run-commit-1533734/")
    return not (relative in files or any(f.startswith(relative.rstrip("/") + "/") for f in files)
                or relative.startswith(directories)
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
app = modal.App(APP_NAME, tags={"project_id": PROJECT_ID, "stage": "3"}, include_source=False)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=VOLUME_VERSION)


def read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def environment_record(spec: dict) -> dict:
    cpuinfo = read_optional("/proc/cpuinfo")
    return {
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "packages": dict(sorted((d.metadata["Name"], d.version)
                                for d in importlib.metadata.distributions() if d.metadata["Name"])),
        "cpuinfo": cpuinfo, "cpu": parse_cpuinfo(cpuinfo),
        "cpu_max": read_optional("/sys/fs/cgroup/cpu.max"), "memory_max": read_optional("/sys/fs/cgroup/memory.max"),
        "base_image": BASE_IMAGE, "uv_version": UV_VERSION, "isa_pins": spec.get("isa_pins") or {},
    }


def deterministic_fingerprints(out: Path) -> dict:
    """Stage 1's identity definition: metrics without SPS, summary, sorted tensor bytes."""
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
        import hashlib
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
              timeout=FUNCTION_TIMEOUT_SECONDS, startup_timeout=STARTUP_TIMEOUT_SECONDS,
              retries=modal.Retries(max_retries=MAX_INFRASTRUCTURE_RETRIES, initial_delay=1.0,
                                    backoff_coefficient=2.0, max_delay=4.0),
              max_containers=MAX_CONCURRENT_CALLS, min_containers=0, buffer_containers=0,
              scaledown_window=2, single_use_containers=True)
def train_one(spec: dict, absolute_deadline: float, source_commit: str, tracked_run_id: str) -> dict:
    """One Stage 3 input. A provider preemption restarts only this input in a new attempt folder."""
    from stage3.specs import validate
    validate(spec)
    spec_hash = canonical_sha(spec)
    run_key = spec["id"]
    if PurePosixPath(run_key).name != run_key or run_key in {"", ".", ".."}:
        raise ValueError("Run id must be a simple path component")
    volume.reload()
    run_dir = Path("/outputs/runs") / run_key
    previous = run_dir / "complete.json"
    if previous.exists():
        result = json.loads(previous.read_text())
        if result["spec_sha256"] != spec_hash:
            raise ValueError("Existing completed run has a different specification")
        return result
    if absolute_deadline - time.time() <= 15:
        raise TimeoutError("No training starts after the shared call deadline")
    attempt = uuid.uuid4().hex
    out = run_dir / "attempts" / attempt
    out.mkdir(parents=True)
    started = time.monotonic()
    before_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    atomic_json(out / "environment.json", environment_record(spec))
    atomic_json(out / "spec.json", spec)
    atomic_json(out / "attempt.json", {
        "attempt": attempt, "started_utc": utc_now(), "spec_sha256": spec_hash, "source_commit": source_commit,
        "modal_function_call_id": modal.current_function_call_id(), "modal_input_id": modal.current_input_id(),
        "absolute_deadline": absolute_deadline})
    volume.commit()
    child_env = dict(os.environ, ORX_COMMIT=source_commit, ORX_RUN_ID=tracked_run_id, **(spec.get("isa_pins") or {}))
    print(json.dumps({"event": "training_start", "id": run_key, "attempt": attempt, "source_commit": source_commit,
                      "cpu": environment_record(spec)["cpu"]}), flush=True)
    command = [sys.executable, "-m", "stage3.run_training", "--spec", str(out / "spec.json"), "--out", str(out)]
    returncode = error = None
    try:
        probe = subprocess.run([sys.executable, "-m", "stage2.isa_probe", "--out", str(out / "isa_probe.json")],
                               cwd="/workspace", env=child_env, capture_output=True, text=True,
                               timeout=max(1, min(600, absolute_deadline - time.time() - 15)))
        if probe.returncode:
            atomic_json(out / "isa_probe_error.json", {"returncode": probe.returncode, "stderr": probe.stderr[-4000:]})
            raise RuntimeError("ISA metadata probe failed; infrastructure retry required")
        with (out / "stdout.log").open("w") as log:
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
        from stage3.run_training import validated_outcome
        inner_exit = json.loads((out / "process_usage.json").read_text())["exit_code"]
        outcome = validated_outcome(out, spec, inner_exit)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        atomic_json(out / "infrastructure_failure.json", {
            "classification": "infrastructure", "error": error,
            "retry_policy": "At most 3 SDK retries of this same Modal input/call; never a replacement candidate.",
            "absolute_deadline": absolute_deadline})
        raise
    finally:
        after_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall = time.monotonic() - started
        usage = {"wall_seconds": wall,
                 "training_core_seconds": after_cpu.ru_utime + after_cpu.ru_stime
                                          - before_cpu.ru_utime - before_cpu.ru_stime,
                 "training_peak_rss_bytes": int(after_cpu.ru_maxrss * 1024),
                 "container_memory_peak_bytes": read_optional("/sys/fs/cgroup/memory.peak"),
                 "container_cpu_stat": read_optional("/sys/fs/cgroup/cpu.stat"),
                 "compute_cost_estimate_usd": str(estimate_cost(wall)), "provider_invoice_cost_usd": None,
                 "cost_basis": "Requested resource limits times measured function wall; excludes startup/teardown. "
                               "Controller guard separately includes whole-call wall and allowance.",
                 "returncode": returncode, "error": error}
        atomic_json(out / "usage.json", usage)
        volume.commit()
    fingerprints = (deterministic_fingerprints(out) if outcome["outcome"] == "success"
                    else {"numerical_failure_sha256": canonical_sha(outcome)})
    atomic_json(out / "fingerprints.json", fingerprints)
    # Preserve prior retry metadata and charges in the final logical result.
    # Hard preemptions may have no usage file; report that unknown explicitly.
    attempts = []
    for prior in sorted((run_dir / "attempts").iterdir()):
        metadata = json.loads((prior / "attempt.json").read_text()) if (prior / "attempt.json").exists() else {}
        prior_usage = json.loads((prior / "usage.json").read_text()) if (prior / "usage.json").exists() else None
        attempts.append({"attempt": prior.name, "metadata": metadata, "usage": prior_usage})
    known_usage = [a["usage"] for a in attempts if a["usage"] is not None]
    aggregate_usage = dict(usage,
        wall_seconds=sum(a["wall_seconds"] for a in known_usage),
        training_core_seconds=sum(a["training_core_seconds"] for a in known_usage),
        training_peak_rss_bytes=max(a["training_peak_rss_bytes"] for a in known_usage),
        compute_cost_estimate_usd=str(sum((Decimal(a["compute_cost_estimate_usd"]) for a in known_usage), Decimal(0))),
        aggregate_attempt_usage=True, attempt_count=len(attempts), attempts_missing_usage=len(attempts)-len(known_usage),
        cost_basis="Sum of retained attempt estimates; missing preemption usage is unknown. Controller guard covers the whole call.")
    retained_files = set(p for p in out.rglob("*") if p.is_file())
    for prior in (run_dir / "attempts").iterdir():
        retained_files.update(p for p in prior.iterdir() if p.is_file() and p.name in {
            "attempt.json", "usage.json", "environment.json", "process_usage.json", "outcome.json",
            "infrastructure_failure.json", "isa_probe.json", "isa_probe_error.json", "stdout.log"})
    result = {"id": run_key, "spec_sha256": spec_hash, "source_commit": source_commit, "attempt": attempt,
              "outcome": outcome["outcome"], "failure": outcome.get("failure"),
              "usage": aggregate_usage, "attempts": attempts, "fingerprints": fingerprints,
              "files": {str(p.relative_to(Path("/outputs"))): sha256_file(p) for p in sorted(retained_files)}}
    atomic_json(previous, result)
    volume.commit()
    print(json.dumps({"event": "training_complete", "id": run_key, "attempt": attempt,
                      "fingerprints": fingerprints, "usage": usage}), flush=True)
    return result
