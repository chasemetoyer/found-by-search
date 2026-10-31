"""Shared Stage 2 constants and helpers; importing this module calls no provider."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from stage1.budget import atomic_json, utc_now  # noqa: F401  (re-exported)

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "64555583-12e3-4641-82a0-b8841b663e70"
APP_NAME = "evolved-intrinsic-rewards-stage2"
VOLUME_NAME = "evolved-intrinsic-rewards-stage2"
# Volumes v1 recommends at most five concurrent writers; v2 supports writes from
# hundreds of containers to distinct files (https://modal.com/docs/guide/volumes).
VOLUME_VERSION = 2
CONTROLLER_LOCK_NAME = VOLUME_NAME + "-controller"
SHARED = (ROOT / "controller-state" if os.environ.get("STAGE2_CLOUD_CONTROLLER") == "1"
          else Path("/Users/chasemetoyer/Desktop/evolved-intrinsic-rewards/handoff-cross-platform"))

# User-approved (2026-10-28). Modal's workspace-wide limit is 100 containers on
# Starter and 5000 on Team (https://modal.com/pricing); max_containers on the
# Function is the separate per-function cap.
MAX_CONCURRENT_CALLS = 32
FUNCTION_TIMEOUT_SECONDS = 3600
STARTUP_TIMEOUT_SECONDS = 120

# User-approved determinism test: cap all three kernel libraries at AVX2.
ISA_PINS = {"ATEN_CPU_CAPABILITY": "avx2", "MKL_CBWR": "AVX2", "ONEDNN_MAX_CPU_ISA": "AVX2"}


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cpuinfo(text: str | None) -> dict:
    """Vendor, family, model and kernel-relevant flags from a saved /proc/cpuinfo.

    gVisor reports "model name: unknown", so identity comes from these fields.
    """
    fields = {}
    for line in (text or "").split("\n\n")[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    flags = set(fields.get("flags", "").split())
    family = int(fields["cpu family"]) if fields.get("cpu family", "").isdigit() else None
    model = int(fields["model"]) if fields.get("model", "").isdigit() else None
    return {"vendor_id": fields.get("vendor_id"), "cpu_family": family, "model": model,
            "stepping": fields.get("stepping"), "logical_processors": (text or "").count("processor\t:"),
            **{flag: flag in flags for flag in ("avx2", "fma", "avx512f", "avx512bw", "avx512dq",
                                                 "avx512vl", "avx512_vnni", "avx512_bf16")}}


def cpu_class(info: dict) -> str:
    """Classes named in the user's test: AMD Zen 3, AMD Zen 4, any Intel; else other."""
    if info.get("vendor_id") == "AuthenticAMD" and info.get("cpu_family") == 0x19 and info.get("model") is not None:
        if 0x00 <= info["model"] <= 0x0F:
            return "zen3"
        if 0x10 <= info["model"] <= 0x1F:
            return "zen4"
    if info.get("vendor_id") == "GenuineIntel":
        return "intel"
    return "other"
