"""Fail closed if any registered input was changed after its lock."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    lock_path = ROOT / "preregistration/stage1-lock.json"
    if not lock_path.exists():
        raise SystemExit("Stage 1 is not locked")
    lock = json.loads(lock_path.read_text())
    implementation = dict(lock.get("implementation_sha256", {}))
    for path in sorted((ROOT / "preregistration").glob("stage1-amendment-*.json")):
        amendment = json.loads(path.read_text())
        if not amendment.get("reason") or not amendment.get("timestamp_utc"):
            raise SystemExit(f"Amendment lacks a reason or timestamp: {path.name}")
        for name, expected in amendment.get("implementation_sha256", {}).items():
            if name in lock["sha256"] or name not in implementation:
                raise SystemExit(f"Implementation amendment cannot replace a protected or unregistered input: {name}")
            implementation[name] = expected
    for path, expected in {**lock["sha256"], **implementation}.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Registered input changed: {path}. Use a timestamped amendment.")
    receipt_path = ROOT / "preregistration/stage1-github-receipt.json"
    if not receipt_path.exists():
        raise SystemExit("Missing externally corroborated GitHub lock receipt")
    receipt = json.loads(receipt_path.read_text())
    commit = receipt["lock_commit"]
    # Execution snapshots contain committed files, not the Git object database.
    # The receipt itself is committed after an observed GitHub ref verification.
    if hashlib.sha256(lock_path.read_bytes()).hexdigest() != receipt["lock_manifest_sha256"]:
        raise SystemExit("Lock manifest differs from the GitHub-verified receipt")
    if (ROOT / ".git").exists():
        raw = subprocess.check_output(["git", "show", f"{commit}:preregistration/stage1-lock.json"], cwd=ROOT)
        if raw != lock_path.read_bytes():
            raise SystemExit("Committed lock differs from current lock")
    if receipt["github_ref_commit"] != commit:
        raise SystemExit("GitHub receipt does not attest the lock commit")
    print(f"LOCK_VERIFIED commit={commit} inputs={len(lock['sha256']) + len(implementation)}")
    return lock


if __name__ == "__main__":
    verify()
