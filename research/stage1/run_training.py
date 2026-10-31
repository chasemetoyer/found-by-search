"""Execute one locked Stage 1 validation or training entry on Modal."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import tomli_w

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(out):
    """Identity of learning results, excluding only performance timing and serialization."""
    import torch
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    for row in rows:
        row.pop("sps", None)
    summary = json.loads((out / "summary.json").read_text())
    metric_bytes = json.dumps(dict(metrics=rows, summary=summary), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    tensors = torch.load(out / "agent.pt", map_location="cpu", weights_only=True)
    model = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        a = tensor.detach().contiguous().numpy()
        model.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return dict(metrics_sha256=hashlib.sha256(metric_bytes).hexdigest(), tensors_sha256=model.hexdigest(),
                excluded_fields=["metrics.sps"], checkpoint_serialization_compared=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("MODAL_IS_REMOTE") and not os.environ.get("MODAL_TASK_ID"):
        raise SystemExit("Training must execute on Modal; local training is disabled.")
    spec = json.loads(args.spec.read_text())
    manifest = json.loads((ROOT / "stage1/manifest.json").read_text())
    if spec not in manifest["runs"]:
        raise SystemExit("Spec is not an exact locked Stage 1 manifest entry")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    cfg_path = out / "config.toml"
    cfg_path.write_bytes(tomli_w.dumps(spec["config"]).encode())
    source_paths = [ROOT / "handoff-cross-platform/code" / spec["source"] / name
                    for name in ("ppo.py", "intrinsic.py")]
    if spec.get("detector") == "corrected":
        source_paths += [ROOT / "generic_reward.py", ROOT / "stage1/corrected_ppo.py"]
    provenance = dict(spec=spec, commit=os.environ.get("ORX_COMMIT"), run_id=os.environ.get("ORX_RUN_ID"),
                      python=sys.version, platform=platform.platform(), processor=platform.processor(),
                      packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                      source_files={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in source_paths})
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print("EFFECTIVE_CONFIG " + json.dumps(provenance, sort_keys=True), flush=True)
    child_env = dict(os.environ)
    if spec["purpose"] == "validation":
        child_env["STAGE1_UNIT_OUTPUT"] = str(out / "unit-tests")
        command = [sys.executable, "-m", "pytest", "-q", "--junitxml", str(out / "validation.xml"), "tests"]
    else:
        if spec.get("detector") == "corrected":
            command = [sys.executable, "-m", "stage1.corrected_ppo", "--source", spec["source"]]
        elif spec["purpose"] == "cross_platform" and spec.get("detector") == "archived":
            command = [sys.executable, str(ROOT / "handoff-cross-platform/code" / spec["source"] / "ppo.py")]
        else:
            raise SystemExit("Only the archived-detector cross-platform exception may bypass corrected detection")
        command += ["--seed", str(spec["seed"]), "--config", str(cfg_path), "--out", str(out)]
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = subprocess.run(command, cwd=ROOT, env=child_env, check=False)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    usage = dict(wall_seconds=time.monotonic() - start,
                 process_core_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                 peak_rss_bytes=after.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                 exit_code=proc.returncode,
                 memory_measurement="child process peak RSS; billing uses requested/actual provider memory",
                 cost_source="controller/provider accounting; process CPU time is not billed allocation time")
    (out / "process_usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    if proc.returncode:
        raise SystemExit(proc.returncode)
    if spec["purpose"] == "validation":
        tree = ET.parse(out / "validation.xml")
        suites = tree.findall(".//testsuite")
        validation = {k: sum(int(s.get(k, 0)) for s in suites)
                      for k in ("tests", "failures", "errors", "skipped")}
        if validation["failures"] or validation["errors"] or validation["skipped"] or not validation["tests"]:
            raise SystemExit("Validation did not complete successfully")
        (out / "summary.json").write_text(json.dumps(dict(validation=validation), indent=2) + "\n")
        identity = {"validation_files_sha256": {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((out / "unit-tests").rglob("*")) if p.is_file()}}
    else:
        identity = fingerprint(out)
    (out / "fingerprint.json").write_text(json.dumps(identity, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps(dict(id=spec["id"], usage=usage, fingerprint=identity,
          result=json.loads((out / "summary.json").read_text())), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
