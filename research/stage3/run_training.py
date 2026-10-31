"""Execute one validated Stage 3 entry on Modal with the unchanged corrected trainer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import tomli_w

from stage1.run_training import fingerprint
from stage2.common import ISA_PINS
from stage3.common import ROOT, atomic_json, canonical_sha
from stage3.specs import validate

SOURCE_FILES = ("generic_reward.py", "stage1/corrected_ppo.py", "stage3/run_training.py",
                "stage3/specs.py", "stage3/common.py", "stage3/instrumentation.py")
NUMERICAL_FAILURE_EXIT = 65
OUTCOME_SCHEMA = "stage3-training-outcome-v1"


class TrainingNumericalFailure(Exception):
    """Explicit nonfinite evidence produces a terminal failure for any method; never area zero."""

    def __init__(self, reason: str, evidence: dict):
        super().__init__(reason)
        self.reason, self.evidence = reason, evidence


def nonfinite_exception_evidence(exc: Exception) -> dict | None:
    """Inspect a known numerical failure's actual tensors, never its text alone.

    The categorical distribution's validation can fail before a model checkpoint
    is written. Import errors, arbitrary ValueError/RuntimeError messages and
    finite-but-invalid tensors do not qualify. This runs only after an exception.
    """
    if not isinstance(exc, ValueError):
        return None
    import torch
    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if (frame.f_globals.get("__name__") == "torch.distributions.distribution"
                and frame.f_code.co_name == "__init__"
                and frame.f_locals.get("param") in {"logits", "probs"}):
            value = frame.f_locals.get("value")
            distribution = frame.f_locals.get("self")
            if (type(distribution) is torch.distributions.Categorical and isinstance(value, torch.Tensor)
                    and not bool(torch.isfinite(value).all())):
                return {"phase": "categorical_parameter_validation", "parameter": frame.f_locals["param"],
                        "shape": list(value.shape), "dtype": str(value.dtype),
                        "nonfinite_elements": int((~torch.isfinite(value)).sum().item()),
                        "exception_type": type(exc).__name__}
        traceback = traceback.tb_next
    return None


def check_finite_outputs(out: Path) -> None:
    """One post-run scan; no changed PPO arithmetic."""
    def scan(value, where):
        if isinstance(value, float) and not math.isfinite(value):
            raise TrainingNumericalFailure("nonfinite_numeric_output", {"phase": "post_run_output", "field": where})
        if isinstance(value, dict):
            for key, child in value.items():
                if key != "sps":
                    scan(child, where + "." + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan(child, f"{where}[{index}]")
    # Missing, corrupt, or structurally invalid files remain infrastructure
    # failures. Only explicit nonfinite numeric values receive the sentinel.
    for index, line in enumerate((out / "metrics.jsonl").read_text().splitlines(), start=1):
        scan(json.loads(line), f"metrics.jsonl:{index}")
    scan(json.loads((out / "summary.json").read_text()), "summary.json")
    import torch
    checkpoint = torch.load(out / "agent.pt", map_location="cpu", weights_only=True)
    for name, tensor in checkpoint.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Checkpoint entry is not a tensor: " + name)
        if not bool(torch.isfinite(tensor).all()):
            raise TrainingNumericalFailure("nonfinite_checkpoint_tensor", {
                "phase": "post_run_checkpoint", "tensor": name, "shape": list(tensor.shape),
                "nonfinite_elements": int((~torch.isfinite(tensor)).sum().item())})


def write_numerical_failure(out: Path, spec: dict, failure: TrainingNumericalFailure) -> dict:
    result = {"schema": OUTCOME_SCHEMA, "spec_sha256": canonical_sha(spec), "outcome": "numerical_failure",
              "failure": {"classification": "training_numerical", "reason": failure.reason,
                          "evidence": failure.evidence,
                          "interpretation": "Operational numerical-failure classification; causation is not inferred. Stage3 stops without zero imputation."}}
    atomic_json(out / "outcome.json", result)
    return result


def validated_outcome(out: Path, spec: dict, returncode: int) -> dict:
    """Require matching structured evidence; unknown failures are never scored zero."""
    outcome = json.loads((out / "outcome.json").read_text())
    if outcome.get("schema") != OUTCOME_SCHEMA or outcome.get("spec_sha256") != canonical_sha(spec):
        raise ValueError("Training outcome has an unknown schema or wrong specification")
    if outcome.get("outcome") == "success" and returncode == 0:
        return outcome
    failure = outcome.get("failure") or {}
    if (outcome.get("outcome") == "numerical_failure" and returncode == NUMERICAL_FAILURE_EXIT
            and failure.get("classification") == "training_numerical"
            and failure.get("reason") in {"nonfinite_categorical_parameter", "nonfinite_numeric_output", "nonfinite_checkpoint_tensor"}
            and isinstance(failure.get("evidence"), dict) and failure["evidence"]):
        return outcome
    raise RuntimeError("Training failure lacks a recognized terminal numerical outcome")


def trainer_child(spec: dict, out: Path, cfg_path: Path) -> None:
    from stage3.instrumentation import load_ppo
    trainer = load_ppo(source=spec["source"])
    sys.argv = [sys.argv[0], "--seed", str(spec["seed"]), "--config", str(cfg_path), "--out", str(out)]
    try:
        try:
            trainer.main()
        except Exception as exc:
            evidence = nonfinite_exception_evidence(exc)
            if evidence is None:
                raise
            raise TrainingNumericalFailure("nonfinite_categorical_parameter", evidence) from exc
        check_finite_outputs(out)
    except TrainingNumericalFailure as exc:
        write_numerical_failure(out, spec, exc)
        raise SystemExit(NUMERICAL_FAILURE_EXIT) from exc
    atomic_json(out / "outcome.json", {"schema": OUTCOME_SCHEMA, "spec_sha256": canonical_sha(spec), "outcome": "success"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trainer-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("MODAL_IS_REMOTE") and not os.environ.get("MODAL_TASK_ID"):
        raise SystemExit("Training must execute on Modal; local training is disabled.")
    spec = json.loads(args.spec.read_text())
    validate(spec)
    pins = spec.get("isa_pins") or {}
    observed = {name: os.environ.get(name) for name in ISA_PINS}
    if pins and any(observed[name] != value for name, value in pins.items()):
        raise SystemExit("ISA pins requested by the spec are not set in the training environment")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.toml"
    cfg_path.write_bytes(tomli_w.dumps(spec["config"]).encode())
    if args.trainer_child:
        trainer_child(spec, out, cfg_path)
        return
    paths = [ROOT / "handoff-cross-platform/code" / spec["source"] / name for name in ("ppo.py", "intrinsic.py")]
    paths += [ROOT / name for name in SOURCE_FILES]
    provenance = dict(spec=spec, commit=os.environ.get("ORX_COMMIT"), run_id=os.environ.get("ORX_RUN_ID"),
                      python=sys.version, platform=platform.platform(), isa_environment=observed,
                      packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                      source_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in paths})
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print("EFFECTIVE_CONFIG " + json.dumps(provenance, sort_keys=True), flush=True)
    command = [sys.executable, "-m", "stage3.run_training", "--trainer-child",
               "--spec", str(args.spec), "--out", str(out)]
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = subprocess.run(command, cwd=ROOT, env=dict(os.environ), check=False)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    usage = dict(wall_seconds=time.monotonic() - start,
                 process_core_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                 peak_rss_bytes=after.ru_maxrss * 1024, exit_code=proc.returncode,
                 memory_measurement="child process peak RSS; billing uses requested/actual provider memory",
                 cost_source="controller/provider accounting; process CPU time is not billed allocation time")
    (out / "process_usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    if proc.returncode not in (0, NUMERICAL_FAILURE_EXIT):
        raise SystemExit(proc.returncode)
    outcome = validated_outcome(out, spec, proc.returncode)
    if outcome["outcome"] == "numerical_failure":
        print("RUN_SUMMARY " + json.dumps(dict(id=spec["id"], usage=usage, **outcome), sort_keys=True), flush=True)
        return  # Structured terminal result; the Modal worker must not retry it.
    identity = fingerprint(out)
    (out / "fingerprint.json").write_text(json.dumps(identity, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps(dict(id=spec["id"], usage=usage, fingerprint=identity,
                                           result=json.loads((out / "summary.json").read_text())),
                                      sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
