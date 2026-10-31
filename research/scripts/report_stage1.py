"""Generate the Stage 1 report and its machine-readable numbers from records.

Run after the preregistration lock and the evidence collection. Missing evidence
remains pending; partial cells cannot select a task. No training is invoked.
An independent check_numbers script must verify the generated handoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage1.score import aggregate_calibration, load_completed_scores, score_seed

ARTIFACTS = Path("/Users/chasemetoyer/.local/share/openresearch/files/evolved-intrinsic-rewards/stage1")
# Metadata of attempts that exist only on the Modal volume (for example a
# preempted attempt), copied with hashes by scripts/run_registry.py.
VOLUME_ATTEMPTS = Path("/Users/chasemetoyer/Desktop/evolved-intrinsic-rewards/handoff-cross-platform/stage1-container-cpu/volume-attempts")


def read_optional(path):
    return json.loads(path.read_text()) if path.exists() else None


def number(value, digits=4):
    if value is None:
        return "pending"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(value).replace("|", "/") for value in row) + " |" for row in rows])


def validation_records(results, manifest):
    records = []
    for spec in manifest["runs"]:
        if spec["purpose"] != "validation":
            continue
        markers = [p for p in results.rglob("complete.json") if p.parent.name == spec["id"]]
        if len(markers) > 1:
            raise ValueError("Duplicate validation completion records")
        record = {"id": spec["id"], "complete": False, "tests": None, "failures": None,
                  "errors": None, "skipped": None, "passed": None, "unit_data": {},
                  "testcase_nodes": None, "passed_cases": None, "passed_subtests": None}
        if markers:
            complete = read_optional(markers[0])
            out = markers[0].parent / "attempts" / complete["attempt"]
            xml = out / "validation.xml"
            if xml.exists():
                tree = ET.parse(xml).getroot()
                suites = list(tree.iter("testsuite"))
                totals = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                          for key in ("tests", "failures", "errors", "skipped")}
                record.update(complete=True, **totals, passed=totals["tests"] > 0 and totals["failures"] == totals["errors"] == totals["skipped"] == 0)
                record["testcase_nodes"] = len(list(tree.iter("testcase")))
                stdout = (out / "stdout.log").read_text() if (out / "stdout.log").exists() else ""
                counts = re.search(r"(\d+) passed, (\d+) subtests passed", stdout)
                if counts:
                    record["passed_cases"], record["passed_subtests"] = map(int, counts.groups())
                record["passing_targeted_tests"] = [case.attrib["name"] for case in tree.iter("testcase")
                                                    if not any(child.tag in ("failure", "error", "skipped") for child in case)]
            record["directory"] = str(out)
            for path in sorted((out / "unit-tests").glob("*.json")):
                record["unit_data"][path.name] = read_optional(path)
        records.append(record)
    return records


def comparison_records(manifest, scores):
    result = []
    reference = read_optional(ROOT / "handoff-cross-platform/scored-results/A1b-scored.json")
    stored = {row["seed"]: row for row in reference["seeds"]}
    for seed in range(40, 45):
        directory = ROOT / "handoff-cross-platform/scored-results/a1b-run-3629f9a7" / f"seed{seed}"
        mac = score_seed(directory, 2_000_000)
        assert math.isclose(mac["curve_area"], stored[seed]["train_auc"], abs_tol=1e-12)
        assert mac["solved"] == (stored[seed]["category"] == "solve")
        row = {"seed": seed, "mac": mac, "archived_modal": None, "corrected_modal": None}
        for purpose, key in (("cross_platform", "archived_modal"), ("cross_platform_corrected", "corrected_modal")):
            matches = [spec for spec in manifest["runs"] if spec["purpose"] == purpose and spec["seed"] == seed]
            if len(matches) != 1:
                raise ValueError(f"Expected one descriptive reference run per seed: {purpose}/{seed}")
            row[key] = scores.get(matches[0]["id"])
        result.append(row)
    aggregates = {}
    for key in ("mac", "archived_modal", "corrected_modal"):
        observed = [row[key] for row in result if row[key] is not None]
        aggregates[key] = {"n": len(observed), "complete": len(observed) == 5,
                           "solves": sum(row["solved"] for row in observed),
                           "mean_area": mean(row["curve_area"] for row in observed) if observed else None}
    return {"seeds": result, "aggregates": aggregates, "inferential_tests": False,
            "interpretation": "Descriptive counts and curve areas only; no cross-platform inference or adjustment."}


def determinism_records(manifest, scores):
    groups = {}
    for spec in manifest["runs"]:
        if spec["purpose"] == "determinism":
            groups.setdefault((spec["task"], spec["method"], spec["seed"], spec["total_timesteps"]), []).append(spec)
    records = []
    for (task, method, seed, budget), specs in sorted(groups.items()):
        completed = [scores[spec["id"]] for spec in sorted(specs, key=lambda s: s["replicate"]) if spec["id"] in scores]
        fingerprints = [score["fingerprints"] for score in completed]
        finished = len(specs) == len(completed) == 2 and all(bool(value) for value in fingerprints)
        records.append(dict(task=task, method=method, seed=seed, budget=budget, completed=len(completed),
                            bit_identical=(fingerprints[0] == fingerprints[1]) if finished else None,
                            fingerprints=fingerprints,
                            definition="Learning metrics except sps, summary, and ordered parameter tensor bytes; timing and serializer metadata excluded."))
    return records


def microarchitecture(vendor, family, model, flags):
    """Range-based CPUID labels; anything not checked against a reference stays unmapped."""
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


def short_cpu(label):
    if label.startswith("AMD Zen 3"):
        return "AMD Zen 3 (Milan range)"
    if label.startswith("AMD Zen 4"):
        return "AMD Zen 4 (Genoa range)"
    if "model 55h" in label:
        return "Intel 6/55h, Cascade Lake class" if "Cascade Lake class" in label else "Intel 6/55h (Skylake-SP family)"
    return label


def cpu_fields(cpuinfo):
    """CPU identity from a worker's saved /proc/cpuinfo; gVisor hides the model name."""
    fields = {}
    for line in (cpuinfo or "").split("\n\n")[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    flags = set(fields.get("flags", "").split())
    family = int(fields["cpu family"]) if fields.get("cpu family", "").isdigit() else None
    model = int(fields["model"]) if fields.get("model", "").isdigit() else None
    # PyTorch's ATen dispatch selects AVX512 kernels only with avx512vl/bw/dq and FMA3.
    if {"avx512vl", "avx512bw", "avx512dq", "fma"} <= flags:
        capability = "AVX512"
    elif {"avx2", "fma"} <= flags:
        capability = "AVX2"
    else:
        capability = "DEFAULT"
    label = microarchitecture(fields.get("vendor_id"), family, model, flags)
    return {"vendor_id": fields.get("vendor_id"), "cpu_family": family, "model": model,
            "avx512f": "avx512f" in flags, "inferred_aten_cpu_capability": capability,
            "microarchitecture": label, "cpu_short": short_cpu(label)}


def container_records(results, manifest, volume_attempts):
    """One row per Modal container attempt: synced outputs plus volume-only copies."""
    purposes = {spec["id"]: spec["purpose"] for spec in manifest["runs"]}
    found = {}
    for environment in sorted(results.glob("*/attempts/*/environment.json")):
        found[environment.parent.name] = (environment.parents[2].name, environment.parent, "synced-results")
    if volume_attempts.exists():
        for environment in sorted(volume_attempts.glob("*/*/environment.json")):
            found.setdefault(environment.parent.name, (environment.parent.parent.name, environment.parent, "modal-volume-copy"))
    records = []
    for attempt, (key, directory, source) in found.items():
        if key not in purposes:
            continue
        raw = (directory / "environment.json").read_bytes()
        meta = read_optional(directory / "attempt.json") or {}
        usage = read_optional(directory / "usage.json") or {}
        complete = read_optional(results / key / "complete.json")
        role = ("final" if complete["attempt"] == attempt else "superseded") if complete else "in progress or superseded"
        records.append({"run_key": key, "purpose": purposes[key], "attempt": attempt, "role": role, "source": source,
                        "environment_path": str(directory / "environment.json"),
                        "environment_sha256": hashlib.sha256(raw).hexdigest(),
                        "started_utc": meta.get("started_utc"), "function_call_id": meta.get("modal_function_call_id"),
                        "wall_seconds": usage.get("wall_seconds"), "attempt_error": usage.get("error"),
                        **cpu_fields(json.loads(raw).get("cpuinfo"))})
    return sorted(records, key=lambda r: (r["started_utc"] or "", r["attempt"]))


def determinism_explanations(manifest, scores, containers, determinism):
    """Where same-seed replicates first diverge, and which CPU type each ran on."""
    final = {c["run_key"]: c for c in containers if c["role"] == "final"}
    explanations = []
    for record in determinism:
        specs = sorted((s for s in manifest["runs"] if s["purpose"] == "determinism" and s["seed"] == record["seed"]
                        and s["task"] == record["task"] and s["method"] == record["method"]),
                       key=lambda s: s["replicate"])
        if len(specs) != 2 or any(s["id"] not in scores for s in specs):
            explanations.append({"seed": record["seed"], "complete": False})
            continue
        directories = [Path(scores[s["id"]]["directory"]) for s in specs]
        rows = [[json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines() if line.strip()]
                for d in directories]
        strip = lambda row: {k: v for k, v in row.items() if k != "sps"}
        differing = [i for i, (a, b) in enumerate(zip(*rows)) if strip(a) != strip(b)]
        first = None
        if differing:
            a, b = rows[0][differing[0]], rows[1][differing[0]]
            fields = sorted(k for k in strip(a) if a.get(k) != b.get(k))
            first = {"row": differing[0], "global_step": a["global_step"], "fields": fields,
                     "relative_differences": {k: abs(a[k] - b[k]) / max(abs(a[k]), abs(b[k])) for k in fields
                                              if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float))
                                              and max(abs(a[k]), abs(b[k])) > 0}}
        rollout = [i for i, (a, b) in enumerate(zip(*rows)) if any(a.get(k) != b.get(k) for k in a if k.startswith("train/"))]
        environments = [json.loads((d / "environment.json").read_text()) for d in directories]
        recorded = [json.loads((d / "spec.json").read_text()) for d in directories]
        explanations.append({
            "seed": record["seed"], "task": record["task"], "method": record["method"], "budget": record["budget"],
            "complete": True, "bit_identical": record["bit_identical"], "logged_rows": [len(r) for r in rows],
            "differing_rows": len(differing), "first_difference": first,
            "first_rollout_difference": {"row": rollout[0], "global_step": rows[0][rollout[0]]["global_step"]} if rollout else None,
            "final_rows": [{k: r[-1].get(k) for k in ("global_step", "train/success", "train/ret", "train/len")} for r in rows],
            "packages_identical": environments[0]["packages"] == environments[1]["packages"],
            "spec_fields_differing": sorted(k for k in recorded[0] if recorded[0][k] != recorded[1].get(k)),
            "replicates": [{"replicate": s["replicate"], "run_key": s["id"],
                            **{k: final.get(s["id"], {}).get(k) for k in ("attempt", "vendor_id", "cpu_family", "model",
                               "microarchitecture", "cpu_short", "avx512f", "inferred_aten_cpu_capability")}}
                           for s in specs]})
    return explanations


def cost_projection(calibration):
    selections = calibration["selection"]
    complete = all(row["status"] != "incomplete" for row in selections)
    training = [row["task"] for row in selections if row["role"] == "training" and row["kept"]]
    kept = [row["task"] for row in selections if row["kept"]]
    per_task = {}
    for task in kept:
        per_task[task] = {}
        for budget in (1_000_000, 2_000_000):
            observations = [score["cost_estimate_usd"] for cell in calibration["cells"]
                            if cell["task"] == task and cell["budget"] == budget
                            for score in cell["seeds"] if score.get("cost_estimate_usd") is not None]
            per_task[task][str(budget)] = {"n": len(observations), "mean_cost_estimate_usd": mean(observations) if observations else None}
    answer = {"selection_complete": complete, "training_tasks": training, "all_kept_tasks": kept,
              "basis": "Measured function wall time times pinned requested CPU/RAM prices; provider invoice is separate. Baseline costs proxy unrun candidate costs.",
              "per_task": per_task, "stage2": [], "stage3": [], "tamper_extra": [],
              "excluded": "Candidate re-evaluation, storage/egress, image builds, failed calls, and billing changes need additional allowance."}
    if not complete:
        return answer
    for budget in (1_000_000, 2_000_000):
        estimates = [per_task[task][str(budget)]["mean_cost_estimate_usd"] for task in training]
        available = all(value is not None for value in estimates)
        answer["stage2"].append({"fitness_budget": budget, "candidate_evaluations_per_method": 10 * 12,
             "candidate_evaluations_both_methods": 2 * 10 * 12, "seeds_per_candidate_task": 3,
             "training_runs": 2 * 10 * 12 * 3 * len(training),
             "cost_estimate_usd": 2 * 10 * 12 * 3 * sum(estimates) if available else None})
    estimates = [per_task[task]["2000000"]["mean_cost_estimate_usd"] for task in kept]
    unit = sum(estimates) if all(value is not None for value in estimates) else None
    for methods in (3, 4, 5):
        answer["stage3"].append({"method_count_scenario": methods, "seeds_per_method_task": 20,
             "training_runs": methods * 20 * len(kept), "cost_estimate_usd": methods * 20 * unit if unit is not None else None})
        for extra_conditions in (1, 2):
            answer["tamper_extra"].append({"method_count_scenario": methods, "additional_training_conditions": extra_conditions,
                "additional_runs": extra_conditions * methods * 20 * len(kept),
                "additional_cost_estimate_usd": extra_conditions * methods * 20 * unit if unit is not None else None})
    return answer


def draft_outline():
    return """# Stage 2 pre-registration outline — DRAFT, NOT LOCKED

No evolutionary or random-search evaluation has been run. Every proposal below
needs approval at the Stage 1 boundary before it becomes a Stage 2 commitment.

- Genome: four events, each with weight in [-1,1], relief in [0,1], and growth either zero or log-scaled in [1/640,1/20]; h0=.5 fixed. Preserve the corrected generic event definitions approved for Stage 1, including exclusion of success-terminal target pickup. Twelve physical parameters. A proposed latent representation uses direct bounded weight/relief coordinates and a growth coordinate with a zero atom plus a positive logarithmic interval. OPEN: zero-atom probability/threshold, transformation, clipping/reflection at CMA boundaries, initialization and sigma.
- Event semantics — four separate OPEN decisions for the Stage 2 lock: retain final-step event inclusion; retain box-opening inclusion; retain successful target-pickup exclusion; retain attachment only after the initial seeded reset. These are fixed for the present Stage 1 batch and must not be silently treated as locked Stage 2 choices. Also confirm exact-view byte novelty, no preinserted reset view, and state-change rather than attempted-action events.
- Fitness proposal: equal-weight mean training-success curve area over the kept training tasks and three seeds per candidate at the chosen fitness budget. No task-specific fitted formula and no held-out-task feedback. OPEN: confirm the budget, aggregation, treatment of failed/time-limited evaluations, and safeguards for repeated positively paid events.
- CMA-ES: population ten, twelve generations. OPEN: optimizer/library version, seed and initial mean/covariance/sigma; stopping behavior under numerical failure; number of finalists and independent re-evaluation seeds/budget; winner tie-break and whether to reserve candidate evaluations for re-evaluation.
- Random-search control: the same number of candidate evaluations, task set, training budgets and seed counts. OPEN: prior over mixed zero/log growth, matched handling of boundary atoms, winner re-evaluation, and whether random-search winner joins final evaluation. No search starts in Stage 1.
- Seeds: calibration 900–999, evolution 1000–1999, random search 3000–3999, final evaluation 2000–2999. OPEN: exact within-role allocation, common versus fresh candidate seed panels, independent finalist panel, and layout seed mapping. Never reuse seeds across roles.
- Cross-CPU numerical reproducibility — OPEN decision for the Stage 2 lock: test whether forcing ATEN_CPU_CAPABILITY=avx2 and MKL_CBWR=AVX2 makes seed 910 bit-identical across Zen 3 and Zen 4 containers. If it does, use these settings in Stage 2. If it does not, record the CPU type for every run and treat same-seed results as unpaired. This test was not run in Stage 1. Note for the test design: PyTorch normally runs batched float32 CPU convolutions through oneDNN, whose kernel instruction set is controlled separately (ONEDNN_MAX_CPU_ISA) and is not set by these two variables.
- Final evaluation: best evolved reward and approved baselines, twenty fresh seeds per kept task, nominal two-million-step training, fixed per-environment evaluation quota. OPEN: baseline roster (task-only, episodic novelty, random-search winner, fixed satiation); whether task aggregation or individual tasks are primary; independent versus paired seed comparisons; minimum worthwhile effect; multiplicity family. Proposed tests retain exact Fisher for solves and exact MWU without ties/permutation MWU with ties for area, Clopper–Pearson intervals, and Holm adjustment. A superiority hit should require the registered direction and adjusted p<.05; numerical level predictions need prewritten intervals. Non-significance does not establish equivalence.
- Tamper test: introduce the done-action need reset only after search. OPEN: freeze policy versus retrain/fine-tune a learner under the selected reward; matched no-tamper condition; free versus paid presses and costs; tasks/methods; budget/seeds; performance noninferiority margin and test; whether press frequency is a separate endpoint. Keep presses and repeated paid events descriptive unless explicit hit rules are locked. Resistance requires a defined performance criterion; bounded return alone is insufficient.
- Budget: approve measured cost projection, re-evaluation overhead, and a hard spending limit. Make the repository public before the Stage 2 lock, commit and push every lock file, and record timestamped amendments. No Stage 3 lock is implied by this draft.
"""


def determinism_lines(data):
    explanations = [e for e in data["determinism_explanation"] if e.get("complete")]
    if not explanations:
        return ["Determinism CPU attribution: pending.", ""]
    lines = ["### Determinism outcome and container CPU types", "",
             table(["Seed", "Replicate", "Attempt", "CPU type", "AVX-512", "Inferred ATen tier", "Pair bit identical"],
                   [(e["seed"], r["replicate"], (r["attempt"] or "pending")[:8], r["cpu_short"] or "pending",
                     number(r["avx512f"]), r["inferred_aten_cpu_capability"] or "pending", number(e["bit_identical"]))
                    for e in explanations for r in e["replicates"]]), ""]
    for e in explanations:
        if e["bit_identical"] is not False:
            continue
        first, rollout, final = e["first_difference"], e["first_rollout_difference"], e["final_rows"]
        relative = ", ".join(f"{k} {v:.1e}" for k, v in sorted(first["relative_differences"].items()))
        text = (f"Seed {e['seed']}: both calls used the same source commit; installed package inventories identical: "
                f"{number(e['packages_identical'])}; the specifications differ only in {', '.join(e['spec_fields_differing'])}. "
                f"The logged metrics first differ at row {first['row']} (step {first['global_step']}), in "
                f"{', '.join(first['fields'])} only (relative differences {relative}); every rollout statistic in that row is identical. ")
        if rollout:
            text += f"Episode statistics first differ at row {rollout['row']} (step {rollout['global_step']}). "
        text += (f"{e['differing_rows']} of {min(e['logged_rows'])} logged rows differ. At the last row (step "
                 f"{final[0]['global_step']}), training success is {final[0]['train/success']} versus "
                 f"{final[1]['train/success']} and mean training return {final[0]['train/ret']:.4f} versus {final[1]['train/ret']:.4f}.")
        lines += [text, ""]
    mismatched = [e for e in explanations if e["bit_identical"] is False]
    matched = [e for e in explanations if e["bit_identical"] is True]
    cross = lambda e: len({(r["microarchitecture"], r["inferred_aten_cpu_capability"]) for r in e["replicates"]}) > 1
    if mismatched and all(cross(e) for e in mismatched) and not any(cross(e) for e in matched):
        lines += [f"Interpretation (consistent with the records, not a controlled test): every mismatched pair ran on two different CPU types and every matching pair ran on one CPU type. "
                  "A difference that starts in the last bits of optimizer statistics while episodes are still identical is the signature of floating-point rounding, not of a seeding or logic difference. "
                  "The matching pair also shows that the replicate label does not enter the computation. "
                  "The pattern is consistent with PyTorch and its math libraries choosing CPU-specific vector kernels: different kernels sum float32 values in a different order, and PPO's sampled actions then amplify the last-bit differences into different trajectories. "
                  f"The evidence is {len(explanations)} planned pairs, and no execution setting was changed in response. "
                  "Modal assigns containers from a mixed CPU pool (next section), so separate calls with the same seed are not guaranteed to be bit identical. The open Stage 2 decision on CPU instruction-set pinning addresses this.", ""]
    elif mismatched:
        lines += ["Interpretation: the container CPU types do not account for every mismatch; the cause remains unexplained.", ""]
    return lines


def container_lines(data):
    containers = data["containers"]
    summary = {}
    for c in containers:
        row = summary.setdefault(c["cpu_short"], {"avx512f": c["avx512f"], "tier": c["inferred_aten_cpu_capability"],
                                                  "final": 0, "superseded": 0, "other": 0})
        row["final" if c["role"] == "final" else "superseded" if c["role"] == "superseded" else "other"] += 1
    superseded = [c for c in containers if c["role"] == "superseded"]
    lines = ["## Modal containers and CPU types", "",
             "Every worker attempt saved /proc/cpuinfo. Modal's gVisor sandbox reports the model name as unknown, so the CPU type is identified from vendor, family, model and instruction-set flags, using range-based CPUID labels. The ATen tier is inferred from the flags with PyTorch's dispatch rule (AVX512 requires avx512vl, avx512bw, avx512dq and FMA3); the workers did not record PyTorch's own capability string. Attempts superseded by a provider preemption exist only on the Modal volume; their metadata was copied with hashes into the shared handoff folder.", "",
             table(["CPU type", "AVX-512", "Inferred ATen tier", "Final attempts", "Superseded attempts", "Unfinished"],
                   [(name, number(v["avx512f"]), v["tier"], v["final"], v["superseded"], v["other"])
                    for name, v in sorted(summary.items())]), ""]
    if superseded:
        lines += ["Superseded attempts. Modal restarted the same input under the same call ID, and the controller charged the whole call, including the lost attempt:", "",
                  table(["Run", "Superseded attempt", "CPU type", "Wall seconds before stop", "Recorded error"],
                        [(c["run_key"], c["attempt"][:8], c["cpu_short"], number(c["wall_seconds"], 1),
                          c["attempt_error"] or "none recorded") for c in superseded]), ""]
    lines += ["The per-container list is in the appendix.", ""]
    return lines


def container_appendix(data):
    return ["## Appendix: CPU type of every Modal container", "",
            table(["Run", "Attempt", "Role", "CPU type", "Vendor / family / model", "AVX-512", "Inferred ATen tier", "Source"],
                  [(c["run_key"], c["attempt"][:8], c["role"], c["cpu_short"], f"{c['vendor_id']} / {c['cpu_family']} / {c['model']}",
                    number(c["avx512f"]), c["inferred_aten_cpu_capability"], c["source"]) for c in data["containers"]]), ""]


def render(data):
    cal = data["calibration"]
    lines = ["# Stage 1 report", "", "Status: " + ("complete" if data["complete"] else "incomplete; unavailable evidence is marked pending") + ". Stage 2 has not started.", "",
        "All empirical numbers below are generated from retained outputs. `stage1-numbers.json` contains the full precision values and provenance. The independent number-check result is required for final handoff.", "",
        "## Carryover verification", "",
        table(["Item", "Location", "Verification / disposition"], [
            ("A1 PPO / CNN", "handoff-cross-platform/code/*/ppo.py; stage1/corrected_ppo.py", "Archived source retained; corrected adapter keeps learner/CNN and Table 10 settings."),
            ("A2 wrappers / modes", "generic_reward.py; stage1/corrected_ppo.py; archived intrinsic.py", "Corrected reset/terminal capture; legacy locked_door_key remains available; validation records below."),
            ("A3 satiation / tamper", "generic_reward.py; archived intrinsic.py", "Need reset and extrinsic press cost covered by validation."),
            ("A4 statistical procedures", "stage1/statistics.py; tests/test_scoring.py", "Archived procedures reused; known statistical examples checked in validation."),
            ("A5 quota evaluation", "archived ppo.py:evaluate", "Per-environment episode quota preserved."),
            ("A6 number checking", "scripts/report_stage1.py; scripts/check_numbers.py", "Generated JSON and report; independent cross-check required."),
            ("B1 generic events", "generic_reward.py", "Four generic events; successful terminal target pickup excluded under the approved correction."),
            ("B2 refill / farming / tamper", "generic_reward.py; stage1/manifest.json", "Growth bounds enforced; repeated paid events retained; tamper experiment deferred."),
            ("B3 advantage normalization", "stage1/manifest.json", "Enabled in every Stage 1 arm."),
            ("B4 easy-task calibration", "stage1/score.py", "Fixed room-to-improve rule applied without tuning."),
            ("B5 primary metrics", "stage1/score.py", "Curve area and solve count primary; behavior labels descriptive."),
            ("C1 immutable lock", "preregistration/stage1-lock.json; scripts/verify_lock.py", "Input hashes checked before runs; later changes need timestamped amendments."),
            ("C2 corroborated lock", "preregistration/stage1-github-receipt.json", "Committed lock and GitHub receipt checked before launch."),
            ("C3 seed roles", "stage1/manifest.json", "Calibration roles separated; paper seeds restricted to descriptive reproduction."),
            ("C4 same platform / pins", "stage1/modal_app.py; pyproject.toml; uv.lock", "All new evidence uses Modal; per-attempt complete installed package inventory retained."),
            ("C5 fixed scoring", "stage1/score.py; preregistration", "No formulas fitted to outcomes; failures and unavailable cells retained."),
            ("C6 credentials", ".gitignore; environment / Modal configuration", "Credential ignore rules established before source publication."),
            ("C7 stage boundary", "this report and draft outline", "Stop for approval before Stage 2."),
        ]), "", "## Corrected detector definitions", "",
        "The new project uses corrected detection throughout. The archived-detector Modal reproduction is the explicitly authorized exception. Physical events are captured before SAME_STEP autoreset, and new_view uses the true final observation on a final step.", "",
        table(["Event", "Exact Stage 1 definition"], [
            ("new_view", "Exact bytes of the symbolic 7x7 post-action observation not yet seen in this episode. The reset view is not preinserted; clear seen views after processing a terminal/truncated transition."),
            ("pickup", "Carrying changes from nothing to any object, except a pickup that itself terminates the task with positive raw task reward. Excluded successful target pickups are recorded separately. A timeout alone does not exclude a pickup."),
            ("drop", "An executed drop action changes carrying from an object to nothing. Blocked attempts do not count."),
            ("toggle", "A door changes open/locked state, including closing/reopening, or a toggle action removes/replaces a box with its contents. Blocked attempts do not count."),
        ]), "",
        table(["Choice", "Fixed Stage 1 behavior", "Stage 2 pre-registration"], [
            ("Final-step events", "Count, subject to the successful target-pickup exception", "OPEN decision"),
            ("Box openings", "Count as toggle", "OPEN decision"),
            ("Successful target pickup", "Excluded from paid pickup event; separately diagnosed", "OPEN decision"),
            ("Detector attachment", "Only after the initial seeded reset", "OPEN decision"),
        ]), "",
        "The corrected descriptive satiation control maps new_view/pickup/toggle to the original three needs and observed state; it adds no drop need. The four-event module retains all four needs, including zero-weight channels.", "",
        "## Validation and signed ceiling", ""]
    for validation in data["validation"]:
        lines.append(f"Validation `{validation['id']}`: complete={number(validation['complete'])}; passed cases={number(validation['passed_cases'])}, passed subtests={number(validation['passed_subtests'])}; XML aggregate tests={number(validation['tests'])}, testcase nodes={number(validation['testcase_nodes'])}, failures={number(validation['failures'])}, errors={number(validation['errors'])}, skipped={number(validation['skipped'])}; passed={number(validation['passed'])}. The XML aggregate includes subtests and is not the number of standalone test cases.")
        for filename, value in validation["unit_data"].items():
            if filename == "bound-check.json":
                checks = value.get("episode_checks", [])
                violations = sum(not row["lower"] - value["tolerance"] <= row["return_value"] <= row["upper"] + value["tolerance"] for row in checks)
                lines.append(f"Numerical bound checks: {len(checks)} episode/prefix records; {violations} tolerance violations; tolerance={value['tolerance']}.")
            elif filename == "detector-trajectory-comparison.json":
                lines.extend(["", "Both detectors on the same recorded trajectories:", "",
                    table(["Seed", "Steps", "new_view disagreements", "pickup disagreements", "drop disagreements", "toggle disagreements", "Reward mismatches"],
                          [(row["seed"], row["scalar_steps"], *[row["event_disagreement_counts"][event]
                            for event in value["events"]], row["reward_mismatch_count"]) for row in value["seeds"]]), "",
                    "Drop is an added channel absent from the archived detector and has zero weight in the reduction configuration; its disagreement counts are not detector defects. Per-step disagreement details and indices are retained in the JSON. The reduction criterion concerns arithmetic on recorded paper events; the detector comparison is separate.", "",
                    "Corrected event totals in the recorded corpus:", "",
                    table(["Event", "Count"], [(event, int(sum(row["full_generic_event_counts"][j] for row in value["seeds"])))
                                               for j, event in enumerate(value["events"])])])
                if sum(row["full_generic_event_counts"][3] for row in value["seeds"]) == 0:
                    lines.append("Coverage limitation: this random DoorKey corpus contains no toggle events. Its detector agreement cannot validate door or box fixes; targeted fixtures cover those cases.")
            elif isinstance(value, dict) and "seeds" in value:
                lines.extend(["", f"`{filename}`:", "", table(["Seed", "Scalar steps", "Bit identical"],
                    [(row["seed"], row.get("scalar_steps", "pending"), number(row.get("bit_identical"))) for row in value["seeds"]])])
        targeted = [
            ("Initial empty/stale door cache", "test_initial_door_cache_is_populated_after_reset"),
            ("Box opening and timeout events", "test_box_and_timeout_events_compare_both_detectors_on_same_transition"),
            ("Terminal pickup, target exclusion and physical events", "test_any_object_pickup_drop_toggle_and_terminal_capture"),
        ]
        lines.extend(["", table(["Targeted coverage", "Test", "Passed"],
                     [(description, name, number(name in validation.get("passing_targeted_tests", [])))
                      for description, name in targeted]), "",
                     "Evidence limits: targeted fixture pass status and source are retained, but detailed fixture trajectories were not saved. The recorded corpus retains observations/actions/done/events/rewards; it does not retain raw object states, raw task rewards, separate termination/truncation flags or need histories. Need histories can be reconstructed from the saved events. The separate bound-rollout helper did not retain its own raw arrays. Independent verification against the saved reduction corpus reproduces the bound returns, but does not establish direct provenance of those unretained helper arrays."])
    lines += ["", r"Let $V(h)=\sum_k w_k h_k^2$, $L=\sum_k\min(w_k,0)$, and $U=\sum_k\max(w_k,0)$. The archived dynamics apply one clip after growth minus event relief: $h_{t+1}=\operatorname{clip}(h_t+g-s e_t,0,1)$, followed by an optional tamper reset. For one episode or finite prefix with $0<\gamma\le1$:", "",
        r"$$G_T=V(h_0)-(1-\gamma)\sum_{t=1}^{T-1}\gamma^{t-1}V(h_t)-\gamma^{T-1}V(h_T).$$", "",
        r"The coefficients on future potentials are nonnegative and sum to one. Consequently $V(h_0)-U\le G_T\le V(h_0)-L$. At fixed $h_0=.5$, the upper bound is $\frac14\sum_{w_k\ge0}w_k+\frac34\sum_{w_k<0}|w_k|\le3$. For nonnegative unit weights on the three legacy needs this reduces to $.75$. Episode resets are not paid transitions; adding returns over repeated episodes does not inherit a single-episode constant ceiling. Floating-point checks use an explicit numerical tolerance.", "",
        "## Modal, reproducibility and spending", "", "Declared environment:", "", "```json", json.dumps(data["environment"], indent=2), "```", "",
        "One function input represents one configuration/task/seed/budget. Outputs and complete package inventories are stored on the Modal volume and synchronized with hashes. Raw timing and serializer metadata are excluded from numerical bit-identity checks.", "",
        "Python 3.12.12 and NumPy 2.5.4 follow the archived Mac runtime log and conditional dependency markers. This choice was registered before new results; the earlier unconditional NumPy 2.4.6 statement was incomplete. The container base digest and each attempt's actual installed package inventory are retained in environment/provenance files. The full transitive package lock is uv.lock.", "",
        table(["Task / method", "Seed", "Budget", "Completed repeats", "Numerically bit identical"],
              [(row["task"] + " / " + row["method"], row["seed"], row["budget"], row["completed"], number(row["bit_identical"])) for row in data["determinism"]]), "",
        *determinism_lines(data), *container_lines(data),
        "Spending record (guard estimates and provider-reported completed intervals are distinct from an invoice):", "", "```json", json.dumps(data["spending"], indent=2), "```", "",
        "## Descriptive Mac reference", "", "The two Modal groups are compared with the same original Mac reference using counts and curve areas only. No inferential tests or cross-platform conclusions are drawn, and no configuration is adjusted from this comparison.", "",
        table(["Seed", "Mac solve / area", "Archived Modal solve / area", "Corrected Modal solve / area"],
              [(row["seed"], *[("pending" if row[key] is None else f"{int(row[key]['solved'])} / {row[key]['curve_area']:.6f}")
                                for key in ("mac", "archived_modal", "corrected_modal")]) for row in data["cross_platform"]["seeds"]]), "",
        table(["Group", "Completed", "Solve count", "Mean curve area"], [(key, value["n"], value["solves"], number(value["mean_area"], 6)) for key, value in data["cross_platform"]["aggregates"].items()]), "",
        "## Calibration", "", "AUC is the arithmetic mean of logged rolling training success. Solve requires mean success at least .9 over the final floor(10%) of log rows. First-solved medians include unsolved seeds right-censored at the nominal budget. Actual training stops at the last complete PPO batch. Cost columns are function-compute estimates, with invoices retained separately. Partial cells are descriptive and cannot select a task.", "",
        table(["Task", "Method", "Budget", "Completed", "Solves", "Mean AUC", "Median first solve", "Censored", "Mean wall seconds", "Mean estimated USD"],
              [(c["task"], c["method"], c["budget"], f"{c['n']}/{c['expected_n']}", c["solve_count"], number(c["mean_curve_area"],6), number(c["median_first_solved_step"],1), c["first_solved_censored_count"], number(c["mean_wall_seconds"],1), number(c["mean_cost_estimate_usd"],6)) for c in cal["cells"]]), "",
        table(["Task", "Role", "Selection", "Best mean AUC", "Reason"], [(row["task"], row["role"], row["status"], number(row["best_mean_curve_area"],6), "; ".join(row["reasons"]) or "Fixed rule passed") for row in cal["selection"]]), "",
        "Training-task baseline separation at the fitness budget uses two-sided MWU (exact without ties, permutation with ties), Holm-adjusted across the three planned tasks. A missing or non-significant result does not establish equal performance; the fitness budget remains a user decision.", "",
        table(["Task", "Novelty − task-only mean AUC", "p", "Holm p", "Detected separation"], [(row["task"], number(row["mean_area_difference_novelty_minus_task_only"],6), number(row["p"],6), number(row["p_holm"],6), number(row["separates"])) for row in cal["separation_at_1M"]]), "",
        "## Cost projection", "", "Stage 2 requires 240 candidates across both search methods, three training seeds per task: 720 training runs per kept training task, before finalist re-evaluation. Candidate runtime is not yet measured; the baseline mean is a planning proxy. No projection selects tasks from incomplete cells.", ""]
    project = data["cost_projection"]
    if not project["selection_complete"]:
        lines.append("Total projections are pending because task selection is incomplete. Per-task measured costs remain in the machine-readable data.")
    else:
        lines += [table(["Stage 2 fitness budget scenario", "Training runs", "Estimated USD"], [(r["fitness_budget"], r["training_runs"], number(r["cost_estimate_usd"],4)) for r in project["stage2"]]), "",
            "Stage 3 baseline-roster scenarios (not commitments):", "", table(["Methods", "Training runs", "Estimated USD"], [(r["method_count_scenario"], r["training_runs"], number(r["cost_estimate_usd"],4)) for r in project["stage3"]]), "",
            "Additional tamper training scenarios, if the no-tamper Stage 3 runs serve as controls:", "", table(["Methods", "Extra tamper conditions", "Additional runs", "Additional estimated USD"], [(r["method_count_scenario"], r["additional_training_conditions"], r["additional_runs"], number(r["additional_cost_estimate_usd"],4)) for r in project["tamper_extra"]])]
    lines += ["", "Excluded from these projections: " + project["excluded"], "", "## Stage 2 draft and decisions", "", draft_outline().replace("# Stage 2 pre-registration outline — DRAFT, NOT LOCKED", "DRAFT, NOT LOCKED"), "", "Stage boundary: no evolution or random search starts from this report. User approval and resolution of the listed open decisions are required.", ""]
    if data["invalid_completed_attempts"]:
        lines += ["Invalid completed-attempt records:", "", "```json", json.dumps(data["invalid_completed_attempts"], indent=2), "```", ""]
    lines += container_appendix(data)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--manifest", type=Path, default=ROOT / "stage1/manifest.json")
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--volume-attempts", type=Path, default=VOLUME_ATTEMPTS)
    args = parser.parse_args()
    manifest = read_optional(args.manifest)
    scores, failures = load_completed_scores(args.results, manifest)
    calibration = aggregate_calibration(manifest, scores)
    validations = validation_records(args.results, manifest)
    containers = container_records(args.results, manifest, args.volume_attempts)
    determinism = determinism_records(manifest, scores)
    ledger = read_optional(args.results / "budget-ledger.json")
    spend = {"completed_training_calls": len(scores), "guard_total_usd": None,
             "provider_invoice_cost_usd": None, "provider_reported_complete_intervals_cost_usd": None}
    if ledger:
        for key in ("guard_total_usd", "provider_invoice_cost_usd", "provider_reported_complete_intervals_cost_usd", "limit_usd", "overhead_allowance_usd", "setup_failures", "setup_attempt_guard_usd", "setup_guard_basis", "limitations"):
            spend[key] = ledger.get(key)
        spend["ledger_call_count"] = len(ledger.get("calls", {}))
    environment = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    data = {"schema": 1, "manifest": str(args.manifest), "validation": validations,
        "environment": {"python": (ROOT / ".python-version").read_text().strip(), "dependencies": environment["dependencies"]},
        "calibration": calibration, "determinism": determinism,
        "determinism_explanation": determinism_explanations(manifest, scores, containers, determinism),
        "containers": containers,
        "cross_platform": comparison_records(manifest, scores), "spending": spend,
        "cost_projection": cost_projection(calibration), "scores_by_run": scores,
        "invalid_completed_attempts": failures,
        "complete": len(scores) == sum(spec["purpose"] != "validation" for spec in manifest["runs"])
                    and all(value["passed"] for value in validations) and bool(validations) and not failures}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage1-numbers.json").write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.out / "stage1-report.md").write_text(render(data))
    (args.out / "stage2-preregistration-draft.md").write_text(draft_outline())
    print(json.dumps({"complete": data["complete"], "report": str(args.out / "stage1-report.md"), "numbers": str(args.out / "stage1-numbers.json")}))


if __name__ == "__main__":
    main()
