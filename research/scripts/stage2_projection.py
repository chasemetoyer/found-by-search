"""Stage 2 cost and duration projection against the user-approved $35 cap.

Inputs are measured, never assumed: Stage 1 per-run 2M costs on the two kept
training tasks (stage1-numbers.json), Stage 1's billed/measured and guard/measured
ratios, and the pinned determinism test's runtime relative to Stage 1's unpinned
runs of the same spec (pinning can select slower kernels on AVX-512 machines).
"""
from __future__ import annotations

import argparse
import json
import statistics
from decimal import Decimal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1.budget import RATE, TERMINATION_ALLOWANCE_SECONDS
from stage2.budget import LIMIT_USD, OVERHEAD_ALLOWANCE_USD
from stage2.common import MAX_CONCURRENT_CALLS, SHARED
from stage2.search import REGISTERED_DRAFT

ART = Path("/Users/chasemetoyer/.local/share/openresearch/files/evolved-intrinsic-rewards")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-numbers", type=Path, default=ART / "stage1/stage1-numbers.json")
    p.add_argument("--stage2-ledger", type=Path, required=True)
    p.add_argument("--pinned-results", type=Path, required=True, help="Results directory of the pinned determinism run")
    p.add_argument("--stage1-results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    numbers = json.loads(a.stage1_numbers.read_text())
    cfg = REGISTERED_DRAFT
    per_task = {}
    for task in cfg.tasks:
        cells = [c for c in numbers["calibration"]["cells"] if c["task"] == task and c["budget"] == 2_000_000]
        costs = [s["cost_estimate_usd"] for c in cells for s in c["seeds"]]
        walls = [s["wall_seconds"] for c in cells for s in c["seeds"]]
        per_task[task] = {"runs": len(costs), "mean_cost_usd": statistics.fmean(costs), "mean_wall_seconds": statistics.fmean(walls)}

    def stage_wall(results: Path, prefix: str) -> list[float]:
        walls = []
        for marker in sorted(results.glob(prefix + "*/complete.json")):
            walls.append(json.loads(marker.read_text())["usage"]["wall_seconds"])
        return walls

    pinned = stage_wall(a.pinned_results, "pinned_determinism-")
    unpinned = stage_wall(a.stage1_results, "determinism-")
    slowdown = statistics.fmean(pinned) / statistics.fmean(unpinned)
    # Stage 1 ratios: provider-billed / measured function compute, and guard charges / measured.
    stage1 = json.loads((SHARED / "budget.json").read_text())["stages"]["stage1"]
    measured_s1 = float(stage1["measured_completed_function_compute_usd"])
    billed_ratio = float(stage1["modal_billed_all_run_hours_usd"]) / measured_s1
    guard_ratio = float(stage1["guard_components"]["completed_guard_charges_usd"]) / measured_s1
    unit = statistics.fmean(per_task[t]["mean_cost_usd"] for t in cfg.tasks)
    unit_wall = statistics.fmean(per_task[t]["mean_wall_seconds"] for t in cfg.tasks)
    search_runs = len(cfg.tasks) * 2 * cfg.rs_candidates * cfg.seeds_per_task
    finalist_runs = len(cfg.tasks) * 2 * cfg.finalists_per_method * cfg.finalist_seeds_per_task
    ledger = json.loads(a.stage2_ledger.read_text())
    spent_guard = float(ledger["guard_total_usd"]) - float(OVERHEAD_ALLOWANCE_USD)
    peak_reservations = MAX_CONCURRENT_CALLS * float(RATE) * (cfg.deadline_seconds + TERMINATION_ALLOWANCE_SECONDS)

    def scenario(factor: float) -> dict:
        measured = (search_runs + finalist_runs) * unit * factor
        return {"pinning_runtime_factor": factor, "measured_compute_usd": measured,
                "expected_billed_usd": measured * billed_ratio,
                "projected_guard_at_end_usd": measured * guard_ratio + spent_guard + float(OVERHEAD_ALLOWANCE_USD),
                "projected_guard_peak_usd": measured * guard_ratio + spent_guard + float(OVERHEAD_ALLOWANCE_USD) + peak_reservations,
                "wall_hours_at_32": (search_runs + finalist_runs) * (unit_wall * factor + 30) / MAX_CONCURRENT_CALLS / 3600}

    # Approved Stage 3 roster: E/R/T/N plus descriptive fixed satiation, every
    # kept task, then two fresh retraining conditions for DoorKey tampering.
    kept = [r['task'] for r in numbers['calibration']['selection'] if r['kept']]
    stage3_units = {}
    for task in kept:
        costs = [s['cost_estimate_usd'] for c in numbers['calibration']['cells']
                 if c['task'] == task and c['budget'] == 2_000_000 for s in c['seeds']]
        stage3_units[task] = statistics.fmean(costs)
    stage3_compute = 5 * 20 * sum(stage3_units.values()) + 2 * 20 * stage3_units['DoorKey-8x8']

    out = {"cap_usd": float(LIMIT_USD), "registered_draft": cfg.name, "search_runs": search_runs,
           "finalist_runs": finalist_runs, "per_task_stage1_2M": per_task, "unit_cost_usd": unit,
           "pinned_mean_wall_s": statistics.fmean(pinned), "unpinned_stage1_mean_wall_s": statistics.fmean(unpinned),
           "pinned_runs": len(pinned), "unpinned_runs": len(unpinned), "pinning_slowdown_131k": slowdown,
           "billed_to_measured_stage1": billed_ratio, "guard_to_measured_stage1": guard_ratio,
           "stage2_guard_spent_before_search_usd": spent_guard, "peak_in_flight_reservations_usd": peak_reservations,
           "scenarios": {"unpinned": scenario(1.0), "pinned": scenario(slowdown)},
           "stage3": {"cap_usd": 10, "method_count": 5, "final_evaluation_runs": 5 * 20 * len(kept),
                      "tamper_retraining_runs": 40, "per_task_measured_unit_cost_usd": stage3_units,
                      "measured_compute_proxy_usd": stage3_compute,
                      "expected_billed_proxy_usd": stage3_compute * billed_ratio,
                      "projected_guard_at_end_usd": stage3_compute * guard_ratio + float(OVERHEAD_ALLOWANCE_USD),
                      "projected_guard_peak_usd": stage3_compute * guard_ratio + float(OVERHEAD_ALLOWANCE_USD) + peak_reservations,
                      "separate_execution_approval_required": True},
           "basis": "Stage 1 measured function-wall cost of baseline runs as a proxy for candidate runs; whole-call "
                    "overhead and guard behavior scaled by Stage 1 ratios; +30 s startup per call for duration."}
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("unit_cost_usd", "pinning_slowdown_131k", "peak_in_flight_reservations_usd")}))
    for name, s in out["scenarios"].items():
        print(name, {k: round(v, 2) for k, v in s.items()})


if __name__ == "__main__":
    main()
