"""Fixed Stage 1 scoring; selection is forbidden from incomplete baseline cells.

``score_seed(directory, budget)`` is independent of Modal transport. The optional
CLI discovers completed attempts and writes the same pure-function result.
Training AUC and solve categories reproduce the archived row-based definitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import numpy as np

from stage1.statistics import cp, holm, mwu

TRAINING_TASKS = ("DoorKey-8x8", "KeyCorridorS3R1", "UnlockPickup")
HELD_OUT_TASKS = ("KeyCorridorS3R2", "BlockedUnlockPickup", "DoorKey-16x16", "MultiRoom-N2-S4")
METHODS = ("none", "episodic_count")
BUDGETS = (1_000_000, 2_000_000)
CALIBRATION_SEEDS = tuple(range(900, 905))
BATCH_SIZE = 16 * 128
LOG_INTERVAL = BATCH_SIZE * 10


class IncompleteRun(ValueError):
    """Missing or inconsistent evidence cannot be scored as a completed run."""


def _read_json(path):
    return json.loads(Path(path).read_text())


def _probability(value, label):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise IncompleteRun(f"{label} must be a finite probability")
    return value


def score_seed(directory, budget):
    """Score a complete archived-format run at its own nominal training budget.

    First-solved is right-censored at the nominal budget. The final-10% window
    is a count of logged rows, following the paper; it is not a time-weighted
    interval. Missing early ``train/success`` values count as zero, as archived.
    """
    directory = Path(directory)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < BATCH_SIZE:
        raise ValueError("budget must be an integer allowing a full PPO rollout")
    try:
        summary = _read_json(directory / "summary.json")
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise IncompleteRun(f"Unreadable evidence in {directory}: {error}") from error
    actual = budget // BATCH_SIZE * BATCH_SIZE
    expected_steps = list(range(LOG_INTERVAL, actual + 1, LOG_INTERVAL))
    if not expected_steps or expected_steps[-1] != actual:
        expected_steps.append(actual)
    if summary.get("total_timesteps") != actual:
        raise IncompleteRun(f"Expected {actual} completed steps; summary reports {summary.get('total_timesteps')}")
    if [row.get("global_step") for row in rows] != expected_steps:
        raise IncompleteRun("Metrics are missing, duplicated, out of order, or do not follow the registered logging schedule")
    success = [_probability(row.get("train/success", 0), "train/success") for row in rows]
    final_rows = max(1, len(rows) // 10)
    final_success = float(np.mean(success[-final_rows:]))
    for key in ("first_solved_step", "first_goal_step"):
        if key not in summary:
            raise IncompleteRun(f"Missing summary field {key}")
        step = summary[key]
        if step is not None and (not isinstance(step, int) or not 0 < step <= actual or step % 16):
            raise IncompleteRun(f"Invalid {key}: {step}")
    first_solved = summary["first_solved_step"]
    if "seed" not in summary or not isinstance(summary["seed"], int):
        raise IncompleteRun("Missing or invalid summary seed")
    out = {
        "seed": summary["seed"], "nominal_budget": budget, "actual_steps": actual,
        "curve_area": float(np.mean(success)), "solved": final_success >= 0.9,
        "final_success": final_success, "final_window_rows": final_rows,
        "log_rows": len(rows), "first_solved_step": first_solved,
        "first_solved_censored": budget if first_solved is None else first_solved,
        "first_solved_is_censored": first_solved is None,
        "first_goal_step": summary["first_goal_step"],
        "success_ever": summary["first_goal_step"] is not None or max(success) > 0,
        "max_logged_success": max(success),
        "wall_seconds": None, "process_core_seconds": None, "peak_rss_bytes": None,
        "cost_usd": None, "cost_estimate_usd": None,
    }
    usage_path = directory / "process_usage.json"
    if usage_path.exists():
        usage = _read_json(usage_path)
        if usage.get("exit_code") != 0:
            raise IncompleteRun("Training process did not exit successfully")
        for key in ("wall_seconds", "process_core_seconds", "peak_rss_bytes"):
            value = usage.get(key)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise IncompleteRun(f"Invalid resource measurement {key}")
            out[key] = value
    function_usage = directory / "usage.json"
    if function_usage.exists():
        usage = _read_json(function_usage)
        if usage.get("returncode") != 0 or usage.get("error") is not None:
            raise IncompleteRun("Modal function did not finish training successfully")
        out["process_wall_seconds"] = out["wall_seconds"]
        out["wall_seconds"] = float(usage["wall_seconds"])
        out["cost_usd"] = usage.get("provider_invoice_cost_usd")
        out["cost_estimate_usd"] = float(usage["compute_cost_estimate_usd"])
        if not all(math.isfinite(out[key]) and out[key] >= 0 for key in ("wall_seconds", "cost_estimate_usd")):
            raise IncompleteRun("Invalid Modal function usage")
    return out


def _short_task(task):
    return task.removeprefix("MiniGrid-").removesuffix("-v0")


def _cell(task, method, budget, specs, scores):
    by_seed = {}
    for spec in specs:
        if (_short_task(spec["task"]), spec["method"], spec["total_timesteps"]) == (task, method, budget):
            seed = spec["seed"]
            if seed in by_seed or seed not in CALIBRATION_SEEDS:
                raise ValueError(f"Duplicate or unauthorized calibration seed in {(task, method, budget)}")
            by_seed[seed] = spec
    observed = []
    missing = []
    for seed in CALIBRATION_SEEDS:
        spec = by_seed.get(seed)
        score = None if spec is None else scores.get(spec["id"])
        if score is None:
            missing.append(seed)
            continue
        if score["seed"] != seed or score["nominal_budget"] != budget:
            raise ValueError(f"Score identity differs from manifest: {spec['id']}")
        observed.append({"id": spec["id"], **score})
    n = len(observed)
    solved = sum(score["solved"] for score in observed)
    cell = {
        "task": task, "method": method, "budget": budget, "n": n,
        "expected_n": len(CALIBRATION_SEEDS), "complete": not missing,
        "missing_seeds": missing, "solve_count": solved,
        "solve_ci95": cp(solved, n) if n else None,
        "mean_curve_area": mean(score["curve_area"] for score in observed) if n else None,
        "median_first_solved_step": median(score["first_solved_censored"] for score in observed) if n else None,
        "first_solved_censored_count": sum(score["first_solved_is_censored"] for score in observed),
        "mean_final_success": mean(score["final_success"] for score in observed) if n else None,
        "any_success_ever": any(score["success_ever"] for score in observed),
        "seeds": observed,
    }
    for key in ("wall_seconds", "process_core_seconds", "peak_rss_bytes", "cost_usd", "cost_estimate_usd"):
        values = [float(score[key]) for score in observed if score.get(key) is not None]
        cell[f"mean_{key}"] = mean(values) if values else None
        cell[f"n_{key}"] = len(values)
        if key in {"cost_usd", "cost_estimate_usd"}:
            cell[f"total_{key}"] = sum(values) if values else None
    return cell


def aggregate_calibration(manifest, scores):
    """Aggregate manifest-id -> score mappings; never select on partial cells."""
    runs = manifest["runs"] if isinstance(manifest, dict) else manifest
    ids = [spec["id"] for spec in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest run ids must be unique")
    specs = [spec for spec in runs if spec["purpose"] == "calibration"]
    for spec in specs:
        if (_short_task(spec["task"]) not in TRAINING_TASKS + HELD_OUT_TASKS
                or spec["method"] not in METHODS or spec["total_timesteps"] not in BUDGETS):
            raise ValueError("Manifest contains a calibration cell outside the fixed Stage 1 design")
    cells = [_cell(task, method, budget, specs, scores)
             for task in TRAINING_TASKS + HELD_OUT_TASKS for method in METHODS for budget in BUDGETS]
    lookup = {(cell["task"], cell["method"], cell["budget"]): cell for cell in cells}
    selections = []
    for task in TRAINING_TASKS + HELD_OUT_TASKS:
        training = task in TRAINING_TASKS
        needed_budgets = BUDGETS if training else (2_000_000,)
        required = [lookup[task, method, budget] for method in METHODS for budget in needed_budgets]
        choice = {"task": task, "role": "training" if training else "held_out",
                  "status": "incomplete", "kept": None, "reasons": [], "best_mean_curve_area": None}
        if not all(cell["complete"] for cell in required):
            choice["reasons"] = ["Selection requires five completed seeds in both baselines at every required budget"]
        else:
            budget = 1_000_000 if training else 2_000_000
            best = max(lookup[task, method, budget]["mean_curve_area"] for method in METHODS)
            choice["best_mean_curve_area"] = best
            if best > 0.85:
                choice["reasons"].append("Best baseline mean curve area exceeds 0.85")
            if training and not any(lookup[task, method, 2_000_000]["any_success_ever"] for method in METHODS):
                choice["reasons"].append("No baseline seed reaches success above zero by 2M")
            if not training and best < 0.05:
                choice["reasons"].append("Best baseline mean curve area is below 0.05")
            choice["kept"] = not choice["reasons"]
            choice["status"] = "kept" if choice["kept"] else "rejected"
        selections.append(choice)
    comparisons = []
    for task in TRAINING_TASKS:
        task_only, novelty = (lookup[task, method, 1_000_000] for method in METHODS)
        comparison = {"task": task, "complete": task_only["complete"] and novelty["complete"],
                      "mean_area_difference_novelty_minus_task_only": None,
                      "p": None, "method": None, "p_holm": None, "separates": None}
        if comparison["complete"]:
            p, method = mwu([s["curve_area"] for s in novelty["seeds"]],
                            [s["curve_area"] for s in task_only["seeds"]], "two-sided")
            comparison.update(p=p, method=method,
                              mean_area_difference_novelty_minus_task_only=novelty["mean_curve_area"] - task_only["mean_curve_area"])
        comparisons.append(comparison)
    adjusted = holm([comparison["p"] for comparison in comparisons], family_size=len(TRAINING_TASKS))
    for comparison, p in zip(comparisons, adjusted):
        comparison.update(p_holm=p, separates=None if p is None else p < 0.05)
    return {
        "cells": cells, "selection": selections, "separation_at_1M": comparisons,
        "complete": all(cell["complete"] for cell in cells),
        "interpretation": "Failure to detect a difference does not establish equivalence. Partial cells are descriptive and never select tasks.",
    }


def load_completed_scores(results_root, manifest):
    """Discover exactly one successful attempt per manifest entry, if present.

    Failed/unfinished attempts are retained on disk but are not learning results.
    Their resource charges must still enter the controller's spending ledger.
    More than one successful attempt for an id is an audit error, not a choice
    of which learning outcome to retain.
    """
    results_root = Path(results_root)
    specs = {spec["id"]: spec for spec in manifest["runs"]}
    scores, failures = {}, []
    for marker in sorted(results_root.rglob("complete.json")):
        complete = _read_json(marker)
        if complete.get("id") not in specs:
            continue
        spec = specs[complete["id"]]
        if spec["purpose"] == "validation":
            continue
        attempt = complete["attempt"]
        if not isinstance(attempt, str) or Path(attempt).name != attempt:
            raise ValueError("Invalid completed attempt identifier")
        directory = marker.parent / "attempts" / attempt
        try:
            recorded_spec = _read_json(directory / "spec.json")
            if recorded_spec != spec:
                raise IncompleteRun("Saved spec differs from locked manifest")
            canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            if hashlib.sha256(canonical).hexdigest() != complete["spec_sha256"]:
                raise IncompleteRun("Completed marker specification hash differs")
            for name, expected in complete["files"].items():
                parts = Path(name).parts
                if len(parts) < 5 or parts[:2] != ("runs", spec["id"]) or ".." in parts:
                    raise IncompleteRun("Invalid file path in completed marker")
                local = marker.parent.joinpath(*parts[2:])
                if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
                    raise IncompleteRun(f"Completed output hash differs: {name}")
            if not (directory / "fingerprint.json").exists():
                raise IncompleteRun("Missing successful learner fingerprint")
            score = score_seed(directory, spec["total_timesteps"])
        except (IncompleteRun, OSError, json.JSONDecodeError) as error:
            failures.append({"id": spec["id"], "directory": str(directory), "error": str(error)})
            continue
        if spec["id"] in scores:
            raise ValueError(f"Multiple successful attempts need adjudication: {spec['id']}")
        score["directory"] = str(directory)
        score["fingerprints"] = complete.get("fingerprints", {})
        score["source_commit"] = complete.get("source_commit")
        scores[spec["id"]] = score
    return scores, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("stage1/manifest.json"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = _read_json(args.manifest)
    scores, failures = load_completed_scores(args.results, manifest)
    report = aggregate_calibration(manifest, scores)
    report.update(scores_by_run=scores, invalid_completed_attempts=failures)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
