"""Generate the fixed Stage 1 schedule; this is not an optimizer."""
import copy
import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ["DoorKey-8x8", "KeyCorridorS3R1", "UnlockPickup"]
HELD_OUT = ["KeyCorridorS3R2", "BlockedUnlockPickup", "DoorKey-16x16", "MultiRoom-N2-S4"]


def manifest():
    base = tomllib.loads((ROOT / "handoff-cross-platform/code/final-round5/config.toml").read_text())
    runs = []

    def add(purpose, task, method, seed, budget, replicate=0):
        cfg = copy.deepcopy(base)
        paper_run = purpose in {"cross_platform", "cross_platform_corrected"}
        if paper_run:
            cfg = tomllib.loads((ROOT / "handoff-cross-platform/code/a1b-run-commit-1533734/config.toml").read_text())
        else:
            cfg["env"]["eval_ids"] = [f"MiniGrid-{task}-v0"]
            cfg["intrinsic"].update(kind=method, coef=0.001 if method == "episodic_count" else 0.0,
                                    event_mode="doorkey", tamper=False)
        cfg["env"]["id"] = f"MiniGrid-{task}-v0"
        cfg["run"].update(seeds=[seed], total_timesteps=budget)
        ident = f"{purpose}-{task}-{method}-{budget}-seed{seed}-r{replicate}"
        runs.append(dict(id=ident, purpose=purpose, task=task, method=method, seed=seed,
                         total_timesteps=budget, replicate=replicate,
                         source="a1b-run-commit-1533734" if paper_run else "final-round5",
                         detector="archived" if purpose == "cross_platform" else "corrected",
                         config=cfg))

    # Validation creates no learned policy and performs no parameter search.
    runs.append(dict(id="validation-detector-and-reward", purpose="validation",
                     task="unit-suite", method="validation", seed=905,
                     total_timesteps=0, replicate=0, source="final-round5",
                     detector="corrected", config=copy.deepcopy(base)))
    for seed in (910, 911):
        for replicate in (0, 1):
            add("determinism", "DoorKey-8x8", "episodic_count", seed, 131_072, replicate)
    for seed in range(40, 45):
        add("cross_platform", "DoorKey-8x8", "homeostatic", seed, 2_000_000)
        add("cross_platform_corrected", "DoorKey-8x8", "homeostatic", seed, 2_000_000)
    for budget in (1_000_000, 2_000_000):
        for seed in range(900, 905):
            for task in TRAINING + HELD_OUT:
                # Counterbalance arm order deterministically, without inspecting results.
                methods = ("none", "episodic_count") if seed % 2 == 0 else ("episodic_count", "none")
                for method in methods:
                    add("calibration", task, method, seed, budget)
    return dict(stage=1, training_tasks=TRAINING, held_out_tasks=HELD_OUT,
                calibration_seeds=list(range(900, 905)), budgets=[1_000_000, 2_000_000],
                no_evolution_or_random_search=True, budget_usd=15.0,
                max_concurrent_calls=4, validation_must_pass_first=True,
                current_event_decisions=dict(final_step_events_count=True,
                    box_openings_count=True, successful_target_pickup_excluded=True,
                    detector_attaches_after_reset=True,
                    stage2_status="All four definitions are open Stage 2 decisions; fixed for Stage 1"),
                runs=runs)


if __name__ == "__main__":
    out = ROOT / "stage1/manifest.json"
    out.write_text(json.dumps(manifest(), indent=2) + "\n")
    print(f"manifest_sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
