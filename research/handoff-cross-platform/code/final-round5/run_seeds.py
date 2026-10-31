"""Run every seed in config.toml in parallel, stream metrics to Trackio, then aggregate.

Usage: python run_seeds.py [--total-timesteps N]   (override only for smoke tests)
Prints a mean ± std summary across seeds and writes results/aggregate.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

CURVE_KEYS = ["train/success", "train/ret", "train/coverage", "train/got_key",
              "train/opened_door", "r_int_mean", "entropy", "sps"]


def start_trackio(cfg):
    try:
        import trackio
        trackio.init(
            project=os.environ.get("TRACKIO_PROJECT", "agents"),
            name=os.environ.get("ORX_RUN_ID"),
            group=os.environ.get("ORX_EXPERIMENT"),
            config={"hypothesis": os.environ.get("ORX_EXPERIMENT_TITLE"),
                    "parent": os.environ.get("ORX_PARENT_EXPERIMENT"),
                    "commit": os.environ.get("ORX_COMMIT"),
                    **{f"{s}.{k}": v for s, d in cfg.items() for k, v in d.items()}},
        )
        return trackio
    except Exception as e:  # tracking is optional; stdout + results/ are the record
        print(f"trackio disabled: {e}", flush=True)
        return None


def summarize(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return f"{np.mean(xs):.3f} ± {np.std(xs):.3f} (n={len(xs)})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.toml")
    p.add_argument("--total-timesteps", type=int, default=None)
    args = p.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    seeds = cfg["run"]["seeds"]
    print("effective config:\n" + json.dumps(cfg, indent=2), flush=True)

    root = Path("results")
    procs, offsets = {}, {}
    for s in seeds:
        cmd = [sys.executable, "ppo.py", "--config", args.config, "--seed", str(s), "--out", str(root / f"seed{s}")]
        if args.total_timesteps:
            cmd += ["--total-timesteps", str(args.total_timesteps)]
        procs[s] = subprocess.Popen(cmd)
        offsets[s] = 0

    # Stream each seed's metrics.jsonl into Trackio as seed{s}/<key>, plus the across-seed mean.
    tk = start_trackio(cfg)
    latest = {s: {} for s in seeds}

    def pump():
        for s in seeds:
            f = root / f"seed{s}" / "metrics.jsonl"
            if not f.exists():
                continue
            with open(f) as fh:
                fh.seek(offsets[s])
                for line in fh:
                    if not line.endswith("\n"):
                        break
                    offsets[s] += len(line)
                    row = json.loads(line)
                    latest[s] = row
                    if tk:
                        tk.log({f"seed{s}/{k}": row[k] for k in CURVE_KEYS if k in row}, step=row["global_step"])
        if tk and all(latest[s] for s in seeds):
            step = min(latest[s]["global_step"] for s in seeds)
            tk.log({f"mean/{k}": float(np.mean([latest[s].get(k, 0.0) for s in seeds]))
                    for k in CURVE_KEYS}, step=step)

    while any(pr.poll() is None for pr in procs.values()):
        pump()
        time.sleep(15)
    pump()

    failed = [s for s, pr in procs.items() if pr.returncode != 0]
    if failed:
        print(f"seeds failed: {failed}", flush=True)
        sys.exit(1)

    summaries = [json.loads((root / f"seed{s}" / "summary.json").read_text()) for s in seeds]
    agg = {
        "intrinsic": cfg["intrinsic"],
        "env": cfg["env"]["id"],
        "seeds": seeds,
        "first_solved_step": [x["first_solved_step"] for x in summaries],
        "coverage_after_solved": [x["coverage_after_solved"] for x in summaries],
        "final_train": {k: [x["final_train"][k] for x in summaries] for k in summaries[0]["final_train"]},
        "eval": {e: {k: [x["eval"][e][k] for x in summaries] for k in summaries[0]["eval"][e]}
                 for e in summaries[0]["eval"]},
    }
    (root / "aggregate.json").write_text(json.dumps(agg, indent=2))

    print("\n===== SUMMARY (mean ± std across seeds) =====")
    print(f"intrinsic: {cfg['intrinsic']}  env: {cfg['env']['id']}  seeds: {seeds}")
    solved = [x for x in agg["first_solved_step"] if x is not None]
    print(f"seeds solved (rolling success >= 0.9): {len(solved)}/{len(seeds)}")
    print(f"first_solved_step: {summarize(agg['first_solved_step'])}   raw={agg['first_solved_step']}")
    print(f"coverage_after_solved: {summarize(agg['coverage_after_solved'])}")
    for k, v in agg["final_train"].items():
        print(f"final_train {k}: {summarize(v)}")
    for e, d in agg["eval"].items():
        print(f"eval {e}: success {summarize(d['success'])} | ret {summarize(d['ret'])} | "
              f"coverage {summarize(d['coverage'])} | key {summarize(d['got_key'])} | door {summarize(d['opened_door'])}")
    if tk:
        tk.log({f"final/eval/{e}/success": float(np.mean(d["success"])) for e, d in agg["eval"].items()})
        tk.finish()


if __name__ == "__main__":
    main()
