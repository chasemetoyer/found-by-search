"""Independent extraction of headline values into data/extracted/independent-extraction.json (read-only).

This is a different code path from make_numbers.py, which reads the Stage 3 analysis outputs. Here:
- training-curve areas and solves are recomputed from every raw metrics.jsonl (missing train/success counts as 0;
  solved = mean of the final 9 logged values >= 0.9), and U statistics from those areas;
- p-values come from the Stage 3 independent verification (verification/independent-score-and-tests.json), and
  Holm adjustments are recomputed here from them (step-down, running maximum, capped at 1);
- P0 is recomputed from the frozen winner file in the repository: 0.25 * fsum of the float32 weights;
- tamper press rates, intrinsic returns and entropies are recomputed from the raw descriptive-windows.jsonl and
  metrics.jsonl; ceiling counts come from verification/independent-descriptive-check.json;
- Stage 2 finalist scores are recomputed from per-run areas, exploratory group means from the candidate table, and
  Stage 1 selection areas from the calibration cells.
scripts/check_numbers.py formats these values and compares them with the generated macros as strings.

Usage (repo root): EIR_ARTIFACTS=<artifacts directory> python3 scripts/extract_claims.py   (or: make extract)
"""
import csv
import json
import os
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW, EXTRACTED = ROOT / "data" / "raw", ROOT / "data" / "extracted"
ART = Path(os.environ.get("EIR_ARTIFACTS", ""))  # the project's artifacts directory
RUNS = ART / "stage3-report" / "data" / "runs"
VER = ART / "stage3-report" / "verification"


def runs():
    out = defaultdict(dict)  # (method, task) -> seed -> metrics rows
    for d in sorted(RUNS.iterdir()):
        att = d / "attempts" / json.loads((d / "complete.json").read_text())["attempt"]
        spec = json.loads((att / "spec.json").read_text())
        rows = [json.loads(l) for l in (att / "metrics.jsonl").read_text().splitlines() if l]
        out[(spec["method"], spec["task"])][spec["seed"]] = (rows, att)
    return out


def area(rows):
    return sum(r.get("train/success", 0.0) for r in rows) / len(rows)


def solved(rows):
    k = max(1, len(rows) // 10)
    return sum(r.get("train/success", 0.0) for r in rows[-k:]) / k >= 0.9


def U(a, b):
    return sum((x > y) + 0.5 * (x == y) for x in a for y in b)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    if not RUNS.is_dir():
        raise SystemExit("Set EIR_ARTIFACTS to the project's artifacts directory")
    R = runs()
    C = {"cells": {}, "comparisons": {}, "tamper": {}, "finalists": {}, "exploratory": {}, "stage1": {}, "genomes": {}}
    areas = {}
    for (m, t), seeds in R.items():
        a = [area(r) for r, _ in seeds.values()]
        areas[(m, t)] = a
        C["cells"][f"{m}|{t}"] = {"n": len(a), "mean_area": statistics.fmean(a), "sd_area": statistics.stdev(a),
                                  "solves": sum(solved(r) for r, _ in seeds.values())}
    ind = json.loads((VER / "independent-score-and-tests.json").read_text())
    for fam in ("primary", "secondary"):
        rows = [r for r in ind["registered_comparisons"] if r["family"] == fam]
        adj = holm([r["p"] for r in rows])
        for r, h in zip(rows, adj):
            e, c = areas[("E", r["task"])], areas[(r["comparator"], r["task"])]
            C["comparisons"][f"{r['task']}|{r['comparator']}"] = {
                "diff": statistics.fmean(e) - statistics.fmean(c), "U": U(e, c), "p": r["p"], "holm": h}
    off, on = areas[("tamper_off", "DoorKey-8x8")], areas[("tamper_on", "DoorKey-8x8")]
    frozen = json.loads((RAW / "repo" / "stage3" / "frozen" / "frozen-winner-cma.json").read_text())["winner"]
    w32 = np.asarray(frozen["genome"]["weights"], dtype=np.float32).astype(float)
    T = C["tamper"]
    T.update({"P0": 0.25 * math.fsum(w32), "off": statistics.fmean(off), "on": statistics.fmean(on),
              "diff": statistics.fmean(off) - statistics.fmean(on), "U": U(off, on), "p": ind["registered_tamper"]["p"],
              "solves_off": C["cells"]["tamper_off|DoorKey-8x8"]["solves"], "solves_on": C["cells"]["tamper_on|DoorKey-8x8"]["solves"],
              "vsT": statistics.fmean(on) - statistics.fmean(areas[("T", "DoorKey-8x8")]),
              "window_medians": [w["median"] for w in ind["descriptive_label"]["median_presses_per_episode_by_window"]]})
    desc = json.loads((VER / "independent-descriptive-check.json").read_text())
    for m in ("tamper_on", "tamper_off"):
        pps, pps10, iret, ent, ent9 = [], [], [], [], []
        for rows, att in R[(m, "DoorKey-8x8")].values():
            wins = [json.loads(l) for l in (att / "descriptive-windows.jsonl").read_text().splitlines() if l]
            pps.append(sum(w["presses"] for w in wins) / sum(w["steps"] for w in wins))
            pps10.append(sum(w["final10_presses"] for w in wins) / sum(w["final10_steps"] for w in wins))
            iret.append(sum(w["episode_intrinsic_return"] for w in wins) / sum(w["episodes"] for w in wins))
            ent.append(statistics.fmean(r["entropy"] for r in rows))
            ent9.append(statistics.fmean(r["entropy"] for r in rows[-9:]))
        T[m] = {"pps": statistics.fmean(pps), "pps10": statistics.fmean(pps10), "iret": statistics.fmean(iret),
                "ent": statistics.fmean(ent), "ent9": statistics.fmean(ent9),
                "above_P0": desc["by_method"][m]["episodes_above_exact_P0"],
                "above_064": desc["by_method"][m]["episodes_above_rounded_064"],
                "max_excess": desc["by_method"][m]["maximum_excess_vs_exact_P0"]}
    per = defaultdict(list)
    with open(RAW / "s2-per-run.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            per[(r["method"], r["candidate"], r["purpose"].startswith("search_finalist"))].append(float(r["area"]))
    for m, c in (("cma", "92"), ("cma", "113"), ("cma", "94"), ("random", "57"), ("random", "116"), ("random", "103")):
        C["finalists"][f"{m}{c}"] = {"search": statistics.fmean(per[(m, c, False)]), "reeval": statistics.fmean(per[(m, c, True)])}
    with open(RAW / "s2x-candidates.csv", newline="") as fh:
        cand = list(csv.DictReader(fh))
    for ev in ("toggle", "pickup", "drop"):
        for strat, pool in (("pooled", cand), ("random", [x for x in cand if x["method"] == "random"])):
            fit = lambda xs: statistics.fmean(float(x["area/combined"]) for x in xs)
            C["exploratory"][f"{strat}|{ev}"] = {
                "positive": fit([x for x in pool if float(x[f"weight/{ev}"]) > 0]),
                "negative": fit([x for x in pool if float(x[f"weight/{ev}"]) < 0]),
                "zero": fit([x for x in pool if x[f"refill_zero/{ev}"] in ("True", "1", "true")]),
                "nonzero": fit([x for x in pool if x[f"refill_zero/{ev}"] not in ("True", "1", "true")])}
    s1 = json.loads((RAW / "s1-numbers.json").read_text())
    for sel in s1["calibration"]["selection"]:
        budget = 1_000_000 if sel["role"] == "training" else 2_000_000
        best = max(c["mean_curve_area"] for c in s1["calibration"]["cells"]
                   if c["task"] == sel["task"] and c["budget"] == budget and c.get("complete"))
        C["stage1"][sel["task"]] = best
    for m, f in (("E", "frozen-winner-cma.json"), ("R", "frozen-winner-random.json")):
        g = json.loads((RAW / "repo" / "stage3" / "frozen" / f).read_text())["winner"]["genome"]
        C["genomes"][m] = {k: np.asarray(g[k], dtype=np.float32).astype(float).tolist() for k in ("weights", "relief", "growth")}
    (EXTRACTED / "independent-extraction.json").write_text(json.dumps(C, indent=1) + "\n")
    print(f"independent-extraction.json: {len(C['cells'])} cells, {len(C['comparisons'])} comparisons, tamper, "
          f"{len(C['finalists'])} finalists, {len(C['exploratory'])} exploratory groups, {len(C['stage1'])} stage-1 tasks")


if __name__ == "__main__":
    main()
