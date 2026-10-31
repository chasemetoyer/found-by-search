"""Descriptive Stage2 genome summaries, specified after Stage2 outcomes existed.

No tests of hypotheses, p-values, new training, winner substitution or scoring
changes. This module is separate from the registered Stage3 inference code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

EVENTS = ("new_view", "pickup", "drop", "toggle")
TASKS = ("DoorKey-8x8", "UnlockPickup")
TARGETS = ("combined",) + TASKS
STATUS = "exploratory; specified after observing Stage2 results; descriptive only"


def average_ranks(values):
    """One-based average ranks, with ties defined by exact recorded equality."""
    values = [float(v) for v in values]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Ranks require finite observations")
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for i in order[start:end]:
            ranks[i] = rank
        start = end
    return ranks


def spearman(x, y):
    """Pearson correlation of average ranks; undefined cases stay explicit."""
    x, y = list(x), list(y)
    if len(x) != len(y):
        raise ValueError("Correlation observations must have equal lengths")
    if not all(math.isfinite(float(v)) for v in x + y):
        raise ValueError("Correlation observations must be finite")
    result = {"n": len(x), "rho": None, "status": "defined"}
    if len(x) < 2:
        result["status"] = "undefined: fewer than two observations"
        return result
    rx, ry = average_ranks(x), average_ranks(y)
    dx = [v - statistics.fmean(rx) for v in rx]
    dy = [v - statistics.fmean(ry) for v in ry]
    sx, sy = math.fsum(v*v for v in dx), math.fsum(v*v for v in dy)
    if sx == 0 or sy == 0:
        result["status"] = "undefined: " + ("constant feature and area" if sx == sy == 0 else "constant feature" if sx == 0 else "constant area")
        return result
    result["rho"] = max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(dx, dy)) / math.sqrt(sx*sy)))
    return result


def quantile(values, p):
    """Linear interpolation at (n-1)*p, including singleton groups."""
    ordered = sorted(values)
    if not ordered:
        return None
    point = (len(ordered)-1)*p
    left = math.floor(point)
    right = min(left + 1, len(ordered)-1)
    return ordered[left] + (point-left)*(ordered[right]-ordered[left])


def summarize(values):
    values = list(values)
    q1, q3 = quantile(values, .25), quantile(values, .75)
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None, "q1": q1, "q3": q3,
            "iqr": q3-q1 if values else None}


def candidate_features(genome):
    if set(genome) != {"weights", "relief", "growth"}:
        raise ValueError("Expected the frozen three genome arrays")
    if any(len(genome[k]) != 4 for k in genome):
        raise ValueError("Genome channels must follow the four registered events")
    out = {}
    for i, event in enumerate(EVENTS):
        w, s, g = (float(genome[k][i]) for k in ("weights", "relief", "growth"))
        if not all(math.isfinite(v) for v in (w, s, g)) or not (-1 <= w <= 1 and 0 <= s <= 1 and (g == 0 or 1/640 <= g <= 1/20)):
            raise ValueError("Genome outside registered bounds")
        out.update({f"weight/{event}": w, f"relief/{event}": s,
                    f"refill_zero/{event}": int(g == 0), f"refill_rate/{event}": g})
    out["any_object_channel_positive_weight_and_refill"] = int(any(
        genome["weights"][i] > 0 and genome["growth"][i] > 0 for i in (1, 2, 3)))
    return out


def build_candidates(results: Path, per_run_csv: Path, selection_path: Path):
    """Read all search panels; use saved decoded genomes, never regenerate CMA."""
    selection = json.loads(selection_path.read_text())
    panels = {}
    with per_run_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["purpose"] not in ("search_cma", "search_random"):
                continue
            method = row["method"]
            candidate, group = int(row["candidate"]), int(row["group"])
            if method not in ("cma", "random") or row["purpose"] != "search_" + method:
                raise ValueError("Search method/purpose mismatch")
            key = row["id"]
            if Path(key).name != key:
                raise ValueError("Unsafe run key")
            marker = json.loads((results/key/"complete.json").read_text())
            raw = (results/key/"attempts"/marker["attempt"]/"spec.json").read_bytes()
            expected_path = f"runs/{key}/attempts/{marker['attempt']}/spec.json"
            if hashlib.sha256(raw).hexdigest() != marker["files"][expected_path]:
                raise ValueError("Specification file hash mismatch")
            spec = json.loads(raw)
            canonical = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if canonical != marker["spec_sha256"] or marker["id"] != key:
                raise ValueError("Frozen specification identity mismatch")
            for field, value in (("id", key), ("method", method), ("candidate", candidate), ("group", group), ("task", row["task"]), ("seed", int(row["seed"]))):
                if spec[field] != value:
                    raise ValueError("Scored row differs from frozen specification: " + field)
            area = float(row["area"])
            if not math.isfinite(area) or not 0 <= area <= 1:
                raise ValueError("Invalid recorded area")
            panel = panels.setdefault((method, candidate), {"method": method, "candidate": candidate, "group": group,
                "genome": spec["genome"], "latent": spec["latent"], "observations": {}})
            if panel["group"] != group or panel["genome"] != spec["genome"] or panel["latent"] != spec["latent"]:
                raise ValueError("Genome varies inside a search panel")
            obs = (row["task"], int(row["seed"]))
            if obs in panel["observations"]:
                raise ValueError("Duplicate task/seed observation")
            panel["observations"][obs] = area
    expected = {(method, candidate) for method in ("cma", "random") for candidate in range(120)}
    if set(panels) != expected:
        raise ValueError("Require all240 registered search candidates, not finalists or a subset")
    rows = []
    for _, panel in sorted(panels.items()):
        base = (1000 if panel["method"] == "cma" else 3000) + 3*panel["group"]
        expected_obs = {(task, base+k) for task in TASKS for k in range(3)}
        if set(panel["observations"]) != expected_obs or panel["group"] != panel["candidate"] // 10:
            raise ValueError("Incomplete or altered search panel")
        combined = float(selection["methods"][panel["method"]]["search_fitness"][str(panel["candidate"])] )
        recomputed = math.fsum(panel["observations"].values()) / 6
        if not math.isclose(combined, recomputed, abs_tol=1e-15, rel_tol=0):
            raise ValueError("Recorded combined fitness differs from its six areas")
        areas = {task: math.fsum(panel["observations"][task, base+k] for k in range(3))/3 for task in TASKS}
        row = {k: panel[k] for k in ("method", "candidate", "group", "genome", "latent")}
        row.update(features=candidate_features(panel["genome"]), areas={"combined": combined, **areas})
        rows.append(row)
    return rows


def analyze_candidates(rows):
    """All requested features/groups for pooled data and separate method strata."""
    correlations, groups = [], []
    strata = {"pooled": list(rows), "cma": [r for r in rows if r["method"] == "cma"],
              "random": [r for r in rows if r["method"] == "random"]}
    for stratum, sample in strata.items():
        features = sorted(candidate_features({"weights": [0.]*4, "relief": [.5]*4, "growth": [0.]*4}))
        for target in TARGETS:
            for feature in features:
                correlations.append({"stratum": stratum, "target": target, "feature": feature,
                    "subset": "all candidates", **spearman([r["features"][feature] for r in sample], [r["areas"][target] for r in sample])})
            for event in EVENTS:
                positive = [r for r in sample if r["features"][f"refill_rate/{event}"] > 0]
                correlations.append({"stratum": stratum, "target": target, "feature": f"refill_rate/{event}",
                    "subset": "positive refill only", **spearman([r["features"][f"refill_rate/{event}"] for r in positive], [r["areas"][target] for r in positive])})
                for sign in ("negative", "zero", "positive"):
                    signed = [r for r in sample if ("negative" if r["features"][f"weight/{event}"] < 0 else "positive" if r["features"][f"weight/{event}"] > 0 else "zero") == sign]
                    groups.append({"stratum": stratum, "target": target, "event": event, "grouping": "weight sign", "weight_sign": sign, "refill_state": "all", **summarize(r["areas"][target] for r in signed)})
                    for refill in ("zero", "nonzero"):
                        joint = [r for r in signed if (r["features"][f"refill_rate/{event}"] == 0) == (refill == "zero")]
                        groups.append({"stratum": stratum, "target": target, "event": event, "grouping": "weight sign x refill", "weight_sign": sign, "refill_state": refill, **summarize(r["areas"][target] for r in joint)})
                for refill in ("zero", "nonzero"):
                    subset = [r for r in sample if (r["features"][f"refill_rate/{event}"] == 0) == (refill == "zero")]
                    groups.append({"stratum": stratum, "target": target, "event": event, "grouping": "refill state", "weight_sign": "all", "refill_state": refill, **summarize(r["areas"][target] for r in subset)})
            for value in (0, 1):
                subset = [r for r in sample if r["features"]["any_object_channel_positive_weight_and_refill"] == value]
                groups.append({"stratum": stratum, "target": target, "event": "any_object_channel_positive_weight_and_refill", "grouping": "parameter boolean", "feature_value": value, **summarize(r["areas"][target] for r in subset)})
    return {"status": STATUS, "candidate_counts": {k: len(v) for k, v in strata.items()},
            "correlations": correlations, "groups": groups, "p_values_computed": False,
            "interpretation": "Associations in an adaptively selected search population; no causal or independent-sample inference, winner substitution, new tests or fitness changes."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--per-run-csv", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = build_candidates(args.results, args.per_run_csv, args.selection)
    output = analyze_candidates(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in (("candidates.json", rows), ("exploratory-summary.json", output)):
        (args.out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    candidate_table = [{"method": row["method"], "candidate": row["candidate"], "group": row["group"],
                        **row["features"], **{"area/"+key: value for key, value in row["areas"].items()}} for row in rows]
    for name, data in (("candidates.csv", candidate_table), ("correlations.csv", output["correlations"]), ("groups.csv", output["groups"])):
        fields = list(dict.fromkeys(k for row in data for k in row))
        with (args.out/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader();writer.writerows(data)
    print(json.dumps({"status": STATUS, "candidate_counts": output["candidate_counts"], "p_values_computed": False}))

if __name__ == "__main__":
    main()
