"""Extract per-seed tamper series for Figure 6 from the verified Stage 3 run folders (read-only).

For each of the 40 tamper runs (DoorKey-8x8, seeds 2020-2039, tamper on and off) this reads the logged PPO policy
entropy from metrics.jsonl and the windowed presses per step from descriptive-windows.jsonl, and writes
data/extracted/tamper-series.csv. It then checks that the seed means reproduce the snapshot
s3-tamper-windows.csv (written by the Stage 3 analysis) to 1e-9.

Usage (repo root): EIR_ARTIFACTS=<artifacts directory> python3 scripts/extract_series.py   (or: make extract)
"""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("EIR_ARTIFACTS", "")) / "stage3-report" / "data" / "runs"  # verified Stage 3 run folders


def main():
    if not RUNS.is_dir():
        raise SystemExit("Set EIR_ARTIFACTS to the project's artifacts directory")
    rows = []
    for method in ("tamper_off", "tamper_on"):
        for seed in range(2020, 2040):
            run = RUNS / f"stage3_science-{method}-DoorKey-8x8-2000000-seed{seed}"
            att = run / "attempts" / json.loads((run / "complete.json").read_text())["attempt"]
            metrics = [json.loads(l) for l in (att / "metrics.jsonl").read_text().splitlines() if l]
            windows = {w["global_step"]: w for w in map(json.loads, (att / "descriptive-windows.jsonl").read_text().splitlines())}
            assert len(metrics) == 98 and set(windows) == {m["global_step"] for m in metrics}
            for m in metrics:
                rows.append({"method": method, "seed": seed, "global_step": m["global_step"], "entropy_nats": m["entropy"],
                             "presses_per_step": windows[m["global_step"]]["presses_per_step"]})
    out = ROOT / "data" / "extracted" / "tamper-series.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ref = list(csv.DictReader(open(ROOT / "data" / "raw" / "s3-tamper-windows.csv")))
    for r in ref:
        sel = [x for x in rows if x["method"] == r["method"] and x["global_step"] == int(r["global_step"])]
        assert len(sel) == 20
        for col, refcol in (("entropy_nats", "mean_entropy_nats"), ("presses_per_step", "mean_presses_per_step")):
            assert abs(sum(x[col] for x in sel) / 20 - float(r[refcol])) < 1e-9, (r, col)
    print(f"{len(rows)} rows -> {out.relative_to(ROOT)}; seed means match s3-tamper-windows.csv")


if __name__ == "__main__":
    main()
