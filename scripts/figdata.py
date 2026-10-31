"""Shared loaders, colours and labels for the paper figures.

Inputs come from data/raw (snapshots) and data/extracted; each figure is drawn into paper/figs/ and the exact values it
plots are written to results/figureN_<name>.csv."""
import csv
import os

import matplotlib

# Reproducible figure files: fixed embedded dates and fixed SVG element ids, so a rebuild changes no bytes.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1793404800")
matplotlib.rcParams["svg.hashsalt"] = "found-by-search"
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC, EXTRACTED = DATA / "raw", DATA / "extracted"
FIGS, RESULTS = ROOT / "paper" / "figs", ROOT / "results"


def out(name):
    """Output stem for a figure: paper/figs/<name> (.pdf for LaTeX, .svg for preview)."""
    FIGS.mkdir(parents=True, exist_ok=True)
    return str(FIGS / name)


def write_csv(name, header, table):
    """Write the values a figure plots to results/<name>.csv."""
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(table)
# Okabe-Ito colours as in the registered Stage 3 figure; T is the grey baseline and "E, tamper on" is black, a
# colour no arm in the held-out panel uses.
COLORS = {"E": "#0072B2", "R": "#009E73", "T": "#7F7F7F", "N": "#E69F00", "S": "#CC79A7",
          "tamper_off": "#0072B2", "tamper_on": "#000000", "cma": "#0072B2", "random": "#009E73"}
LABELS = {"E": "E (CMA-ES-selected)", "R": "R (random-search-selected)", "T": "T (task-only)",
          "N": "N (episodic novelty)", "S": r"S (satiation)$^\dagger$",
          "tamper_off": "E, tamper off", "tamper_on": "E, tamper on"}
SHORT = {"E": "E", "R": "R", "T": "T", "N": "N", "S": "S", "tamper_off": "off", "tamper_on": "on"}


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def series(path, key, xcol, ycol, seedcol="seed"):
    """-> {key value: (x list, {seed: y list})} for long-format CSVs."""
    out = defaultdict(lambda: defaultdict(list))
    xs = defaultdict(list)
    for r in rows(path):
        out[r[key]][r[seedcol]].append((float(r[xcol]), float(r[ycol])))
    res = {}
    for k, seeds in out.items():
        curves = [sorted(v) for v in seeds.values()]
        grid = [x for x, _ in curves[0]]
        assert all([x for x, _ in c] == grid for c in curves), k
        res[k] = (grid, [[y for _, y in c] for c in curves])
    return res
