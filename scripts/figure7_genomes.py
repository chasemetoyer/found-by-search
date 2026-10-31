"""Figure 7: the frozen CMA-ES-selected (E) and random-search-selected (R) genomes.

Effective float32 parameters from the frozen winners. A hatched bar marks an event whose single occurrence empties
the need (relief >= h0) and whose need never refills (growth 0).
Regenerate: make figure7 (data: data/raw/s3-genomes.csv; plotted values: results/figure7_genomes.csv)."""
import numpy as np
from figdata import COLORS, LABELS, SRC, out, rows, write_csv
from orx_figstyle import BASELINE, TEXT, figure_grid, panel_labels, save, use_style

EVENTS = ("new_view", "pickup", "drop", "toggle")
H0 = 0.5  # checked against the locked protocol by make_numbers.py (key d.h0)


def main():
    use_style()
    gen = {(r["method"], r["event"]): r for r in rows(SRC / "s3-genomes.csv")}
    fig, axes = figure_grid(1, 3, width=TEXT, ratio=0.31)
    x = np.arange(len(EVENTS))
    w = 0.36
    for ax, col, ylabel, scale in ((axes[0], "weight_effective_float32", "Weight", 1),
                                   (axes[1], "relief_effective_float32", "Relief per event", 1),
                                   (axes[2], "growth_effective_float32", r"Growth per step ($\times10^{-3}$)", 1e3)):
        for i, m in enumerate(("E", "R")):
            vals = np.array([float(gen[(m, e)][col]) * scale for e in EVENTS])
            once = [float(gen[(m, e)]["relief_effective_float32"]) >= H0 and float(gen[(m, e)]["growth_effective_float32"]) == 0
                    for e in EVENTS]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, color=COLORS[m], edgecolor="black", linewidth=0.4,
                          label=LABELS[m] if ax is axes[0] else None)
            for b, o in zip(bars, once):
                if o:
                    b.set_hatch("////")
                    b.set_edgecolor("white") if False else None
            if col.startswith("growth"):
                for xi, v in zip(x + (i - 0.5) * w, vals):
                    if v == 0:
                        ax.annotate("0", xy=(xi, 0), xytext=(0, 1.5), textcoords="offset points", ha="center",
                                    va="bottom", fontsize=5.5, color=COLORS[m])
        ax.set_xticks(x, [e.replace("_", " ") for e in EVENTS])
        ax.tick_params(axis="x", length=0, labelsize=6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#eeeeee", lw=0.5)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_ylim(-1, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axhline(H0, color=BASELINE, lw=0.8, ls="--")
    axes[1].annotate("$h_0$", xy=(3.45, H0), xytext=(0, 2), textcoords="offset points", ha="right", va="bottom",
                     fontsize=6, color=BASELINE)
    axes[2].set_ylim(0, 10)
    panel_labels(axes)
    fig.legend(loc="outside lower center", ncol=2, frameon=False, fontsize=6)
    save(fig, out("figure7_genomes"))
    write_csv("figure7_genomes", ["method", "event", "weight", "relief", "growth_per_step", "spent_by_one_event"],
              [[m, e, gen[(m, e)]["weight_effective_float32"], gen[(m, e)]["relief_effective_float32"],
                gen[(m, e)]["growth_effective_float32"],
                float(gen[(m, e)]["relief_effective_float32"]) >= H0 and float(gen[(m, e)]["growth_effective_float32"]) == 0]
               for m in ("E", "R") for e in EVENTS])


if __name__ == "__main__":
    main()
