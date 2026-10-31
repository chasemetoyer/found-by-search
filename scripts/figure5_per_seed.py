"""Figure 5: per-seed training-curve areas for every task and method, and for the tamper retraining.

Filled markers are solved seeds (final-window mean success >= 0.9); the bar is the seed mean.
Regenerate: make figure5 (data: data/raw/s3-per-seed.csv; plotted values: results/figure5_per_seed.csv)."""
import numpy as np
from figdata import COLORS, SHORT, SRC, out, rows, write_csv
from orx_figstyle import TEXT, save, use_style
import matplotlib.pyplot as plt

PANELS = [("KeyCorridorS3R2", ("E", "R", "T", "N", "S"), "(a) KeyCorridorS3R2, held out"),
          ("DoorKey-8x8", ("E", "R", "T", "N", "S"), "(b) DoorKey-8x8"),
          ("UnlockPickup", ("E", "R", "T", "N", "S"), "(c) UnlockPickup"),
          ("DoorKey-8x8", ("tamper_off", "tamper_on"), "(d) Tamper")]


def main():
    use_style()
    data = rows(SRC / "s3-per-seed.csv")
    fig, axes = plt.subplots(1, 4, figsize=(TEXT, TEXT * 0.33), layout="constrained", sharey=True,
                             gridspec_kw={"width_ratios": [5, 5, 5, 2.3]})
    rng = np.random.default_rng(0)  # horizontal jitter only
    plotted = []
    for ax, (task, methods, title) in zip(axes, PANELS):
        for i, m in enumerate(methods):
            sel = [r for r in data if r["task"] == task and r["method"] == m]
            assert len(sel) == 20, (task, m, len(sel))
            y = np.array([float(r["curve_area"]) for r in sel])
            solved = np.array([r["solved"] == "True" for r in sel])
            x = i + rng.uniform(-0.18, 0.18, len(y))
            ax.scatter(x[solved], y[solved], s=7, color=COLORS[m], linewidths=0, zorder=3)
            ax.scatter(x[~solved], y[~solved], s=7, facecolors="white", edgecolors=COLORS[m], linewidths=0.6, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [y.mean()] * 2, color="black", lw=1.0, zorder=4)
            plotted += [[task, m, r["seed"], r["curve_area"], r["solved"], repr(float(y.mean()))] for r in sel]
        ax.set_xticks(range(len(methods)), [SHORT[m] for m in methods])
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_ylim(-0.02, 1.0)
        ax.grid(axis="y", color="#eeeeee", lw=0.5)
        ax.tick_params(axis="x", length=0)
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                    ha="left", va="bottom", fontweight="bold", fontsize=plt.rcParams["axes.labelsize"])
    axes[0].set_ylabel("Training-curve area per seed")
    for ax in axes:
        ax.set_xlabel("Method" if ax is not axes[3] else "Tamper")
    save(fig, out("figure5_per_seed"))
    write_csv("figure5_per_seed", ["task", "method", "seed", "curve_area", "solved", "seed_mean"], plotted)


if __name__ == "__main__":
    main()
