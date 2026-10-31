"""Figure 6: tamper presses per step and policy entropy over training, tamper on and off.

Mean over 20 seeds with 95% t-intervals; grey dashed lines are arithmetic references for a uniform policy over the
7 MiniGrid actions (press rate 1/7, entropy ln 7), not measured policies.
Regenerate: make figure6 (data: data/extracted/tamper-series.csv; plotted values: results/figure6_tamper_dynamics.csv)."""
import math
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter
from figdata import COLORS, EXTRACTED, LABELS, out, series, write_csv
from orx_figstyle import BASELINE, TEXT, band, figure_grid, mean_ci, panel_labels, save, use_style


def main():
    use_style()
    press = series(EXTRACTED / "tamper-series.csv", "method", "global_step", "presses_per_step")
    ent = series(EXTRACTED / "tamper-series.csv", "method", "global_step", "entropy_nats")
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.33)
    handles, plotted = [], []
    for ax, data, ref, reflabel, ylabel in ((axes[0], press, 1 / 7, "uniform policy (1/7)", "Tamper presses per step"),
                                            (axes[1], ent, math.log(7), "uniform policy (ln 7)", "Policy entropy (nats)")):
        for m in ("tamper_off", "tamper_on"):
            steps, ys = data[m]
            assert len(ys) == 20
            line = band(ax, np.array(steps), np.array(ys), color=COLORS[m], lw=1.2, label=LABELS[m])
            mean, lo, hi = mean_ci(np.array(ys))
            plotted += [[ylabel, m, int(s), repr(float(a)), repr(float(b)), repr(float(c)), len(ys)]
                        for s, a, b, c in zip(steps, mean, lo, hi)]
            if ax is axes[0]:
                handles.append(line)
        ax.axhline(ref, color=BASELINE, lw=0.8, ls="--", zorder=1)
        ax.annotate(reflabel, xy=(1_950_000, ref), xytext=(0, 2), textcoords="offset points", ha="right", va="bottom",
                    fontsize=6, color=BASELINE)
        ax.set(xlim=(0, 2_000_000), xlabel="Environment steps", ylabel=ylabel)
        top = max(max(np.array(data[m][1]).mean(axis=0) + 2.1 * np.array(data[m][1]).std(axis=0, ddof=1) / np.sqrt(20))
                  for m in ("tamper_off", "tamper_on"))
        ax.set_ylim(0, max(ref, top) * 1.08)  # derived from the data so no band is clipped
        ax.xaxis.set_major_locator(FixedLocator([0, 1_000_000, 2_000_000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M"))
        ax.grid(axis="y", color="#eeeeee", lw=0.5)
    panel_labels(axes)
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=6)
    save(fig, out("figure6_tamper_dynamics"))
    write_csv("figure6_tamper_dynamics", ["panel", "method", "global_step", "mean", "ci95_low", "ci95_high", "seeds"], plotted)


if __name__ == "__main__":
    main()
