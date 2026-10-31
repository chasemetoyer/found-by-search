"""Figure 4: the registered Stage 3 learning-curve figure, redrawn from its data with relabelled legend entries.

Same data (s3-learning-curves.csv, written by stage3-report/figures/plot_results.py from the verified runs), same
marks and axes as the registered figure, with a new tamper-on colour and a shorter height.
Regenerate: make figure4 (data: data/raw/s3-learning-curves.csv; plotted seed means: results/figure4_heldout_tamper.csv)."""
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter
from figdata import COLORS, LABELS, SRC, out, series, write_csv
from orx_figstyle import TEXT, figure_grid, panel_labels, save, use_style


def main():
    use_style()
    data = series(SRC / "s3-learning-curves.csv", "method", "environment_steps", "training_success")
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=2.8 / TEXT, sharey=True)
    handles, plotted = [], []
    for ax, methods in zip(axes, [("E", "R", "T", "N", "S"), ("tamper_off", "tamper_on")]):
        for m in methods:
            steps, ys = data[m]
            ys = np.array(ys)
            assert ys.shape == (20, 98) and steps[-1] == 1_998_848
            for y in ys:
                ax.plot(steps, y, color=COLORS[m], alpha=0.075, lw=0.4)
            (line,) = ax.plot(steps, ys.mean(axis=0), color=COLORS[m], lw=1.5, label=LABELS[m])
            plotted += [[m, int(s), repr(float(v)), ys.shape[0]] for s, v in zip(steps, ys.mean(axis=0))]
            handles.append(line)
        ax.set(xlim=(0, 2_000_000), ylim=(0, 1.025), xlabel="Environment steps")
        ax.xaxis.set_major_locator(FixedLocator([0, 1_000_000, 2_000_000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M"))
        ax.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1]))
        ax.grid(axis="y", color="#eeeeee", lw=0.5)
    axes[0].set_ylabel("Training success")
    panel_labels(axes, labels=["(a) Held-out KeyCorridorS3R2", "(b) DoorKey-8x8 tamper retraining"])
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=6)
    save(fig, out("figure4_heldout_tamper"))
    write_csv("figure4_heldout_tamper", ["method", "environment_steps", "seed_mean_training_success", "seeds"], plotted)


if __name__ == "__main__":
    main()
