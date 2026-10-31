"""Figure 3: finalist search fitness against re-evaluated fitness, both methods.

Regenerate: make figure3 (data: data/raw/s2-finalists.csv, s2-per-run.csv; plotted values: results/figure3_finalists.csv)."""
from collections import Counter
from figdata import COLORS, SRC, out, rows, write_csv
from orx_figstyle import BASELINE, COLUMN, figure, save, use_style


# Label offsets (points) chosen so candidate labels do not collide; layout only.
OFFSETS = {"92": (4, -8), "113": (-4, 4), "94": (4, -3), "57": (-4, 4), "116": (4, -7), "103": (4, 3)}


def main():
    use_style()
    fin = rows(SRC / "s2-finalists.csv")
    runs = rows(SRC / "s2-per-run.csv")
    per = Counter((r["method"], r["candidate"], r["purpose"].startswith("search_finalist")) for r in runs)
    n_search = {per[(r["method"], r["candidate"], False)] for r in fin}
    n_reeval = {per[(r["method"], r["candidate"], True)] for r in fin}
    assert len(n_search) == 1 and len(n_reeval) == 1
    fig, ax = figure(width=COLUMN, ratio=0.72)
    lo, hi = 0.45, 0.75
    ax.plot([lo, hi], [lo, hi], color=BASELINE, lw=0.8, ls="--", zorder=1)
    ax.annotate("re-evaluated = search", xy=(0.515, 0.523), fontsize=6, color=BASELINE, rotation=45, ha="center",
                va="center", backgroundcolor="white")
    for m, marker, name in (("cma", "o", "CMA-ES finalists"), ("random", "s", "random-search finalists")):
        sel = [r for r in fin if r["method"] == m]
        for i, r in enumerate(sel):
            x, y = float(r["search_fitness"]), float(r["reevaluated_fitness"])
            win = (m, r["candidate"]) in (("cma", "113"), ("random", "57"))
            ax.scatter([x], [y], marker=marker, s=26 if win else 18, color=COLORS[m] if win else "white",
                       edgecolors=COLORS[m], linewidths=1.0, zorder=3, label=name if i == 0 else None)
            tag = r["candidate"] + (" = E" if (m, r["candidate"]) == ("cma", "113") else " = R" if win else "")
            dx, dy = OFFSETS.get(r["candidate"], (4, 3))
            ax.annotate(tag, xy=(x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6, color=COLORS[m],
                        ha="right" if dx < 0 else "left")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Search fitness (mean of {next(iter(n_search))} runs)")
    ax.set_ylabel(f"Re-evaluated fitness (mean of {next(iter(n_reeval))} runs)")
    ax.legend(loc="upper left", fontsize=6, frameon=True, framealpha=1.0, edgecolor="black")
    save(fig, out("figure3_finalists"))
    write_csv("figure3_finalists", ["method", "candidate", "search_fitness", "reevaluated_fitness", "selected"],
              [[r["method"], r["candidate"], r["search_fitness"], r["reevaluated_fitness"],
                (r["method"], r["candidate"]) in (("cma", "113"), ("random", "57"))] for r in fin])


if __name__ == "__main__":
    main()
