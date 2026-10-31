# Found by Search, Broken by a Shortcut

Artifact for the paper **"Found by Search, Broken by a Shortcut: Transfer and Tampering of Searched Intrinsic
Rewards"** (Chase J. Metoyer, Northeastern University). The paper is [`paper/paper.pdf`](paper/paper.pdf); its
source is [`paper/paper.tex`](paper/paper.tex).

The study ran in three pre-registered stages. It searched a 12-parameter space of homeostatic intrinsic rewards with
CMA-ES and an equal-budget random search, evaluated the two selected rewards on a held-out MiniGrid task, and
retrained the CMA-ES-selected reward with a free reset action. This repository holds everything needed to check
every number in the paper: the committed result data, the scripts that turn them into the paper's numbers, tables
and figures, and a checker that verifies them.

Every number in the PDF is a hyperlink to its line in [`results/numbers.json`](results/numbers.json), and every table
and figure caption names the `make` target and the committed file that produce it. Links point at the tag
`v1.1`, so they keep pointing at the evidence after `main` moves. The repository at `v1.1` is archived on
Zenodo: [doi:10.5281/zenodo.23828842](https://doi.org/10.5281/zenodo.23828842).

## Reproduce

```bash
git clone https://github.com/chasemetoyer/found-by-search && cd found-by-search
make all      # numbers, tables, figures, paper/paper.pdf, then the number check (must print PASS)
make demo     # seconds: numbers and tables only, then the check
```

`make` uses [uv](https://docs.astral.sh/uv/) to run Python 3.12 with the pinned versions in
[`requirements.txt`](requirements.txt). Without uv: `pip install -r requirements.txt`, then
`make PY=python3 all`. The PDF needs a TeX installation with `latexmk`. On a laptop, everything except the PDF takes
well under a minute.

`make all` regenerates `results/` and `paper/` from the committed inputs in `data/`. It does not retrain anything:
the PPO training runs, which used cloud CPUs, are recorded, not re-run. Seeds, settings and code for those
runs are fixed in the pre-registration (see *Training code and pre-registration* below).

## Map from the paper to commands and files

| Paper | Command | Committed result | Inputs (in `data/`) |
|---|---|---|---|
| Every number in the text | `make numbers` | [`results/numbers.json`](results/numbers.json) (one value per line, with its source and evidence level) | `raw/`, `raw/repo/` |
| Table 1, task selection | `make table1` | [`results/table1_calibration.csv`](results/table1_calibration.csv) | `raw/s1-numbers.json` |
| Table 2, finalists | `make table2` | [`results/table2_finalists.csv`](results/table2_finalists.csv) | `raw/s2-finalists.csv` |
| Table 3, final evaluation | `make table3` | [`results/table3_final.csv`](results/table3_final.csv) | `raw/s3-cells.csv` |
| Table 4, every planned comparison | `make table4` | [`results/table4_comparisons.csv`](results/table4_comparisons.csv) | `raw/s3-comparisons.csv`, `raw/s3-fisher.csv` |
| Table 5, tamper endpoints | `make table5` | [`results/table5_tamper.csv`](results/table5_tamper.csv) | `raw/s3-tamper.json`, `raw/s3-tamper-per-seed.csv`, `raw/s3-cells.csv` |
| Table 6, frozen genomes | `make table6` | [`results/table6_genomes.csv`](results/table6_genomes.csv) | `raw/s3-genomes.csv` |
| Table 7 (App. A), registered rules and outcomes | `make table7` | [`results/table7_outcomes.csv`](results/table7_outcomes.csv) | `raw/`, `raw/repo/preregistration/` |
| Table 8 (App. B), amendments | `make table8` | [`results/table8_amendments.csv`](results/table8_amendments.csv) | `raw/repo/amendment-commits.json`, `raw/repo/preregistration/` |
| Table 9 (App. D), costs | `make table9` | [`results/table9_costs.csv`](results/table9_costs.csv) | `raw/s1-numbers.json`, `raw/s2-cost.json`, `raw/s3-cost.json`, `raw/s3-billing.json` |
| Table 10 (App. E), exploratory genome features | `make table10` | [`results/table10_exploratory.csv`](results/table10_exploratory.csv) | `raw/s2x-groups.csv`, `raw/s2x-correlations.csv` |
| Table 11 (App. F), PPO settings and versions | `make table11` | [`results/table11_hyper.csv`](results/table11_hyper.csv) | `raw/repo/handoff-cross-platform/code/final-round5/config.toml`, `raw/repo/uv.lock` |
| Figure 1, study design | `make paper` (TikZ in `paper.tex`) | counts are linked values in `results/numbers.json` | — |
| Figure 2, search fitness | `make figure2` | [`results/figure2_search.csv`](results/figure2_search.csv) | `raw/s2x-candidates.csv` |
| Figure 3, finalist re-evaluation | `make figure3` | [`results/figure3_finalists.csv`](results/figure3_finalists.csv) | `raw/s2-finalists.csv`, `raw/s2-per-run.csv` |
| Figure 4, learning curves (held-out task; tamper) | `make figure4` | [`results/figure4_heldout_tamper.csv`](results/figure4_heldout_tamper.csv) (seed means) | `raw/s3-learning-curves.csv` (every seed) |
| Figure 5, per-seed curve areas | `make figure5` | [`results/figure5_per_seed.csv`](results/figure5_per_seed.csv) | `raw/s3-per-seed.csv` |
| Figure 6, tamper presses and entropy | `make figure6` | [`results/figure6_tamper_dynamics.csv`](results/figure6_tamper_dynamics.csv) (means, 95% intervals) | `extracted/tamper-series.csv` |
| Figure 7, genome comparison | `make figure7` | [`results/figure7_genomes.csv`](results/figure7_genomes.csv) | `raw/s3-genomes.csv` |
| All of the above | `make check` | prints `PASS` | — |

Headline claims and where to find them:

- **Transfer (registered primary family, §4.3).** The three tests of E against task-only PPO, episodic novelty and R
  on KeyCorridorS3R2 are rows 1–3 of `results/table4_comparisons.csv`; keys `kc.ET.*`, `kc.EN.*`, `kc.ER.*` in
  `results/numbers.json`.
- **The satiation reference (S), §4.3.** `results/table3_final.csv` (rows for S); keys `kc.S.*`, `up.EminusS`.
- **Training tasks (registered secondary family, §4.4).** Rows 4–9 of `results/table4_comparisons.csv`.
- **Tamper collapse (§4.5).** `results/table5_tamper.csv`; keys `tam.*`.

## Layout

```
README.md            this file
LICENSE              MIT, for the code
LICENSE-DATA         CC BY 4.0, for data/, results/ and paper/
CITATION.cff         how to cite this artifact
requirements.txt     pinned Python dependencies for the paper pipeline (Python 3.12)
Makefile             one target per table and figure
data/
  raw/               the saved Stage 1-3 result files the paper reads (SHA256SUMS, SOURCES.json)
    repo/            pinned copies of the registration, receipt and code files the numbers cite
  extracted/         values re-extracted from the full Stage 3 run logs (independent check; Figure 6 series)
scripts/             make_numbers, make_tables, figure2-7, check_numbers, and the snapshot/extraction steps
results/             committed outputs: numbers.json, one CSV per table and per figure
paper/               paper.tex, paper.pdf, references.bib, generated numbers.tex, tables/, figs/
research/            training code, locked pre-registration, amendments and receipts (make verify-lock)
```

`data/raw` holds byte-for-byte copies of the saved result files, except that absolute local paths (run directory
names and the like, never numbers) were replaced by placeholders such as `<openresearch>/`. `SOURCES.json` records
the SHA256 of each original and of each copy. `paper/worded-counts.json` records the evidence for every count the paper
writes as a word ("three stages"), which the checker cannot trace automatically.

## Training code and pre-registration

[`research/`](research/) holds the study's code, locked protocols, amendments, public lock receipts and frozen
rewards, copied from commit `7acab8c` of the project's working repository
([github.com/chasemetoyer/evolved-intrinsic-rewards](https://github.com/chasemetoyer/evolved-intrinsic-rewards),
branch `orx/stage-3-modal-launch-path-and-frozen-evaluation`). The Stage 2 protocol in
`research/preregistration/stage2.md`, which also registered every Stage 3 test, was locked in commit `f757692` and
pushed publicly before any search result existed.

`make verify-lock` checks every file the Stage 1 and Stage 2 locks hashed: each must match its registered SHA256 or a
replacement hash recorded by a later amendment. To keep those hashes valid, the hashed files are byte-identical copies,
and 13 of them still contain absolute paths from the machine that ran the study. Other copied files had such paths
replaced by placeholders. The training environment is pinned in `research/uv.lock`; running it needs a Modal account.

Two inputs are not in this repository: the full Stage 3 run archive (per-run logs and policy checkpoints), from which
`data/raw` and `data/extracted` were taken, and the cloud setup that ran the training. `make snapshot extract`
rebuilds `data/` from that archive and needs the working repository's history.

## Status against the open-science checklist

- [x] Repository URL on page 1 of the paper (footnote), as a full URL.
- [x] Every number, table and figure links to its file at a pinned tag (`v1.1`).
- [x] Every number appears in a committed file or is computed by a committed script (`make check`).
- [x] Totals and percentages are computed by scripts from committed per-item data.
- [x] Result files are named after the tables and figures they back.
- [x] This README maps each paper artifact to its command and file.
- [x] `git clone` plus `make all` regenerates `results/`.
- [x] Seeds and versions are pinned (registered seeds in the protocol; `requirements.txt`; the training environment in
  `data/raw/repo/uv.lock`).
- [x] `LICENSE` is present.
- [x] A Zenodo snapshot with a DOI, cited in the paper: [doi:10.5281/zenodo.23828842](https://doi.org/10.5281/zenodo.23828842).

## License and citation

Code: MIT ([`LICENSE`](LICENSE)). Data, results, figures and paper text: CC BY 4.0
([`LICENSE-DATA`](LICENSE-DATA)). Until the paper has an arXiv identifier, cite it as: Chase J. Metoyer, *Found by
Search, Broken by a Shortcut: Transfer and Tampering of Searched Intrinsic Rewards*, preprint, 2026,
<https://github.com/chasemetoyer/found-by-search> (tag `v1.1`), doi:10.5281/zenodo.23828842.

