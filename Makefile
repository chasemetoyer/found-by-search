# One target per table and figure in the paper. `make all` rebuilds everything from the committed data in data/.
# Runtime: well under a minute for the numbers, tables and figures on a laptop; the PDF needs a TeX installation
# with latexmk. Python dependencies are pinned in requirements.txt (Python 3.12); uv installs them on the fly.
PY      ?= uv run --isolated --no-project --python 3.12 --with-requirements requirements.txt python
LATEXMK ?= latexmk -pdf -interaction=nonstopmode -halt-on-error

TABLES  := table1 table2 table3 table4 table5 table6 table7 table8 table9 table10 table11
FIGURES := figure2 figure3 figure4 figure5 figure6 figure7

.PHONY: all demo numbers tables $(TABLES) figures $(FIGURES) paper check verify-lock snapshot extract clean

all: numbers tables figures paper check verify-lock  ## everything under results/ and paper/, then the number check

demo: numbers tables check                   ## seconds: regenerates every number and table and verifies them

numbers:                                     ## results/numbers.json and paper/numbers.tex (every value in the paper)
	$(PY) scripts/make_numbers.py

tables: numbers                              ## all tables: paper/tables/*.tex and results/table*.csv
	$(PY) scripts/make_tables.py

$(TABLES): numbers                           ## one table, e.g. make table3
	$(PY) scripts/make_tables.py --table $(subst table,,$@)

figures: $(FIGURES)

figure2: ; $(PY) scripts/figure2_search.py
figure3: ; $(PY) scripts/figure3_finalists.py
figure4: ; $(PY) scripts/figure4_heldout_tamper.py
figure5: ; $(PY) scripts/figure5_per_seed.py
figure6: ; $(PY) scripts/figure6_tamper_dynamics.py
figure7: ; $(PY) scripts/figure7_genomes.py

paper:                                       ## paper/paper.pdf (Figure 1 is drawn in paper.tex with TikZ)
	cd paper && SOURCE_DATE_EPOCH=1793404800 FORCE_SOURCE_DATE=1 $(LATEXMK) paper.tex

check:                                       ## verifies every number, table, link and file mapping; must print PASS
	$(PY) scripts/check_numbers.py

verify-lock:                                 ## pre-registration: every locked file matches its registered hash
	python3 scripts/verify_lock.py

# The next two need the full Stage 3 run archive, which is not in this repository:
#   EIR_ARTIFACTS=/path/to/artifacts make snapshot extract
snapshot:                                    ## re-copy data/raw from the artifacts and the research branch
	$(PY) scripts/snapshot_sources.py --refresh

extract:                                     ## data/extracted/ from the raw Stage 3 run logs
	$(PY) scripts/extract_series.py
	$(PY) scripts/extract_claims.py

clean:
	cd paper && latexmk -C paper.tex
