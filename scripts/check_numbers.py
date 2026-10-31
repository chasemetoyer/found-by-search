"""Check every number in paper/paper.tex against the sources. Reports problems; exit code 1 if any.

Checks, in order:
  1. Regenerate the number macros (make_numbers.py) and every table (make_tables.py) in memory and compare them with
     the files on disk (paper/numbers.tex, results/numbers.json, paper/tables/*.tex, results/table*.csv).
  2. Verify the snapshots in data/raw (and data/raw/repo) against data/raw/SHA256SUMS.
  3. Cross-check headline values against the independent extraction data/extracted/independent-extraction.json (extract_claims.py, a
     different code path that recomputes from the raw run logs), comparing formatted strings.
  4. Every \\val{key} used in paper.tex is defined.
  5. Scan the document for literal numbers, including TikZ node text. A literal passes only if it is a declared
     design constant (ALLOWED, with its justification) or sits inside a declared structural phrase (PHRASES).
  6. Bibliography: every entry has a DOI, an arXiv id or a verified URL; every cited key exists; every arXiv id
     appears in paper.bbl; the preprint entry still carries its placeholder note.
  7. Banned phrases (terminology and claims the user ruled out) fail the check. Causal words in the exploratory and
     discussion text are listed for review by hand.
  8. Main text length: the references, which precede the appendices, must start on page 12-16 (from paper.aux).
  9. Counts written as words cannot be traced automatically. Each occurrence, identified by its surrounding words,
     must appear in paper/worded-counts.json with the evidence confirmed by hand; "TO FILL" fails.
     `--confirm-worded` adds new occurrences with evidence "TO FILL", to be replaced by hand.
 10. The fixed-satiation arm (S) appears in the abstract, the final-evaluation table and the main figure.
 11. Open-science layout: each table and figure number in paper.aux matches its results/tableN_* or figureN_* file,
     every \src data file exists, every results/ file is listed in README.md, every value's link targets its own
     line of results/numbers.json, and LICENSE is present.

A pass means every digit in the document is a generated value or a declared constant, every generated value matches
its source, and every worded count was confirmed by hand. It does not check that the words around a number describe
it correctly; that is what reading is for.

Usage (repo root): make check
"""
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_numbers as M  # noqa: E402
import make_tables  # noqa: E402

ROOT, SRC, PAPER, RESULTS = M.ROOT, M.SRC, M.PAPER, M.RESULTS
EXTRACTED = ROOT / "data" / "extracted"
problems, notes = [], []


def bad(msg):
    problems.append(msg)


# ------------------------------------------------------------------ 1. regenerate macros and tables
M.V.clear()
tex, js = M.main()
if (PAPER / "numbers.tex").read_text() != tex:
    old, new = set((PAPER / "numbers.tex").read_text().splitlines()), set(tex.splitlines())
    for line in sorted(old - new)[:20]:
        bad(f"numbers.tex has a stale line: {line}")
    for line in sorted(new - old)[:20]:
        bad(f"numbers.tex is missing regenerated line: {line}")
if (RESULTS / "numbers.json").read_text() != js:
    bad("results/numbers.json differs from the regenerated values")
vals = {k: v["text"] for k, v in M.V.items()}
written = make_tables.main()
for rel, text in written.items():
    disk = PAPER / rel if rel.startswith("tables/") else ROOT / rel
    if not disk.exists():
        bad(f"{rel}: generated but missing on disk")
    elif disk.read_text() != text:
        bad(f"{rel}: file on disk differs from regenerated output")
for f in list((PAPER / "tables").glob("*.tex")):
    if f"tables/{f.name}" not in written:
        bad(f"tables/{f.name}: on disk but no longer generated")
notes.append(f"regenerated {len(vals)} number macros and {len(written)} table files (LaTeX and CSV)")

# ------------------------------------------------------------------ 2. snapshot integrity
for line in (SRC / "SHA256SUMS").read_text().splitlines():
    h, fn = line.split(maxsplit=1)
    if hashlib.sha256((SRC / fn).read_bytes()).hexdigest() != h:
        bad(f"snapshot {fn}: sha256 differs from SHA256SUMS")

# ------------------------------------------------------------------ 3. independent cross-check
C = json.loads((EXTRACTED / "independent-extraction.json").read_text())
f, sg, fp, pct, n = M.f, M.sg, M.fp, M.pct, M.n
cross = {}
for tk, task in M.TASKS.items():
    for m in M.METHODS:
        c = C["cells"][f"{m}|{task}"]
        cross[f"{tk}.{m}.area"] = f(c["mean_area"])
        cross[f"{tk}.{m}.sd"] = f(c["sd_area"])
        cross[f"{tk}.{m}.solve"] = f"{n(c['solves'])}/{n(c['n'])}"
    for cmp_ in ("T", "N", "R"):
        c = C["comparisons"][f"{task}|{cmp_}"]
        cross[f"{tk}.E{cmp_}.diff"] = sg(c["diff"])
        cross[f"{tk}.E{cmp_}.U"] = f(c["U"], 1)
        cross[f"{tk}.E{cmp_}.p"] = fp(c["p"])
        cross[f"{tk}.E{cmp_}.ph"] = fp(c["holm"])
T = C["tamper"]
cross.update({"tam.P0": f(T["P0"], 4), "tam.P0full": repr(T["P0"]), "tam.off.area": f(T["off"]), "tam.on.area": f(T["on"]),
              "tam.diff": f(T["diff"]), "tam.U": f(T["U"], 1), "tam.p": fp(T["p"]), "tam.vsT": sg(T["vsT"]),
              "tam.solve.off": f"{n(T['solves_off'])}/20", "tam.solve.on": f"{n(T['solves_on'])}/20",
              "tam.medmin": f(min(T["window_medians"]), 1), "tam.medmax": f(max(T["window_medians"]), 1),
              "tam.off.excess": M.sci(T["tamper_off"]["max_excess"])})
for m, k in (("tamper_on", "on"), ("tamper_off", "off")):
    t = T[m]
    cross.update({f"tam.{k}.pps": pct(t["pps"]), f"tam.{k}.pps10": pct(t["pps10"]), f"tam.{k}.iret": f(t["iret"]),
                  f"tam.{k}.ent": f(t["ent"]), f"tam.{k}.ent9": f(t["ent9"]), f"tam.{k}.aboveP0": n(t["above_P0"]),
                  f"tam.{k}.above064": n(t["above_064"])})
for k, v in C["finalists"].items():
    cross[f"fin.{k}.search"] = f(v["search"])
    cross[f"fin.{k}.reeval"] = f(v["reeval"])
    cross[f"fin.{k}.change"] = sg(v["reeval"] - v["search"])
for k, v in C["exploratory"].items():
    strat, ev = k.split("|")
    for g in ("positive", "negative", "zero", "nonzero"):
        cross[f"x.{strat}.{ev}.{g}"] = f(v[g])
for task, a in C["stage1"].items():
    cross[f"s1.{task.replace('-', '')}.area"] = f(a)
for m, g in C["genomes"].items():
    for i, ev in enumerate(("newview", "pickup", "drop", "toggle")):
        cross[f"g.{m}.{ev}.w"] = f(g["weights"][i])
        cross[f"g.{m}.{ev}.s"] = f(g["relief"][i])
        cross[f"g.{m}.{ev}.g"] = "0" if g["growth"][i] == 0 else f(g["growth"][i], 4)
mism = 0
for k, want in cross.items():
    if k not in vals:
        bad(f"cross-check: {k} missing from the regenerated macros")
    elif vals[k] != want:
        mism += 1
        bad(f"cross-check: {k} = {vals[k]!r} but the independent extraction gives {want!r}")
notes.append(f"cross-checked {len(cross)} values against independent-extraction.json ({mism} mismatches)")

# ------------------------------------------------------------------ 4. every \val used is defined
main = (PAPER / "paper.tex").read_text()
body = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in main.splitlines())  # strip comments, keep \%
body = body.replace(r"\%", "PERCENT").split(r"\begin{document}")[1].split(r"\end{document}")[0]
body = re.sub(r"\\(bibliographystyle|bibliography)\{[^}]*\}", " ", body)
used = set(re.findall(r"\\(?:raw)?val\{([^}]+)\}", body))
used |= {k + ".p" for k in re.findall(r"\\pv\{([^}]+)\}", body)} | {k + ".ph" for k in re.findall(r"\\ph\{([^}]+)\}", body)}
for k in sorted(used - set(vals)):
    bad(f"paper.tex uses undefined value {k}")
notes.append(f"paper.tex uses {len(used)} distinct values; {len(set(vals) - used)} generated values are not cited in the prose")

# ------------------------------------------------------------------ 5. literal numbers (prose, captions, TikZ node text)
ALLOWED = {  # literal: justification
    "2M": "nominal environment steps per run (registered budget)", "1M": "Stage 1 calibration budget",
    "95PERCENT": "interval level", "10PERCENT": "final tenth of training, as registered", "0.64": "rounded P0 reference fixed by the descriptive amendment",
}
PHRASES = [
    r"Stages?~[1-3]", r"Stages~2--3", r"Stages 1--3", r"Stage~[1-3]'s", r"Zen~[34]", r"7\$\\times\$7", r"7 actions",
    r"seed 0\b", r"\$\[0,1\]\^4\$", r"\\in\[0,1\]\^4", r"\\\{0,1\\\}\^4", r"\$\[-1,1\]\$", r"\$\[0,1\]\$", r"\\in\(0,1\]",
    r"\$\\gamma\\in\(0,1\]\$", r"t=0", r"T-1", r"t-1", r"V_1", r"V_0", r"\(a\)", r"\(b\)", r"\(c\)", r"\(d\)",
    r"Figure~\\ref", r"cand\.\\ ", r"doi:10\.5281/zenodo\.23708846", r"SHA256", r"float(32|64)",
    # equation structure (bounds, indicator sets, signs), not measured values
    r"\[0,1\]", r"\\\{0,1\\\}", r"(max|min)\(w ,0\)", r"\$7\\times7\$", r"2z-1", r"u<0", r">0\$", r"\$1/g\$",
    r"\(1-\\gamma\)", r"\(2-s\)",
]
prose = body
prose = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", prose, flags=re.S)
prose = re.sub(r"[_^]\{[^{}]*\}", " ", prose)
prose = re.sub(r"[_^][0-9A-Za-z]", " ", prose)
for cmd in (r"\\label\{[^}]*\}", r"\\ref\{[^}]*\}", r"\\eqref\{[^}]*\}", r"\\cite[pt]?\{[^}]*\}", r"\\input\{[^}]*\}",
            r"\\includegraphics\[[^\]]*\]\{[^}]*\}", r"\\(?:raw)?val\{[^}]*\}", r"\\url\{[^}]*\}",
            r"\\src\{[^}]*\}\{[^}]*\}", r"\\pv\{[^}]*\}", r"\\ph\{[^}]*\}",
            r"\\href\{[^}]*\}", r"\\nolinkurl\{[^}]*\}", r"\\texttt\{[^}]*\}", r"\[itemsep=[^\]]*\]",
            r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", r"\[!?[htbp]+\]", r"\\setlength\{[^}]*\}\{[^}]*\}",
            r"\\(clearpage|appendix|centering|small|footnotesize|scriptsize)"):
    prose = re.sub(cmd, " ", prose)
# TikZ: drop option brackets, node names and coordinate expressions, keep node text in braces.
prose = re.sub(r"\\node\[[^\]]*\]", r"\\node ", prose)
prose = re.sub(r"\\draw\[[^\]]*\]", r"\\draw ", prose)
prose = re.sub(r"\((?:[a-z0-9]+(?:\.[a-z]+)?)\)", " ", prose)
prose = re.sub(r"\$\([^$]*\)\$", " ", prose)
prose = re.sub(r"\+\+\([^)]*\)", " ", prose)
prose = re.sub(r"\s+", " ", prose)
covered = [False] * len(prose)
for ph in PHRASES:
    for m in re.finditer(ph, prose):
        for i in range(m.start(), m.end()):
            covered[i] = True
tok = re.compile(r"(?<![A-Za-z0-9_.\\-])(\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?(?:--\d+)?(?:PERCENT|k|M)?)(?![A-Za-z0-9])")
unacc = []
for m in tok.finditer(prose):
    s = m.group(1)
    if s in ALLOWED or all(covered[m.start():m.end()]):
        continue
    unacc.append((s, prose[max(0, m.start() - 45):m.end() + 25]))
for s, ctx in unacc:
    bad(f"literal number {s!r} is neither a \\val nor a declared constant: ...{ctx}...")
notes.append(f"literal-number scan: {len(unacc)} unaccounted literals")

# ------------------------------------------------------------------ 6. bibliography
bib = (PAPER / "references.bib").read_text()
entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S)
keys = {k for k, _ in entries}
bbl = (PAPER / "paper.bbl").read_text() if (PAPER / "paper.bbl").exists() else ""
for k, b in entries:
    doi = re.search(r"\bdoi\s*=\s*\{([^}]+)\}", b)
    arx = re.search(r"\beprint\s*=\s*\{([^}]+)\}", b)
    url = re.search(r"\burl\s*=\s*\{https://(proceedings\.neurips\.cc|jmlr\.org)/[^}]+\}", b)
    if not (doi or arx or url):
        bad(f"bib entry {k} has no DOI, arXiv id or verified proceedings/JMLR URL")
    if arx and arx.group(1) not in bbl:
        bad(f"bib entry {k}: arXiv {arx.group(1)} does not appear in paper.bbl")
pre = dict(entries).get("metoyer2026satisfied", "")
if "placeholder" not in pre:
    bad("the preprint entry lost its placeholder note (replace with the arXiv id only when one exists)")
cited = set()
for c in re.findall(r"\\cite[pt]?\{([^}]+)\}", body):
    cited |= {x.strip() for x in c.split(",")}
for k in sorted(cited - keys):
    bad(f"cited key {k} is not in references.bib")
for k in sorted(keys - cited):
    notes.append(f"bib entry {k} is not cited (it will not print)")
notes.append(f"bibliography: {len(keys)} entries, {len(cited)} cited")

# ------------------------------------------------------------------ 7. banned phrases and causal words
flat = re.sub(r"\s+", " ", body)
BANNED = [r"random reward", r"evolved reward", r"CMA-ES (transfers|generalizes|is better|outperforms|beats)",
          r"\bmatched\b", r"replicat\w*\W+(?:\w+\W+){0,5}preprint", r"preprint\W+(?:\w+\W+){0,5}replicat",
          r"random-search reward\b(?!-)", r"CMA-ES reward\b", r"selection does not protect"]
for pat in BANNED:
    for m in re.finditer(pat, flat, re.I):
        bad(f"banned phrase /{pat}/: ...{flat[max(0, m.start() - 50):m.end() + 40]}...")
expl = flat[flat.find(r"\paragraph{Genome features"):flat.find(r"\section{What our")] + \
    flat[flat.find(r"\section{Discussion}"):flat.find(r"\section{Limitations}")]
causal = [(m.group(0), expl[max(0, m.start() - 60):m.end() + 40]) for m in
          re.finditer(r"\b(because|caus\w*|led to|due to|effects?|drove|explains?)\b", expl, re.I)]
for w, ctx in causal:
    notes.append(f"REVIEW causal word '{w}' in exploratory/discussion text: ...{ctx}...")

# ------------------------------------------------------------------ 8. main-text length
aux = (PAPER / "paper.aux").read_text() if (PAPER / "paper.aux").exists() else ""
m = re.search(r"\\newlabel\{sec:references\}\{\{[^}]*\}\{(\d+)\}", aux)
if not m:
    bad("paper.aux has no page for sec:references; compile first")
else:
    page = int(m.group(1))
    if not 12 <= page <= 16:
        bad(f"main text ends on page {page}; the target is 12-16 pages")
    notes.append(f"main text ends on page {page} (the references start there)")

# ------------------------------------------------------------------ 9. worded counts confirmed by hand
WORDS = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|both|once|twice|half|single|"
                   r"none|no seed|every seed|all seeds|each of the|almost every seed)\b", re.I)


def plain(x):
    x = re.sub(r"\\(val|pv|ph)\{[^}]*\}", " N ", x)
    x = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", x)
    x = re.sub(r"[{}$~\\]", " ", x)
    return re.sub(r"\s+", " ", x).strip().lower()


ptxt = plain(body)
contexts = []
for m in WORDS.finditer(ptxt):
    pre_, post = ptxt[:m.start()].split()[-3:], ptxt[m.end():].split()[:3]
    contexts.append(" ".join(pre_ + [m.group(0).lower()] + post))
wfile = PAPER / "worded-counts.json"
confirmed = json.loads(wfile.read_text()) if wfile.exists() else {}
new_ctx = sorted({c for c in contexts if c not in confirmed})
if "--confirm-worded" in sys.argv and new_ctx:
    for c in new_ctx:
        confirmed[c] = "TO FILL"
    wfile.write_text(json.dumps(dict(sorted(confirmed.items())), indent=1, ensure_ascii=False) + "\n")
    notes.append(f"added {len(new_ctx)} worded counts to {wfile.name} with evidence 'TO FILL'")
else:
    for c in new_ctx:
        bad(f"worded count not yet confirmed: '{c}'")
for c, ev in confirmed.items():
    if ev == "TO FILL" and c in contexts:
        bad(f"worded count '{c}' still has evidence TO FILL")
notes.append(f"worded counts: {len(contexts)} occurrences, {len(set(contexts))} distinct contexts")

# ------------------------------------------------------------------ 10. S in abstract, table and main figure
abstract = body.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
if "satiation" not in abstract:
    bad("the fixed-satiation result is missing from the abstract")
if "S (satiation" not in (PAPER / "tables" / "final.tex").read_text():
    bad("S is missing from the final-evaluation table")
if '"S"' not in (ROOT / "scripts" / "figure4_heldout_tamper.py").read_text():
    bad("S is missing from the main figure")

# ------------------------------------------------------------------ 11. open-science layout
NUM = {"tab:calibration": "table1_calibration", "tab:finalists": "table2_finalists", "tab:final": "table3_final",
       "tab:comparisons": "table4_comparisons", "tab:tamper": "table5_tamper", "tab:genomes": "table6_genomes",
       "tab:outcomes": "table7_outcomes", "tab:amendments": "table8_amendments", "tab:costs": "table9_costs",
       "tab:exploratory": "table10_exploratory", "tab:hyper": "table11_hyper", "fig:search": "figure2_search",
       "fig:finalists": "figure3_finalists", "fig:curves": "figure4_heldout_tamper", "fig:perseed": "figure5_per_seed",
       "fig:tamperdyn": "figure6_tamper_dynamics", "fig:genomes": "figure7_genomes"}
for lab, stem in NUM.items():
    m = re.search(r"\\newlabel\{" + re.escape(lab) + r"\}\{\{(\d+)\}", aux)
    want = re.match(r"(?:table|figure)(\d+)_", stem).group(1)
    if not m or m.group(1) != want:
        bad(f"{lab}: paper number {m.group(1) if m else '?'} but results file is {stem}.csv")
    if not (RESULTS / f"{stem}.csv").exists():
        bad(f"results/{stem}.csv is missing")
for f in re.findall(r"\\src\{[^}]*\}\{([^}]*)\}", body):
    if not (ROOT / f).exists():
        bad(f"caption cites {f}, which does not exist")
readme = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
for f in sorted(RESULTS.glob("*")):
    if f"results/{f.name}" not in readme:
        bad(f"results/{f.name} is not listed in README.md")
jlines = (RESULTS / "numbers.json").read_text().splitlines()
for k, ln in re.findall(r"\\@namedef\{line@([^}]+)\}\{(\d+)\}", (PAPER / "numbers.tex").read_text()):
    if not jlines[int(ln) - 1].startswith(json.dumps(k) + ":"):
        bad(f"value {k} links to line {ln}, which holds another value")
if not (ROOT / "LICENSE").exists():
    bad("LICENSE is missing")
notes.append(f"open-science layout: {len(NUM)} tables and figures mapped to results/, {len(jlines) - 2} linked values")

for line in notes:
    print("note:", line)
for line in problems:
    print("PROBLEM:", line)
print(f"\n{'PASS' if not problems else 'FAIL'}: {len(problems)} problem(s)")
sys.exit(1 if problems else 0)
