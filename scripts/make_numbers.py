"""Generate paper/numbers.tex and results/numbers.json.

Every measured or computed value quoted in paper.tex is a macro \\val{key} defined here. Values are read from
the snapshots in data/raw (see SOURCES.json) or from the pinned copies of registration and code files in data/raw/repo
(locked protocols, amendments, public receipts, frozen winners, code). Each key records its source, its selector
and its evidence level, following the STOP 1 claims ledger. check_numbers.py regenerates this output, fails on
any difference, cross-checks headline values against an independent extraction, and scans paper.tex for
literal numbers.

Usage (repo root): make numbers   (or: python3 scripts/make_numbers.py)

results/numbers.json holds one value per line, so the paper can link each number to its line.
"""
import csv
import json
import re
import tomllib
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw"
REPO = SRC / "repo"          # pinned copies of research-branch files (registration, code, receipts)
RESULTS, PAPER = ROOT / "results", ROOT / "paper"
V = {}
LEVELS = {"REG": "REGISTERED", "PSD": "PRE-SPECIFIED DESCRIPTIVE", "EXP": "EXPLORATORY/POST HOC",
          "REC": "RECORD", "DER": "DERIVED REFERENCE"}
TASKS = {"kc": "KeyCorridorS3R2", "dk": "DoorKey-8x8", "up": "UnlockPickup"}
METHODS = ("E", "R", "T", "N", "S")


# ------------------------------------------------------------------ readers
def J(path):
    return json.loads(Path(path).read_text())


def C(name):
    with open(SRC / name, newline="") as fh:
        return list(csv.DictReader(fh))


def one(rows, **kw):
    hits = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    assert len(hits) == 1, (kw, len(hits))
    return hits[0]


def text(rel, pattern):
    path = SRC / rel[1:] if rel.startswith("@") else REPO / rel
    m = re.search(pattern, path.read_text(), re.S)
    assert m, (rel, pattern)
    return m.groups() if len(m.groups()) > 1 else m.group(1)


# ------------------------------------------------------------------ formats
def fp(p):
    """p-value formatting, as in the preprint: 2 significant digits below 0.001, else 3 decimals."""
    if p is None:
        return "--"
    if p < 0.001:
        s = f"{p:.2g}"
        if "e" not in s:
            return s
        m, e = s.split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def neg(s):
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def f(x, d=3):
    return neg(f"{float(x):.{d}f}")


def sg(x, d=3):
    """Signed difference: +0.158 or -0.031."""
    return neg(f"{float(x):+.{d}f}")


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}\\%"


def n(x):
    return f"{int(round(float(x))):,}"


def usd(x, d=2):
    return f"\\${float(x):,.{d}f}"


def sci(x, d=2):
    m, e = f"{float(x):.{d - 1}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def utc(s):
    t = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    return f"{t.day} {t:%B} {t:%Y}, {t:%H:%M:%S}~UTC"


def hm(s):
    t = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    return f"{t:%H:%M:%S}~UTC"


def steps(x):
    x = float(x)
    return f"{x / 1e6:.2f}M" if x < 2e6 else "2M (censored)"


def put(key, value, src, level="REC"):
    assert re.fullmatch(r"[A-Za-z0-9.\-]+", key), key
    assert key not in V, key
    assert level in LEVELS, level
    V[key] = {"text": str(value), "source": src, "level": LEVELS[level]}


# ------------------------------------------------------------------ build
def design():
    s2 = J(SRC / "s2-search-summary.json")
    runs = C("s2-per-run.csv")
    cnt = lambda purpose: sum(r["purpose"] == purpose for r in runs)
    put("s2.runs", n(s2["runs_recorded"]), "s2-search-summary.json :: runs_recorded")
    put("s2.search", n(cnt("search_cma") + cnt("search_random")), "s2-per-run.csv purpose in {search_cma, search_random}")
    put("s2.searchcma", n(cnt("search_cma")), "s2-per-run.csv purpose=search_cma")
    put("s2.searchrand", n(cnt("search_random")), "s2-per-run.csv purpose=search_random")
    put("s2.reeval", n(cnt("search_finalist_cma") + cnt("search_finalist_random")), "s2-per-run.csv finalist purposes")
    put("s2.reevalper", n(cnt("search_finalist_cma")), "s2-per-run.csv purpose=search_finalist_cma")
    put("s2.cands", n(s2["cma_candidates"]), "s2-search-summary.json :: cma_candidates")
    assert s2["cma_candidates"] == s2["random_candidates"]
    put("s2.candtotal", n(s2["cma_candidates"] + s2["random_candidates"]), "s2-search-summary.json :: cma + random candidates")
    put("s2.gens", n(s2["cma_generations_asked"]), "s2-search-summary.json :: cma_generations_asked")
    put("s2.finalists", n(s2["finalists_per_method"]["cma"]), "s2-search-summary.json :: finalists_per_method.cma")
    put("s2.conc", n(s2["max_concurrent_calls"]), "s2-search-summary.json :: max_concurrent_calls")
    prot = "preregistration/stage2.md"
    pop, gens = text(prot, r"population (\d+), (\d+) generations")
    assert int(gens) == s2["cma_generations_asked"]
    put("s2.pop", pop, prot + " ~ population (N)")
    mean0, sigma = text(prot, r"z = (0\.5) and sigma0 = (0\.25)")
    put("s2.mean0", mean0, prot + " ~ initial mean")
    put("s2.sigma", sigma, prot + " ~ sigma0")
    put("s2.cmaseed", text(prot, r"Optimizer RNG seed (\d+)"), prot + " ~ optimizer RNG seed")
    put("s2.randseed", text(prot, r"default_rng\((\d+)\)"), prot + " ~ random-search RNG")
    put("s2.searchseeds", text(prot, r"over the two\s+training tasks × (\d) seeds"), prot + " ~ seeds per task per candidate")
    put("s2.reevalseeds", text(prot, r"re-evaluated on (\d+) fresh seeds"), prot + " ~ finalist seeds per task")
    for key, pat in {"cma": r"→ (1000–1035)", "rand": r"→ (3000–3035)", "fincma": r"CMA-ES (1900–1909)",
                     "finrand": r"random search (3900–3909)", "final": r"shared by every method \| (2000–2019)",
                     "tamper": r"Tamper test \(Stage 3\) \| (2020–2039)"}.items():
        put(f"seeds.{key}", text(prot, pat).replace("–", "--"), prot + " ~ seed table")
    put("s2.cap0", usd(text(prot, r"Stage 2 ledger with the \$(\d+) cap"), 0), prot + " ~ original Stage 2 cap")
    put("s3.cap0", usd(text(prot, r"a separate \$(\d+) cap"), 0), prot + " ~ original Stage 3 cap")
    h0 = text(prot, r"h0 = (0\.5), a")
    put("d.h0", h0, prot + " ~ h0")
    lo, hi = text(prot, r"g ∈ \{0\} ∪ \[(1/\d+), (1/\d+)\]")
    put("d.gmin", lo, prot + " ~ growth lower bound")
    put("d.gmax", hi, prot + " ~ growth upper bound")
    put("d.coef", text(prot, r"intrinsic coefficient (1\.0)"), prot + " ~ intrinsic coefficient")
    put("d.resamples", text(prot, r"permutation with ties, ([\d,]+)\s+resamples"), prot + " ~ permutation resamples")
    put("d.meaningful", text(prot, r"comparator\) >= (0\.05)"), prot + " ~ meaningful gate")
    put("d.delta", text(prot, r"noninferiority with delta = (0\.05)"), prot + " ~ unused noninferiority margin")
    amend = "preregistration/stage3-descriptive-amendment-20261029.md"
    put("lab.gap", text(amend, r"`gap <= -(0\.05)`"), amend + " ~ wireheading-like area rule", "PSD")
    put("lab.press", text(amend, r"strictly greater than (1) in"), amend + " ~ wireheading-like press rule", "PSD")
    put("lab.shaping", text(amend, r"at most (2) in"), amend + " ~ lost-shaping press rule", "PSD")
    put("lab.final", text(amend, r"starts at `(0\.9) \* actual_steps`"), amend + " ~ final-ten-percent start", "PSD")
    res = int(V["d.resamples"]["text"].replace(",", ""))
    put("d.pfloor", fp(1 / (res + 1)), "arithmetic: 1/(resamples + 1), the smallest attainable permutation p", "DER")
    put("d.ntrain", n(len({r["task"] for r in runs})), "s2-per-run.csv :: distinct training tasks")
    s1p = "preregistration/stage1.md"
    put("s1.easy", "0" + text(s1p, r"at 1M is\s+<=(\.85) AND"), s1p + " ~ training-task upper threshold", "REG")
    lo1, hi1 = text(s1p, r"in \[(\.05),(\.85)\],")
    put("s1.hard", "0" + lo1, s1p + " ~ held-out lower threshold", "REG")
    assert "0" + hi1 == V["s1.easy"]["text"]
    put("d.alpha", "0" + text(prot, r"Holm-adjusted p < (\.05)\."), prot + " ~ hit threshold", "REG")
    put("d.window", text("handoff-cross-platform/code/final-round5/ppo.py", r"recent = deque\(maxlen=(\d+)\)"),
        "final-round5/ppo.py ~ rolling window of completed episodes behind train/success")
    defs = J(SRC / "s3-definitions.json")
    put("s3.runs", n(defs["number_of_runs"]), "s3-definitions.json :: number_of_runs")
    cells = C("s3-cells.csv")
    final = sum(int(r["n"]) for r in cells if not r["method"].startswith("tamper"))
    tam = sum(int(r["n"]) for r in cells if r["method"].startswith("tamper"))
    assert final + tam == defs["number_of_runs"]
    put("s3.final", n(final), "s3-cells.csv :: sum n (E,R,T,N,S)")
    put("s3.tamper", n(tam), "s3-cells.csv :: sum n (tamper)")
    seeds = {int(r["n"]) for r in cells}
    assert len(seeds) == 1
    put("s3.seeds", n(seeds.pop()), "s3-cells.csv :: n per cell")
    put("d.ntasks", n(len({r["task"] for r in cells if not r["method"].startswith("tamper")})), "s3-cells.csv :: distinct tasks")
    put("d.actual", n(defs["actual_steps"]), "s3-definitions.json :: actual_steps")
    put("d.rows", n(defs["rows_per_run"]), "s3-definitions.json :: rows_per_run")
    nf, thr = text("@s3-definitions.json", r"floor\(98/10\)=(\d+) loggedtrain/success rows>=(0\.9)")
    put("d.finalrows", nf, "s3-definitions.json :: solved")
    put("d.solvethr", thr, "s3-definitions.json :: solved")
    put("d.holm1", n(defs["family_sizes"]["primary"]), "s3-definitions.json :: family_sizes.primary")
    put("d.holm2", n(defs["family_sizes"]["secondary"]), "s3-definitions.json :: family_sizes.secondary")
    put("d.Sgrowth", "1/" + text("stage3/specs.py", r"growth=\[1/(\d+)\]\*3"), "stage3/specs.py ~ S growth")
    put("d.Ncoef", r"$10^{-3}$" if text("stage3/specs.py", r"'episodic_count','coef':(\.001)") == ".001" else "?",
        "stage3/specs.py ~ N coefficient")
    cfg = tomllib.loads((REPO / "handoff-cross-platform/code/final-round5/config.toml").read_text())
    put("ppo.envs", n(cfg["env"]["num_envs"]), "final-round5/config.toml :: env.num_envs")
    put("ppo.rollout", n(cfg["ppo"]["num_steps"]), "final-round5/config.toml :: ppo.num_steps")
    put("ppo.batch", n(cfg["env"]["num_envs"] * cfg["ppo"]["num_steps"]), "num_envs x num_steps")
    put("ppo.gamma", cfg["ppo"]["gamma"], "final-round5/config.toml :: ppo.gamma")
    put("ppo.evalep", n(cfg["env"]["eval_episodes"]), "final-round5/config.toml :: env.eval_episodes")


def registration():
    lock1 = J(REPO / "preregistration/stage1-lock.json")
    rec1 = J(REPO / "preregistration/stage1-github-receipt.json")
    put("reg.s1.lock", utc(lock1["locked_at_utc"]), "preregistration/stage1-lock.json :: locked_at_utc")
    put("reg.s1.push", utc(rec1["github_pushed_at"]), "stage1-github-receipt.json :: github_pushed_at")
    lock2 = J(REPO / "preregistration/stage2-lock.json")
    rec2 = J(REPO / "preregistration/stage2-github-receipt.json")
    put("reg.s2.lock", utc(lock2["locked_utc"]), "preregistration/stage2-lock.json :: locked_utc")
    put("reg.s2.tested", lock2["tested_commit"][:7], "stage2-lock.json :: tested_commit")
    put("reg.s2.commit", rec2["lock_commit"][:7], "stage2-github-receipt.json :: lock_commit")
    put("reg.s2.push", utc(rec2["github_repo_pushed_at"]), "stage2-github-receipt.json :: github_repo_pushed_at")
    put("reg.s2.obs", hm(rec2["observed_utc"]), "stage2-github-receipt.json :: observed_utc")
    pre1, pre2 = J(REPO / "stage3/proofs/pre-summary-amendment-commit.json"), J(REPO / "stage3/proofs/amendment-commit.json")
    assert pre1["descriptive_amendment_sha256"] == pre2["descriptive_amendment_sha256"]
    put("reg.r1.commit", pre1["commit"][:7], "stage3/proofs/pre-summary-amendment-commit.json :: commit")
    put("reg.r1.public", hm(pre1["public_commit_utc"]), "pre-summary-amendment-commit.json :: public_commit_utc")
    put("reg.r1.verified", hm(pre1["verified_utc"]), "pre-summary-amendment-commit.json :: verified_utc")
    put("reg.r2.commit", pre2["commit"][:7], "stage3/proofs/amendment-commit.json :: commit")
    put("reg.r2.public", hm(pre2["public_commit_utc"]), "amendment-commit.json :: public_commit_utc")
    put("reg.r2.verified", hm(pre2["verified_utc"]), "amendment-commit.json :: verified_utc")
    put("reg.descsha", pre1["descriptive_amendment_sha256"][:12], "pre-summary-amendment-commit.json :: descriptive_amendment_sha256")
    cost = J(SRC / "s3-cost.json")
    put("reg.launch", hm(cost["current_service_lease"]["started_utc"]), "s3-cost.json :: current_service_lease.started_utc")
    put("reg.launchdate", utc(cost["current_service_lease"]["started_utc"]), "s3-cost.json :: current_service_lease.started_utc")
    # The gate report states, without a time, that the budget-50 and Mac-resume amendments were publicly pushed
    # (commit 3cfaa88) before the restart checks; the restarted controller's startup was verified at this time.
    sv = J(REPO / "handoff-cross-platform/mac-resume/startup-verification.json")
    put("reg.macrestart", hm(sv["verified_utc"]), "handoff-cross-platform/mac-resume/startup-verification.json :: verified_utc")
    assert "were publicly pushed at source3cfaa88" in (REPO / "handoff-cross-platform/mac-resume/gate-report.md").read_text()
    hashes = J(SRC / "s3-winner-hashes.json")
    put("win.E", n(hashes["cma"]["candidate"]), "s3-winner-hashes.json :: cma.candidate", "REG")
    put("win.R", n(hashes["random"]["candidate"]), "s3-winner-hashes.json :: random.candidate", "REG")
    put("win.Esha", hashes["cma"]["sha256"][:12], "s3-winner-hashes.json :: cma.sha256")
    put("win.Rsha", hashes["random"]["sha256"][:12], "s3-winner-hashes.json :: random.sha256")
    ref = J(SRC / "SOURCES.json")["repo/uv.lock"]["artifact"].split(":")[1]
    put("reg.research", ref, "data/raw/SOURCES.json :: research-branch commit the repo/ copies come from")
    integ = J(SRC / "s3-integrity.json")
    put("s3.files", n(integ["marker_listed_file_hashes_verified"]), "s3-integrity.json :: marker_listed_file_hashes_verified")
    put("s3.srcfiles", n(integ["registered_source_files_verified"]), "s3-integrity.json :: registered_source_files_verified")
    put("s3.commit", integ["source_commit"][:7], "s3-integrity.json :: source_commit")


def stage1():
    s1 = J(SRC / "s1-numbers.json")
    for t in ("DoorKey-8x8", "UnlockPickup", "KeyCorridorS3R2", "KeyCorridorS3R1", "MultiRoom-N2-S4",
              "BlockedUnlockPickup", "DoorKey-16x16"):
        r = [x for x in s1["calibration"]["selection"] if x["task"] == t][0]
        k = t.replace("-", "")
        put(f"s1.{k}.area", f(r["best_mean_curve_area"]), f"s1-numbers.json :: calibration.selection[{t}].best_mean_curve_area", "REG")
    for r in s1["calibration"]["separation_at_1M"]:
        k = r["task"].replace("-", "")
        put(f"s1.sep.{k}.diff", sg(r["mean_area_difference_novelty_minus_task_only"]), "s1-numbers.json :: separation_at_1M", "PSD")
        put(f"s1.sep.{k}.p", fp(r["p"]), "s1-numbers.json :: separation_at_1M.p (two-sided)", "PSD")
        put(f"s1.sep.{k}.ph", fp(r["p_holm"]), "s1-numbers.json :: separation_at_1M.p_holm", "PSD")
    exp_n = {c["expected_n"] for c in s1["calibration"]["cells"]}
    assert len(exp_n) == 1
    put("s1.seeds", n(exp_n.pop()), "s1-numbers.json :: calibration.cells[*].expected_n", "REC")
    sel = s1["calibration"]["selection"]
    put("s1.rejected", n(sum(not r["kept"] for r in sel)), "s1-numbers.json :: calibration.selection kept=false", "REG")
    put("s1.tooeasy", n(sum(any("exceeds" in x for x in r["reasons"]) for r in sel)), "s1-numbers.json :: reasons 'exceeds 0.85'", "REG")
    put("s1.toohard", n(sum(any("below" in x for x in r["reasons"]) for r in sel)), "s1-numbers.json :: reasons 'below 0.05'", "REG")
    put("s1.sep.count", n(sum(not r["separates"] for r in s1["calibration"]["separation_at_1M"])),
        "s1-numbers.json :: count of separates=false", "PSD")
    put("s1.det", "not bit-identical" if not s1["determinism"][0]["bit_identical"] else "bit-identical",
        "s1-numbers.json :: determinism[0].bit_identical", "PSD")
    pin = J(SRC / "s2pre-pinned-determinism.json")
    put("pin.n", n(pin["completed_replicates"]), "s2pre-pinned-determinism.json :: completed_replicates", "PSD")
    put("pin.fp", n(pin["distinct_fingerprint_triples"]), "s2pre-pinned-determinism.json :: distinct_fingerprint_triples", "PSD")
    for c in ("zen3", "zen4", "intel"):
        put(f"pin.{c}", n(pin["class_counts"][c]), f"s2pre-pinned-determinism.json :: class_counts.{c}", "PSD")
    put("s1.guard", usd(s1["spending"]["guard_total_usd"]), "s1-numbers.json :: spending.guard_total_usd")
    put("s1.billed", usd(s1["spending"]["provider_reported_complete_intervals_cost_usd"]),
        "s1-numbers.json :: spending.provider_reported_complete_intervals_cost_usd")
    put("s1.cap", usd(s1["spending"]["limit_usd"], 0), "s1-numbers.json :: spending.limit_usd")


def stage2():
    fin = C("s2-finalists.csv")
    for r in fin:
        k = f"{r['method']}{r['candidate']}"
        put(f"fin.{k}.search", f(r["search_fitness"]), f"s2-finalists.csv[{k}] :: search_fitness", "PSD")
        put(f"fin.{k}.reeval", f(r["reevaluated_fitness"]), f"s2-finalists.csv[{k}] :: reevaluated_fitness", "PSD")
        put(f"fin.{k}.change", sg(float(r["reevaluated_fitness"]) - float(r["search_fitness"])),
            f"s2-finalists.csv[{k}] :: reevaluated - search", "EXP")
    e, r_ = one(fin, method="cma", candidate="113"), one(fin, method="random", candidate="57")
    put("fin.EminusR", sg(float(e["reevaluated_fitness"]) - float(r_["reevaluated_fitness"])),
        "s2-finalists.csv :: cma113 - random57 re-evaluated", "PSD")
    drops = [float(x["reevaluated_fitness"]) - float(x["search_fitness"]) for x in fin if x["method"] == "cma"]
    rises = [float(x["reevaluated_fitness"]) - float(x["search_fitness"]) for x in fin if x["method"] == "random"]
    put("fin.cmafell", n(sum(d < 0 for d in drops)), "s2-finalists.csv :: CMA-ES finalists with lower re-evaluated fitness", "EXP")
    put("fin.randrose", n(sum(d > 0 for d in rises)), "s2-finalists.csv :: random finalists with higher re-evaluated fitness", "EXP")
    cand = C("s2x-candidates.csv")
    for m in ("cma", "random"):
        best = max(cand, key=lambda x: float(x["area/combined"]) if x["method"] == m else -1)
        put(f"cand.{m}.best", f(best["area/combined"]), f"s2x-candidates.csv :: max area/combined ({m})", "PSD")
        put(f"cand.{m}.bestid", best["candidate"], f"s2x-candidates.csv :: argmax area/combined ({m})", "PSD")
    att, cost = J(SRC / "s2-attempts.json"), J(SRC / "s2-cost.json")
    put("s2.calls", n(att["provider_calls"]), "s2-attempts.json :: provider_calls")
    put("s2.attempts", n(att["physical_attempts"]), "s2-attempts.json :: physical_attempts")
    put("s2.extra", n(att["additional_same_call_attempts"]), "s2-attempts.json :: additional_same_call_attempts")
    put("s2.terminated", n(att["terminated_provider_calls"]), "s2-attempts.json :: terminated_provider_calls")
    put("s2.manual", n(att["manual_same_input_retry_calls"]), "s2-attempts.json :: manual_same_input_retry_calls")
    put("s2.nousage", n(att["attempts_missing_usage"]), "s2-attempts.json :: attempts_missing_usage")
    put("s2.ctrl", n(len(C("s2-interruptions.csv"))), "s2-interruptions.csv :: rows")
    for c in ("zen3", "intel", "zen4", "other"):
        put(f"cpu.s2.{c}", n(cost["scientific_success_cpu_mix"][c]), f"s2-cost.json :: scientific_success_cpu_mix.{c}")
    put("s2.ulp", text("@s2-report.md", r"(four) latent coordinates differed by 1 float64 ULP"),
        "s2-report.md ~ ULP diagnosis")
    put("s2.ulpgrowth", text("@s2-report.md", r"one growth value by (\d) ULPs"), "s2-report.md ~ ULP diagnosis")
    put("cost.s2.cap", usd(cost["hard_cap_usd"], 0), "s2-cost.json :: hard_cap_usd")
    put("cost.s2.guard", usd(cost["guard_total_usd"]), "s2-cost.json :: guard_total_usd")
    put("cost.s2.measured", usd(cost["completed_function_compute_usd"]), "s2-cost.json :: completed_function_compute_usd")
    put("cost.s2.billed", usd(cost["modal_billed_complete_intervals_usd"]), "s2-cost.json :: modal_billed_complete_intervals_usd")


def stage3():
    comp, cells, fish = C("s3-comparisons.csv"), C("s3-cells.csv"), C("s3-fisher.csv")
    for tk, task in TASKS.items():
        fam = "primary" if tk == "kc" else "secondary"
        for c in ("T", "N", "R"):
            r = one(comp, family=fam, task=task, comparator=c)
            k = f"{tk}.E{c}"
            put(f"{k}.diff", sg(r["mean_difference"]), f"s3-comparisons.csv[{fam},{task},{c}] :: mean_difference", "REG")
            put(f"{k}.gap", f(r["mean_difference"]), f"s3-comparisons.csv[{fam},{task},{c}] :: mean_difference", "REG")
            put(f"{k}.p", fp(float(r["raw_p"])), f"s3-comparisons.csv[{fam},{task},{c}] :: raw_p", "REG")
            put(f"{k}.ph", fp(float(r["holm_adjusted_p"])), f"s3-comparisons.csv[{fam},{task},{c}] :: holm_adjusted_p", "REG")
            put(f"{k}.U", f(r["U"], 1), f"s3-comparisons.csv[{fam},{task},{c}] :: U", "REG")
            put(f"{k}.hit", "hit" if r["rank_hit"] == "True" else "miss", f"s3-comparisons.csv :: rank_hit", "REG")
            put(f"{k}.meaningful", "yes" if r["meaningful_improvement"] == "True" else "no", "s3-comparisons.csv :: meaningful_improvement", "REG")
            put(f"{k}.method", "exact" if r["statistical_method"].startswith("exact") else "permutation",
                "s3-comparisons.csv :: statistical_method", "REG")
            fr = one(fish, task=task, comparison=f"E > {c}")
            put(f"{k}.fisher", fp(float(fr["descriptive_one_sided_fisher_p"])), f"s3-fisher.csv[{task},E > {c}]", "PSD")
        for m in METHODS:
            r = one(cells, method=m, task=task)
            k = f"{tk}.{m}"
            put(f"{k}.area", f(r["mean_area"]), f"s3-cells.csv[{m},{task}] :: mean_area", "PSD")
            put(f"{k}.sd", f(r["sd_area"]), f"s3-cells.csv[{m},{task}] :: sd_area", "PSD")
            put(f"{k}.solve", f"{n(r['solve_count'])}/{n(r['n'])}", f"s3-cells.csv[{m},{task}] :: solve_count/n", "PSD")
            put(f"{k}.solve.of", f"{n(r['solve_count'])} of {n(r['n'])}", f"s3-cells.csv[{m},{task}] :: solve_count of n", "PSD")
            put(f"{k}.ci", f"[{float(r['solve_ci95_lower']):.2f}, {float(r['solve_ci95_upper']):.2f}]",
                f"s3-cells.csv[{m},{task}] :: solve_ci95", "PSD")
            put(f"{k}.fs", steps(r["median_censored_first_solved_step"]), f"s3-cells.csv[{m},{task}] :: median_censored_first_solved_step", "PSD")
    for tk in TASKS:
        fam = "primary" if tk == "kc" else "secondary"
    put("kc.hits", n(sum(r["rank_hit"] == "True" for r in comp if r["family"] == "primary")), "s3-comparisons.csv :: primary hits", "REG")
    put("kc.meaningful", n(sum(r["meaningful_improvement"] == "True" for r in comp if r["family"] == "primary")),
        "s3-comparisons.csv :: primary meaningful", "REG")
    put("sec.hits", n(sum(r["rank_hit"] == "True" for r in comp if r["family"] == "secondary")), "s3-comparisons.csv :: secondary hits", "REG")
    g = lambda m, t: float(one(cells, method=m, task=TASKS[t])["mean_area"])
    put("kc.SminusE", sg(g("S", "kc") - g("E", "kc")), "s3-cells.csv :: S - E mean area, KeyCorridorS3R2", "PSD")
    put("up.EminusS", sg(g("E", "up") - g("S", "up")), "s3-cells.csv :: E - S mean area, UnlockPickup", "PSD")
    put("up.EminusS.gap", f(g("E", "up") - g("S", "up")), "s3-cells.csv :: E - S mean area, UnlockPickup", "PSD")
    tam, lab = J(SRC / "s3-tamper.json"), J(SRC / "s3-label.json")
    put("tam.P0", f(tam["registered_P0"], 4), "s3-tamper.json :: registered_P0", "REG")
    put("tam.P0full", repr(tam["registered_P0"]), "s3-tamper.json :: registered_P0", "REG")
    put("tam.P0five", f(tam["registered_P0"], 5), "s3-tamper.json :: registered_P0", "REG")
    put("tam.off.area", f(tam["mean_area_off"]), "s3-tamper.json :: mean_area_off", "REG")
    put("tam.on.area", f(tam["mean_area_on"]), "s3-tamper.json :: mean_area_on", "REG")
    put("tam.diff", f(tam["mean_area_off_minus_on"]), "s3-tamper.json :: mean_area_off_minus_on", "REG")
    put("tam.p", fp(tam["raw_p"]), "s3-tamper.json :: raw_p", "REG")
    put("tam.U", f(tam["U"], 1), "s3-tamper.json :: U", "REG")
    put("tam.solve.off", f"{n(tam['solves_off'])}/{n(tam['n_off'])}", "s3-tamper.json :: solves_off/n_off", "PSD")
    put("tam.solve.on", f"{n(tam['solves_on'])}/{n(tam['n_on'])}", "s3-tamper.json :: solves_on/n_on", "PSD")
    put("tam.vsT", sg(lab["mean_area_gap_vs_task_only"]), "s3-label.json :: mean_area_gap_vs_task_only", "PSD")
    put("tam.vsT.gap", f(-lab["mean_area_gap_vs_task_only"]), "s3-label.json :: -mean_area_gap_vs_task_only", "PSD")
    put("tam.label", lab["label"], "s3-label.json :: label", "PSD")
    meds = [w["median"] for w in lab["median_presses_per_episode_by_window"]]
    put("tam.medmin", f(min(meds), 1), "s3-label.json :: min window median presses per episode", "PSD")
    put("tam.medmax", f(max(meds), 1), "s3-label.json :: max window median presses per episode", "PSD")
    put("tam.windows", n(len(meds)), "s3-label.json :: number of final-10% windows", "PSD")
    ts = C("s3-tamper-per-seed.csv")
    mean = lambda m, col: sum(float(r[col]) for r in ts if r["method"] == m) / sum(r["method"] == m for r in ts)
    for m, k in (("tamper_on", "on"), ("tamper_off", "off")):
        put(f"tam.{k}.pps10", pct(mean(m, "final10_presses_per_step")), f"s3-tamper-per-seed.csv[{m}] :: mean final10_presses_per_step", "PSD")
        put(f"tam.{k}.pps10b", pct(mean(m, "final10_presses_per_step"), 3), f"s3-tamper-per-seed.csv[{m}] :: mean final10_presses_per_step", "PSD")
        put(f"tam.{k}.pps", pct(mean(m, "presses_per_step")), f"s3-tamper-per-seed.csv[{m}] :: mean presses_per_step", "PSD")
        put(f"tam.{k}.ppe10", f(mean(m, "final10_presses_per_completed_episode"), 1), f"s3-tamper-per-seed.csv[{m}] :: mean final10 presses/episode", "PSD")
        put(f"tam.{k}.ppe", f(mean(m, "presses_per_completed_episode"), 1), f"s3-tamper-per-seed.csv[{m}] :: mean presses/episode", "PSD")
        put(f"tam.{k}.iret", f(mean(m, "mean_episode_intrinsic_return")), f"s3-tamper-per-seed.csv[{m}] :: mean episode intrinsic return", "PSD")
        put(f"tam.{k}.iret10", f(mean(m, "final10_mean_episode_intrinsic_return")), f"s3-tamper-per-seed.csv[{m}] :: final10 intrinsic return", "PSD")
        put(f"tam.{k}.iretmax", f(max(float(r["maximum_episode_intrinsic_return"]) for r in ts if r["method"] == m), 5),
            f"s3-tamper-per-seed.csv[{m}] :: max maximum_episode_intrinsic_return", "PSD")
        put(f"tam.{k}.aboveP0", n(sum(float(r["episodes_above_exact_P0"]) for r in ts if r["method"] == m)),
            f"s3-tamper-per-seed.csv[{m}] :: sum episodes_above_exact_P0", "PSD")
        put(f"tam.{k}.above064", n(sum(float(r["episodes_above_rounded_064"]) for r in ts if r["method"] == m)),
            f"s3-tamper-per-seed.csv[{m}] :: sum episodes_above_rounded_064", "PSD")
        cl = one(cells, method=m, task="DoorKey-8x8")
        put(f"tam.{k}.ent", f(cl["mean_entropy_all_logged_rows_nats"]), f"s3-cells.csv[{m}] :: mean_entropy_all_logged_rows_nats", "PSD")
        put(f"tam.{k}.ent9", f(cl["mean_entropy_final9_rows_nats"]), f"s3-cells.csv[{m}] :: mean_entropy_final9_rows_nats", "PSD")
    put("tam.off.excess", sci(max(float(r["maximum_episode_excess_vs_exact_P0"]) for r in ts if r["method"] == "tamper_off")),
        "s3-tamper-per-seed.csv[tamper_off] :: max maximum_episode_excess_vs_exact_P0", "PSD")
    import math
    put("tam.uniform", pct(Fraction(1, 7)), "arithmetic: 1/7 (7 MiniGrid actions)", "DER")
    put("tam.ln7", f(math.log(7)), "arithmetic: ln 7 nats (uniform policy over 7 actions)", "DER")
    gen = C("s3-genomes.csv")
    for m in ("E", "R"):
        for ev in ("new_view", "pickup", "drop", "toggle"):
            r = one(gen, method=m, event=ev)
            k = f"g.{m}.{ev.replace('_', '')}"
            put(f"{k}.w", f(r["weight_effective_float32"]), f"s3-genomes.csv[{m},{ev}] :: weight_effective_float32")
            put(f"{k}.s", f(r["relief_effective_float32"]), f"s3-genomes.csv[{m},{ev}] :: relief_effective_float32")
            gro = float(r["growth_effective_float32"])
            put(f"{k}.g", "0" if gro == 0 else f(gro, 4), f"s3-genomes.csv[{m},{ev}] :: growth_effective_float32")
            put(f"{k}.period", "--" if gro == 0 else n(1 / gro), f"s3-genomes.csv[{m},{ev}] :: 1/growth (steps)", "DER")
            once = float(r["relief_effective_float32"]) >= float(V["d.h0"]["text"]) and gro == 0
            put(f"{k}.once", "yes" if once else "no", f"s3-genomes.csv[{m},{ev}] :: relief >= h0 and growth = 0", "DER")
    rd = one(gen, method="R", event="drop")
    wd, sd = float(rd["weight_effective_float32"]), float(rd["relief_effective_float32"])
    assert wd < 0
    # One drop changes that step's reward by w*s*(2x - s), x = need after growth <= 1; magnitude at most |w| s (2 - s).
    put("g.R.droppenalty", f(abs(wd) * sd * (2 - sd)), "s3-genomes.csv[R,drop] :: |w| s (2-s), the largest per-drop reward change", "DER")
    cost, bill = J(SRC / "s3-cost.json"), J(SRC / "s3-billing.json")
    sci_ = cost["physical_by_profile"]["science"]
    for c in ("zen3", "intel", "zen4", "other"):
        put(f"cpu.s3.{c}", n(sci_["cpu_mix"][c]), f"s3-cost.json :: physical_by_profile.science.cpu_mix.{c}")
    put("s3.extra", n(sci_["additional_same_call_attempts"]), "s3-cost.json :: science.additional_same_call_attempts")
    put("s3.manual", n(cost["manual_scientific_retries"]), "s3-cost.json :: manual_scientific_retries")
    put("s3.ctrl", n(cost["scientific_controller_restarts"]), "s3-cost.json :: scientific_controller_restarts")
    put("s3.handles", n(cost["distinct_worker_provider_handles"]), "s3-cost.json :: distinct_worker_provider_handles")
    put("cost.s3.cap", usd(cost["hard_cap_usd"], 0), "s3-cost.json :: hard_cap_usd")
    put("cost.s3.guard", usd(cost["frozen_final_guard_usd"]), "s3-cost.json :: frozen_final_guard_usd")
    put("cost.s3.measured", usd(sci_["worker_compute_estimate_usd"]), "s3-cost.json :: science.worker_compute_estimate_usd")
    put("cost.s3.billed", usd(bill["retained_billed_floor_usd"]), "s3-billing.json :: retained_billed_floor_usd")
    lid = J(REPO / "stage3/proofs/lid-closed-waiver.json")
    put("s3.lid", "waived, unverified" if lid["status"] == "WAIVED_UNVERIFIED" and not lid["actual_lid_closed_verified"] else "?",
        "stage3/proofs/lid-closed-waiver.json :: status")


def exploratory():
    grp, cor = C("s2x-groups.csv"), C("s2x-correlations.csv")
    for ev in ("toggle", "pickup", "drop"):
        for grouping, key, vals, other in (("weight sign", "weight_sign", ("positive", "negative"), ("refill_state", "all")),
                                           ("refill state", "refill_state", ("zero", "nonzero"), ("weight_sign", "all"))):
            for v in vals:
                r = one(grp, target="combined", event=ev, grouping=grouping, **{key: v, other[0]: other[1]})
                for strat in ("pooled", "random"):
                    put(f"x.{strat}.{ev}.{v}", f(r[f"{strat}_mean"]), f"s2x-groups.csv[{ev},{grouping}={v}] :: {strat}_mean", "EXP")
                    put(f"x.{strat}.{ev}.{v}.n", n(r[f"{strat}_n"]), f"s2x-groups.csv[{ev},{grouping}={v}] :: {strat}_n", "EXP")
        for strat in ("pooled", "random"):
            r = one(cor, stratum=strat, target="combined", feature=f"weight/{ev}", subset="all candidates")
            put(f"x.rho.{strat}.{ev}", f(r["rho"], 2), f"s2x-correlations.csv[{strat},weight/{ev}] :: rho", "EXP")
    put("x.sixteen", "1/16", "arithmetic: (1/2)^4 under the uniform latent prior", "DER")
    put("x.sixteenpct", pct(Fraction(1, 16), 2), "arithmetic: (1/2)^4", "DER")


def main():
    V.clear()
    design()
    registration()
    stage1()
    stage2()
    stage3()
    exploratory()
    keys = list(V)
    js = "{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(V[k], ensure_ascii=False)}" for k in keys) + "\n}\n"
    line = {k: i + 2 for i, k in enumerate(keys)}  # line 1 is "{"
    lines = ["% Generated by scripts/make_numbers.py from data/raw. Do not edit by hand.", r"\makeatletter"]
    lines += [rf"\@namedef{{val@{k}}}{{{V[k]['text']}}}\@namedef{{line@{k}}}{{{line[k]}}}" for k in keys]
    lines += [r"\makeatother", ""]
    return "\n".join(lines), js


if __name__ == "__main__":
    tex, js = main()
    (PAPER / "numbers.tex").write_text(tex)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "numbers.json").write_text(js)
    print(f"{len(V)} values -> paper/numbers.tex, results/numbers.json")
