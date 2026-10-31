"""Step 0: copy every saved result file, and every research-branch file, that the paper reads into data/raw.

The paper's generators read only these snapshots and repository files, never the artifacts directory, so the
draft can be rebuilt from the repository alone. Absolute local paths inside a source file (run directories and
the like, never numbers) are replaced by placeholders such as <openresearch>/ while copying; SOURCES.json records
both the original artifact's SHA256 and the scrubbed snapshot's SHA256, and SHA256SUMS lists the snapshots.
Registration, receipt and code files are copied from the research-branch commit RESEARCH_REF into data/raw/repo,
with the same scrubbing, and amendment-commits.json records which commit added each amendment and when that commit
first became public (GitHub push feed, or the verified pre-launch receipts). Re-running refuses to overwrite a
snapshot whose source has changed unless --refresh is given.

Usage (repo root): EIR_ARTIFACTS=<artifacts directory> make snapshot   (needs the research branch in this clone)
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw"
REPO = SRC / "repo"
RESEARCH_REF = "7acab8c"  # head of orx/stage-3-modal-launch-path-and-frozen-evaluation with the paper-2 draft
AMEND = ["stage1-amendment-20261028T032036Z", "stage1-amendment-20261028T032433Z", "stage1-amendment-20261028T033052Z",
         "stage1-amendment-20261028T034510Z", "stage1-amendment-20261028T043622Z",
         "stage2-dashboard-amendment-20261028T161947Z", "stage2-recovery-amendment-20261028",
         "stage2-recovery2-amendment-20261028", "stage2-cloud-amendment-20261028", "stage2-budget50-amendment-20261028",
         "stage2-mac-resume-amendment-20261028", "stage3-budget15-amendment-20261029", "stage3-prelaunch-update-20261029",
         "stage3-prelaunch-hardening-20261029", "stage3-guardian-name-amendment-20261029",
         "stage3-web-concurrency-amendment-20261029"]
REPO_FILES = ([f"preregistration/{a}.json" for a in AMEND] +
              ["preregistration/stage1.md", "preregistration/stage2.md", "preregistration/stage1-lock.json",
               "preregistration/stage1-github-receipt.json", "preregistration/stage2-lock.json",
               "preregistration/stage2-github-receipt.json", "preregistration/stage3-descriptive-amendment-20261029.md",
               "preregistration/stage3-exploratory-analysis-plan.md", "stage3/specs.py", "generic_reward.py",
               "stage3/proofs/pre-summary-amendment-commit.json", "stage3/proofs/amendment-commit.json",
               "stage3/proofs/lid-closed-waiver.json", "stage3/frozen/frozen-winner-cma.json",
               "stage3/frozen/frozen-winner-random.json", "handoff-cross-platform/mac-resume/startup-verification.json",
               "handoff-cross-platform/mac-resume/gate-report.md", "handoff-cross-platform/code/final-round5/config.toml",
               "handoff-cross-platform/code/final-round5/ppo.py", "uv.lock", ".python-version"])
ART = Path(os.environ.get("EIR_ARTIFACTS", ""))
SCRUB = [(re.compile(rb"/Users/[^/\"\s]+/\.local/share/openresearch/"), b"<openresearch>/"),
         (re.compile(rb"/Users/[^/\"\s]+/Desktop/evolved-intrinsic-rewards/"), b"<repo>/"),
         (re.compile(rb"/(?:Users|home)/[^/\"\s]+/"), b"<home>/")]


def scrub(data):
    for pattern, placeholder in SCRUB:
        data = pattern.sub(placeholder, data)
    return data

FILES = {  # snapshot name: artifact path (relative to ART)
    "s1-numbers.json": "stage1/stage1-numbers.json",
    "s2-finalists.csv": "stage2-report/finalists.csv",
    "s2-per-run.csv": "stage2-report/per-run.csv",
    "s2-search-summary.json": "stage2-report/search-summary.json",
    "s2-search-selection.json": "stage2-report/search-selection.json",
    "s2-cost.json": "stage2-report/cost-and-attempt-summary.json",
    "s2-attempts.json": "stage2-report/attempt-accounting-check.json",
    "s2-interruptions.csv": "stage2-report/interruptions-and-controller-history.csv",
    "s2-report.md": "stage2-report/stage2-report.md",
    "s2pre-pinned-determinism.json": "stage2-review/pinned-determinism-summary.json",
    "s2x-candidates.csv": "stage3-report/exploratory-stage2/candidates.csv",
    "s2x-groups.csv": "stage3-report/exploratory-stage2/pooled-random-object-groups.csv",
    "s2x-correlations.csv": "stage3-report/exploratory-stage2/correlations.csv",
    "s2x-summary.json": "stage3-report/exploratory-stage2/exploratory-summary.json",
    "s3-cells.csv": "stage3-report/analysis/cells.csv",
    "s3-comparisons.csv": "stage3-report/analysis/registered-comparisons.csv",
    "s3-fisher.csv": "stage3-report/analysis/descriptive-fisher.csv",
    "s3-tamper.json": "stage3-report/analysis/registered-tamper.json",
    "s3-label.json": "stage3-report/analysis/descriptive-label.json",
    "s3-tamper-per-seed.csv": "stage3-report/analysis/tamper-per-seed-descriptives.csv",
    "s3-tamper-windows.csv": "stage3-report/analysis/tamper-window-series.csv",
    "s3-genomes.csv": "stage3-report/analysis/winner-genomes.csv",
    "s3-definitions.json": "stage3-report/analysis/definitions.json",
    "s3-per-seed.csv": "stage3-report/analysis/per-seed.csv",
    "s3-learning-curves.csv": "stage3-report/figures/learning-curves.csv",
    "s3-winner-hashes.json": "stage3-report/frozen/winner-hashes.json",
    "s3-cost.json": "stage3-report/cost/cost-summary.json",
    "s3-billing.json": "stage3-report/cost/billing-final-report-refresh.json",
    "s3-integrity.json": "stage3-report/integrity-check.json",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def amendment_commits():
    """Which commit added each amendment, and when that commit first became public."""
    feed = json.loads((ROOT / "data" / "extracted" / "github-push-events.json").read_text())
    pushes = sorted(feed["push_events"], key=lambda e: e["created_at"])
    receipts = [json.loads((REPO / f"stage3/proofs/{n}.json").read_text()) for n in ("pre-summary-amendment-commit", "amendment-commit")]

    def contains(commit, head):
        return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, head],
                              capture_output=True).returncode == 0

    out = {}
    for name in AMEND + ["stage3-descriptive-amendment-20261029", "stage3-exploratory-analysis-plan"]:
        path = f"preregistration/{name}" + (".md" if name.startswith(("stage3-descriptive", "stage3-exploratory")) else ".json")
        commit = git("log", RESEARCH_REF, "--diff-filter=A", "--format=%h", "--", path).decode().split()[-1]
        when = git("log", "-1", "--format=%cI", commit).decode().strip()
        first = next(({"utc": e["created_at"], "evidence": "feed"} for e in pushes if contains(commit, e["head"])), None)
        if first is None:
            first = next(({"utc": r["public_commit_utc"], "evidence": "receipt"} for r in receipts if contains(commit, r["commit"])),
                         {"utc": None, "evidence": "none found"})
        out[name] = {"path": path, "commit": commit, "commit_time": when, "first_public": first}
    return out


def main():
    if not (ART / "stage3-report").is_dir():
        raise SystemExit("Set EIR_ARTIFACTS to the project's artifacts directory")
    refresh = "--refresh" in sys.argv
    SRC.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, rel in FILES.items():
        raw = (ART / rel).read_bytes()
        clean, dst = scrub(raw), SRC / name
        if dst.exists() and sha(dst.read_bytes()) != sha(clean) and not refresh:
            raise SystemExit(f"{name}: source changed since the snapshot; inspect it, then rerun with --refresh")
        dst.write_bytes(clean)
        manifest[name] = {"artifact": rel, "artifact_sha256": sha(raw), "sha256": sha(clean), "bytes": len(clean),
                          "paths_scrubbed": clean != raw}
    for rel in REPO_FILES:
        raw = git("show", f"{RESEARCH_REF}:{rel}")
        clean, dst = scrub(raw), REPO / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(clean)
        manifest[f"repo/{rel}"] = {"artifact": f"git:{RESEARCH_REF}:{rel}", "artifact_sha256": sha(raw), "sha256": sha(clean),
                                   "bytes": len(clean), "paths_scrubbed": clean != raw}
    commits = (json.dumps(amendment_commits(), indent=1) + "\n").encode()
    (REPO / "amendment-commits.json").write_bytes(commits)
    manifest["repo/amendment-commits.json"] = {"artifact": f"derived from git history at {RESEARCH_REF}", "artifact_sha256": sha(commits),
                                               "sha256": sha(commits), "bytes": len(commits), "paths_scrubbed": False}
    (SRC / "SOURCES.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (SRC / "SHA256SUMS").write_text("".join(f"{v['sha256']}  {k}\n" for k, v in sorted(manifest.items())))
    print(f"{len(manifest)} snapshots, {sum(v['bytes'] for v in manifest.values()):,} bytes, "
          f"{sum(v['paths_scrubbed'] for v in manifest.values())} scrubbed of local paths")


if __name__ == "__main__":
    main()
