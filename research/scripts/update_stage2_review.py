"""Refresh the unlocked review from retained pass/fail and accounting records."""
import argparse
import json
from decimal import Decimal
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ART = Path('/Users/chasemetoyer/.local/share/openresearch/files/evolved-intrinsic-rewards/stage2-review')
SHARED = Path('/Users/chasemetoyer/Desktop/evolved-intrinsic-rewards/handoff-cross-platform/stage2-review')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    args = p.parse_args()
    receipt = json.loads((ART / 'search-mode-smoke-result.json').read_text())
    if receipt['status'] != 'PASS':
        raise SystemExit('Exact search path has not passed')
    projection = json.loads((ART / 'stage2-cost-projection.json').read_text())
    ledger = json.loads((args.results / 'budget-ledger.json').read_text())
    measured = sum((Decimal(c['measured']['compute_cost_estimate_usd']) for c in ledger['calls'].values()
                    if c.get('measured')), Decimal(0))
    accounting = dict(guard_total_usd=ledger['guard_total_usd'], measured_compute_usd=str(measured),
                      modal_billed_complete_intervals_usd=ledger.get('provider_reported_complete_intervals_cost_usd'),
                      cap_usd=ledger['limit_usd'], invoice_cost_usd=None,
                      limitations='Guard is protective; compute is an estimate; billed intervals omit recent usage and are not an invoice.')
    (ART / 'current-spending.json').write_text(json.dumps(accounting, indent=2) + '\n')
    s = projection['scenarios']['unpinned']
    s3 = projection['stage3']
    report = f'''# Stage 2 updated review — exact search path PASS

Pre-lock review snapshot. The scientific registration is not locked by this
document. The approved implementation has passed the exact-path rehearsal.

## Exact search-mode smoke

**PASS.** Tracked run `{receipt['run_id']}`, source `{receipt['tested_commit']}`.
The real search-mode controller, worker validator and plan wiring executed with
tiny budgets and seeds >=5000. Retained-output verification passed and its
receipt binds every mandatory runtime hash. Smoke learned performance, candidate
fitness and winner parameters are deliberately not reported.

[Pass/fail receipt](search-mode-smoke-result.json) ·
[Updated unlocked draft](stage2-preregistration-draft.md) ·
[Local safety checks](search-path-local-tests.xml).

## Current spending — separate quantities

| Quantity | USD |
|---|---:|
| Stage 2 budget guard, including allowance | {Decimal(accounting['guard_total_usd']):.4f} |
| Measured function compute estimate | {measured:.4f} |
| Modal-reported complete billing intervals | {Decimal(accounting['modal_billed_complete_intervals_usd']):.4f} |
| Stage 2 approved cap | 35.00 |
| Stage 3 separate approved cap | 10.00 |

Do not add the three spending figures. Recent provider charges can be absent;
none is claimed to be a final invoice. Every prior check and charge remains in
the shared Stage 2 ledger. [Exact accounting](current-spending.json).

## Refreshed unpinned projection

| Quantity | USD |
|---|---:|
| Future Stage 2 measured-compute proxy | {s['measured_compute_usd']:.2f} |
| Future Stage 2 expected-billed proxy | {s['expected_billed_usd']:.2f} |
| Stage 2 guard at end, including pre-lock work | {s['projected_guard_at_end_usd']:.2f} |
| Stage 2 conservative guard peak | {s['projected_guard_peak_usd']:.2f} |
| Headroom below Stage 2 cap at peak | {35-s['projected_guard_peak_usd']:.2f} |
| Stage 3 measured-compute proxy | {s3['measured_compute_proxy_usd']:.2f} |
| Stage 3 expected-billed proxy | {s3['expected_billed_proxy_usd']:.2f} |
| Stage 3 guard at end | {s3['projected_guard_at_end_usd']:.2f} |
| Stage 3 conservative guard peak | {s3['projected_guard_peak_usd']:.2f} |

Stage 2 duration proxy is {s['wall_hours_at_32']:.1f} hours at 32 concurrent
calls. Baseline runtime and Stage 1 lifecycle ratios proxy unrun candidates;
these estimates do not guarantee completion under either cap. Stage 3 execution
still requires the Stage 2 boundary approval. [Projection inputs and exact values](stage2-cost-projection.json).

## Approved decisions carried into the draft

- Unpinned runs; CPU logged per attempt; unpaired comparisons; exact
  cross-machine reproducibility is a limitation. No COMPATIBLE follow-up.
- 2M fitness, 3600-second call deadline, raw equal task weighting; coefficient 1;
  50% zero-growth encoding, initial mean .5, sigma .25; three finalists per
  method and ten fresh finalist seeds per training task.
- Candidate numerical failures score area zero without replacement. Infrastructure
  retries preserve the same input/call, at most three retries. Exhaustion stops
  for reconciliation. The per-plan >5% completed-logical-run failure guard is
  latched and drains active calls; its exact denominator is in the draft.
- E/R/T/N roster, descriptive fixed satiation; Holm m=3 on KeyCorridorS3R2 area.
  Meaningful comparisons also require observed mean-area gain >=.05. The primary
  transfer claim rests on one held-out task.
- Free-press tamper retraining only. Freeze the winner and compute the signed
  start payoff before tamper data: positive predicts off>on collapse; zero or
  negative predicts location-shift noninferiority with delta .05. The draft
  states the estimand and the exact conditional rule.

## Gate to launch

Deliver this draft and PASS result, corroborate that the user made the repository
public, then commit/push/verify the scientific lock and launch. The lock must
freeze exactly the exercised implementation; it may change the execution profile
data from smoke to registered but may add no new runtime code. No lock is
created by this report generator.

Earlier pre-check evidence remains available:
[CPU pinning](pinned-determinism-summary.json),
[history secret scan](secret-scan-report.md),
[incremental scan](incremental-secret-scan.json),
[old smoke mode](smoke-test-result.json).
'''
    (ART / 'stage2-review-package.md').write_text(report)
    shutil.copy2(ROOT / 'preregistration/stage2-draft.md', ART / 'stage2-preregistration-draft.md')
    for destination in (ROOT / 'handoff-cross-platform/stage2-review', SHARED):
        destination.mkdir(parents=True, exist_ok=True)
        for path in ART.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)
    shutil.copy2(ART / 'search-mode-smoke-result.json', ROOT / 'stage2/search-path-smoke-receipt.json')
    print(json.dumps({'smoke_status': receipt['status'], 'report': str(ART / 'stage2-review-package.md'), 'accounting': accounting}))


if __name__ == '__main__':
    main()
