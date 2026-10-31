"""Independent accounting arithmetic for the approved Stage 2/3 projection."""
import argparse
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path

ART = Path('/Users/chasemetoyer/.local/share/openresearch/files/evolved-intrinsic-rewards')
SHARED = Path('/Users/chasemetoyer/Desktop/evolved-intrinsic-rewards/handoff-cross-platform')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ledger', type=Path, required=True)
    a = p.parse_args()
    def read(path): return json.loads(path.read_text())
    projection = read(ART / 'stage2-review/stage2-cost-projection.json')
    numbers = read(ART / 'stage1/stage1-numbers.json')
    stage1 = read(SHARED / 'budget.json')['stages']['stage1']
    ledger = read(a.ledger)
    costs = {task: statistics.fmean(seed['cost_estimate_usd'] for cell in numbers['calibration']['cells']
                                  if cell['task'] == task and cell['budget'] == 2000000 for seed in cell['seeds'])
             for task in ('DoorKey-8x8', 'UnlockPickup', 'KeyCorridorS3R2')}
    billed_ratio = float(stage1['modal_billed_all_run_hours_usd']) / float(stage1['measured_completed_function_compute_usd'])
    guard_ratio = float(stage1['guard_components']['completed_guard_charges_usd']) / float(stage1['measured_completed_function_compute_usd'])
    checks = {}
    def eq(name, observed, expected): checks[name] = math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
    future = 780 * (costs['DoorKey-8x8'] + costs['UnlockPickup'])
    scenario = projection['scenarios']['unpinned']
    peak = 32 * 3660 * .00001754
    eq('stage2_run_count', projection['search_runs'] + projection['finalist_runs'], 1560)
    eq('stage2_compute', scenario['measured_compute_usd'], future)
    eq('stage2_expected_bill', scenario['expected_billed_usd'], future * billed_ratio)
    eq('stage2_guard_end', scenario['projected_guard_at_end_usd'], future * guard_ratio + float(ledger['guard_total_usd']))
    eq('stage2_guard_peak', scenario['projected_guard_peak_usd'], future * guard_ratio + float(ledger['guard_total_usd']) + peak)
    future3 = 100 * sum(costs.values()) + 40 * costs['DoorKey-8x8']
    stage3 = projection['stage3']
    eq('stage3_compute', stage3['measured_compute_proxy_usd'], future3)
    eq('stage3_bill', stage3['expected_billed_proxy_usd'], future3 * billed_ratio)
    eq('stage3_guard_end', stage3['projected_guard_at_end_usd'], future3 * guard_ratio + .5)
    eq('stage3_guard_peak', stage3['projected_guard_peak_usd'], future3 * guard_ratio + .5 + peak)
    accounting = read(ART / 'stage2-review/current-spending.json')
    eq('current_guard', float(accounting['guard_total_usd']), float(ledger['guard_total_usd']))
    eq('current_compute', float(accounting['measured_compute_usd']), float(sum(
        (Decimal(call['measured']['compute_cost_estimate_usd']) for call in ledger['calls'].values()), Decimal(0))))
    out = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
               scope='Independent accounting and cost-projection arithmetic; no smoke performance reported.')
    (ART / 'stage2-review/cost-number-check.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))
    if not all(checks.values()): raise SystemExit(1)


if __name__ == '__main__':
    main()
