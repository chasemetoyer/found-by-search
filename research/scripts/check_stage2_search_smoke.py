"""Pass/fail verification of retained exact-search-path smoke outputs; no training.

Never print or export smoke fitness, learned performance or winner parameters.
The receipt ties the exercised implementation to the future registration gate.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage2.common import canonical_sha, utc_now
from stage2.registration import TESTED_IMPLEMENTATION
from stage2.search import SEARCH_PATH_SMOKE, SearchPlan
from stage2.specs import validate


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--tested-commit', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    checks, failed = [], []

    def check(name, condition):
        checks.append(name)
        if not condition:
            failed.append(name)

    cfg = SEARCH_PATH_SMOKE
    ledger = read(args.results / 'budget-ledger.json')
    check('approved_cap_and_preserved_history', ledger['limit_usd'] == '35' and bool(ledger.get('limit_amendments')))
    check('within_cap', float(ledger['guard_total_usd']) < 35)
    check('all_calls_reconciled', all(c['status'] == 'complete' for c in ledger['calls'].values()))
    check('no_controller_stop', not (args.results / 'controller-stop.json').exists())
    summary = read(args.results / 'search-summary.json')
    for field in ('equal_candidate_budget', 'all_genomes_in_bounds', 'replay_reproduces_cma_candidates'):
        check(field, summary[field] is True)
    check('winners_selected', summary['winner_selected'] == {'cma': True, 'random': True})
    expected = len(cfg.tasks) * (cfg.population * cfg.generations * cfg.seeds_per_task +
                               cfg.rs_candidates * cfg.seeds_per_task +
                               2 * cfg.finalists_per_method * cfg.finalist_seeds_per_task)
    check('summary_complete', summary['runs_expected'] == summary['runs_recorded'] == expected)
    plan = SearchPlan(cfg)
    markers = {p.parent.name: p for p in args.results.glob(cfg.name + '_*/complete.json')}
    check('exact_unique_input_count', len(markers) == expected)
    cpu_rows = [json.loads(line) for line in (args.results / 'container-cpu-log.jsonl').read_text().splitlines() if line.strip()]
    logged_attempts = {row['attempt'] for row in cpu_rows}
    for key in sorted(markers, key=plan.reconcile_sort_key):
        marker = markers[key]
        result = read(marker)
        directory = marker.parent / 'attempts' / result['attempt']
        spec = read(directory / 'spec.json')
        validate(spec)  # actual committed search-mode validator
        check(key + ':identity', result['id'] == key and result['spec_sha256'] == canonical_sha(spec))
        check(key + ':source', result['source_commit'] == args.tested_commit)
        check(key + ':smoke_only', spec['seed'] >= 5000 and spec['total_timesteps'] == cfg.budget_steps)
        check(key + ':unpinned', not spec.get('isa_pins'))
        check(key + ':success', result.get('outcome') == 'success')
        environment = read(directory / 'environment.json')
        check(key + ':cpu', bool(environment['cpu'].get('vendor_id')))
        provenance = read(directory / 'provenance.json')
        check(key + ':actual_unpinned_environment', all(value is None for value in provenance['isa_environment'].values()))
        for attempt in result.get('attempts', [{'attempt': result['attempt']}]):
            check(key + ':cpu_attempt:' + attempt['attempt'], attempt['attempt'] in logged_attempts)
        for relative, expected_hash in result['files'].items():
            parts = Path(relative).parts
            check(key + ':safe_path:' + relative, parts[:2] == ('runs', key) and '..' not in parts)
            check(key + ':file_hash:' + relative, digest(marker.parent.joinpath(*parts[2:])) == expected_hash)
        plan.record(key, result, directory)  # replay, without exposing any fitness
    check('replayed_plan_complete', plan.done(ledger))
    check('replayed_summary', all(plan.summary()[field] == summary[field] for field in
                                ('runs_recorded', 'runs_expected', 'replay_reproduces_cma_candidates', 'winner_selected')))
    # Read only for internal consistency, never expose the smoke selection data.
    check('selection_export', read(args.results / (cfg.name + '-selection.json')) == plan.selection_record())
    hashes = {}
    for name in TESTED_IMPLEMENTATION:
        tested = subprocess.check_output(['git', 'show', args.tested_commit + ':' + name], cwd=ROOT)
        hashes[name] = hashlib.sha256(tested).hexdigest()
        check('unchanged_tested_code:' + name, digest(ROOT / name) == hashes[name])
        check('exported_tested_code:' + name, digest(args.results.parent / name) == hashes[name])
    receipt = dict(status='FAIL' if failed else 'PASS', mode={'mode': 'search', 'profile': 'smoke'},
                   run_id=args.run_id, tested_commit=args.tested_commit, verified_utc=utc_now(),
                   implementation_sha256=hashes, failed_checks=failed, checks=checks,
                   reports_performance=False, results_directory=str(args.results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': receipt['status'], 'failed_checks': failed, 'run_id': args.run_id}))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
