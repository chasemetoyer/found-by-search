"""Read-only Stage 2 observer. No provider jobs, tracking writes or controller imports.

Inputs: the saved run registry, volume budget-ledger.json and volume result
files. Writes only its own local cache, JSON, HTML, process state and history.
Run: uv run --locked python scripts/stage2_dashboard.py --watch --interval 600
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time

SHARED = Path('<repo>/handoff-cross-platform')
DEFAULT_OUT = Path('<openresearch>/files/evolved-intrinsic-rewards/stage2-progress')
ASSETS = Path(__file__).parent / 'stage2_dashboard_assets'
sys.path.insert(0, str(ASSETS))
from orx_chart import render_chart

TITLE = 'in progress, not the stage-2 report'
KEY = re.compile(r'^search_(cma|random|finalist_cma|finalist_random)-')
PLANNED = 1560
TASKS = ('DoorKey-8x8', 'UnlockPickup')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cpu_class(env):
    cpu = env.get('cpu') or {}
    if not cpu and env.get('cpuinfo'):
        fields = dict(line.split(':', 1) for line in env['cpuinfo'].split('\n\n')[0].splitlines() if ':' in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        cpu = {'vendor_id': fields.get('vendor_id'), 'cpu_family': int(fields.get('cpu family', -1)),
               'model': int(fields.get('model', -1))}
    if cpu.get('vendor_id') == 'AuthenticAMD' and cpu.get('cpu_family') == 25:
        if 0 <= cpu.get('model', -1) < 16:
            return 'AMD Zen 3'
        if 16 <= cpu.get('model', -1) < 32:
            return 'AMD Zen 4'
    if cpu.get('vendor_id') == 'GenuineIntel':
        return 'Intel'
    return 'Other CPU' if cpu.get('vendor_id') else 'Unknown CPU'


class Reader:
    """The only volume API calls in this observer are read_file and listdir."""
    def __init__(self, volume, cache):
        self.volume, self.cache = volume, cache
        cache.mkdir(parents=True, exist_ok=True)

    def raw(self, path):
        return b''.join(self.volume.read_file(path))

    def json(self, path):
        return json.loads(self.raw(path))

    def attempt(self, key, attempt, names, finished):
        identity = sha((key + '/' + attempt).encode())
        cache = self.cache / (identity + '.json')
        if finished and cache.exists():
            return json.loads(cache.read_text())
        base = f'/runs/{key}/attempts/{attempt}/'
        row = {'key': key, 'attempt': attempt}
        for name in ('attempt.json', 'environment.json', 'usage.json', 'infrastructure_failure.json'):
            if name in names:
                row[name] = self.json(base + name)
        if finished:
            atomic(cache, row)
        return row

    def result(self, key):
        cache = self.cache / (sha(key.encode()) + '-result.json')
        if cache.exists():
            return json.loads(cache.read_text())
        marker = self.json('/runs/' + key + '/complete.json')
        if marker['id'] != key:
            raise ValueError('Mismatched completed run identity')
        prefix = f'runs/{key}/attempts/{marker["attempt"]}/'

        def checked(name):
            path = prefix + name
            raw = self.raw('/' + path)
            if sha(raw) != marker['files'].get(path):
                raise ValueError('Result hash mismatch: ' + path)
            return raw

        spec = json.loads(checked('spec.json'))
        canonical = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if sha(canonical) != marker['spec_sha256'] or spec['id'] != key:
            raise ValueError('Specification hash mismatch')
        if marker.get('outcome', 'success') == 'candidate_failure':
            area = 0.0
        else:
            rows = [json.loads(line) for line in checked('metrics.jsonl').decode().splitlines() if line.strip()]
            values = [row.get('train/success', 0.) for row in rows]
            if not values or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError('Missing or invalid success curve')
            # Match the registered float64 arithmetic without importing learner code.
            import numpy as np
            area = float(np.mean(values))
        row = {k: spec[k] for k in ('purpose', 'candidate', 'group', 'task', 'seed')}
        row.update(key=key, area=area, outcome=marker.get('outcome', 'success'),
                   marker_sha256=sha(json.dumps(marker, sort_keys=True).encode()))
        atomic(cache, row)
        return row


def aggregate_candidates(rows, tasks=TASKS):
    groups = defaultdict(list)
    for row in rows:
        if row['purpose'] in ('search_cma', 'search_random') and row['task'] in tasks:
            groups[row['purpose'], row['candidate'], row['group']].append(row)
    candidates, incomplete = [], 0
    for (purpose, candidate, group), observations in sorted(groups.items()):
        base = (1000 if purpose == 'search_cma' else 3000) + 3 * group
        expected = {(task, base + k) for task in tasks for k in range(3)}
        observed = {(row['task'], row['seed']) for row in observations}
        if observed != expected or len(observations) != 3 * len(tasks):
            incomplete += 1
            continue
        import numpy as np
        ordered = sorted(observations, key=lambda row: (tasks.index(row['task']), row['seed']))
        candidates.append({'method': 'cma' if purpose == 'search_cma' else 'random', 'candidate': candidate,
                           'group': group, 'fitness': float(np.mean([row['area'] for row in ordered]))})
    generations = []
    for g in range(12):
        values = [c['fitness'] for c in candidates if c['method'] == 'cma' and c['group'] == g]
        if len(values) == 10:
            generations.append({'generation': g + 1, 'best': max(values), 'mean': statistics.fmean(values), 'n': 10})
    return candidates, generations, incomplete



def task_for_key(key):
    match = re.search(r'-(DoorKey-8x8|UnlockPickup)-[0-9]+-seed[0-9]+$', key)
    return match.group(1) if match else None


def task_breakdown(ledger, records, physical):
    """Descriptive task projections; never change registered equal-task fitness."""
    result = {}
    for task in TASKS:
        calls = {k: c for k, c in ledger['calls'].items() if KEY.match(k) and task_for_key(k) == task}
        rows = [r for r in records if r['task'] == task]
        attempts = [r for r in physical if task_for_key(r['key']) == task]
        candidates, generations, incomplete = aggregate_candidates(rows, (task,))
        states = Counter(c['status'] for c in calls.values())
        historical = [c for c in ledger.get('infrastructure_history', []) if task_for_key(c.get('logical_key', '')) == task]
        guards = sum((Decimal(c['guard_charge_usd']) for c in list(calls.values()) + historical), Decimal(0))
        measured = sum((Decimal(c['measured']['compute_cost_estimate_usd']) for c in calls.values() if c.get('measured')), Decimal(0))
        by_key = Counter(r['key'] for r in attempts)
        result[task] = {
            'planned_runs': PLANNED // len(TASKS), 'completed_runs': states['complete'],
            'validated_completed_runs': len(rows), 'running_runs': states['running'] + states['reserved'],
            'retry_pending_runs': states['retry_pending'], 'uncertain_runs': states['uncertain'],
            'not_submitted_runs': PLANNED // len(TASKS) - len(calls),
            'candidates': candidates, 'generations': generations, 'incomplete_candidate_groups': incomplete,
            'candidate_failures': sum(c.get('outcome') == 'candidate_failure' for c in calls.values()),
            'infrastructure_failure_attempts': sum('infrastructure_failure.json' in r for r in attempts),
            'observed_restarts': sum(max(0, n - 1) for n in by_key.values()),
            'cpu_attempt_mix': dict(Counter(cpu_class(r.get('environment.json', {})) for r in attempts)),
            'input_guard_usd': float(guards), 'measured_compute_estimate_usd': float(measured),
        }
    return result

def collect(reader, registry, workers=4):
    ledger = reader.json('/budget-ledger.json')
    calls = {key: row for key, row in ledger['calls'].items() if KEY.match(key)}
    done = [key for key, row in calls.items() if row['status'] == 'complete']
    attempts = defaultdict(set)
    for entry in reader.volume.listdir('/runs', recursive=True):
        parts = Path(entry.path.lstrip('/')).parts
        if parts and parts[0] != 'runs':
            parts = ('runs',) + parts
        if len(parts) == 5 and parts[2] == 'attempts' and KEY.match(parts[1]):
            attempts[parts[1], parts[3]].add(parts[4])
    records, physical, errors = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(reader.result, key): ('result', key) for key in done}
        for (key, attempt), names in attempts.items():
            finished = 'usage.json' in names or key in done
            jobs[pool.submit(reader.attempt, key, attempt, names, finished)] = ('attempt', key)
        for future in as_completed(jobs):
            kind, key = jobs[future]
            try:
                (records if kind == 'result' else physical).append(future.result())
            except Exception as exc:
                errors.append({'kind': kind, 'key': key, 'error': f'{type(exc).__name__}: {exc}'})
    candidates, generations, incomplete = aggregate_candidates(records)
    by_key = defaultdict(list)
    for row in physical:
        by_key[row['key']].append(row)
    infra = sum('infrastructure_failure.json' in row for row in physical)
    restarts = sum(max(0, len(rows) - 1) for rows in by_key.values())
    confirmed, unknown = 0, 0
    for rows in by_key.values():
        rows.sort(key=lambda row: row.get('attempt.json', {}).get('started_utc', ''))
        for row in rows[:-1]:
            reason = str(row.get('attempt.json', {}).get('termination_reason', '')) + str(row.get('usage.json', {}).get('error', ''))
            if 'preempt' in reason.lower():
                confirmed += 1
            elif 'infrastructure_failure.json' not in row:
                unknown += 1
    measured = sum((Decimal(c['measured']['compute_cost_estimate_usd']) for c in ledger['calls'].values()
                    if c.get('measured')), Decimal(0))
    registry_calls = {r['run_key']: r for r in registry.get('calls', []) if KEY.match(r['run_key'])}
    drift = sum(registry_calls.get(key, {}).get('status') != row['status'] for key, row in calls.items())
    return {'label': TITLE, 'observed_utc': now(), 'ledger_updated_utc': ledger.get('updated_utc'),
            'registry_updated_utc': registry.get('updated_utc'), 'snapshot_status_differences': drift,
            'planned_runs': PLANNED, 'completed_runs': len(done), 'validated_completed_runs': len(records),
            'running_runs': sum(c['status'] in ('running', 'reserved') for c in calls.values()),
            'uncertain_runs': sum(c['status'] == 'uncertain' for c in calls.values()),
            'retry_pending_runs': sum(c['status'] == 'retry_pending' for c in calls.values()),
            'not_submitted_runs': PLANNED - len(calls), 'candidates': candidates, 'generations': generations,
            'incomplete_candidate_groups': incomplete, 'per_task': task_breakdown(ledger, records, physical),
            'guard_usd': float(ledger['guard_total_usd']), 'cap_usd': float(ledger['limit_usd']),
            'measured_compute_estimate_usd': float(measured),
            'billed_complete_intervals_usd': (float(ledger['provider_reported_complete_intervals_cost_usd'])
                                            if ledger.get('provider_reported_complete_intervals_cost_usd') is not None else None),
            'candidate_failures': sum(c.get('outcome') == 'candidate_failure' for c in calls.values()),
            'infrastructure_failure_attempts': infra, 'observed_restarts': restarts,
            'confirmed_preemptions': confirmed if confirmed else None,
            'unclassified_restarts': unknown, 'cpu_attempt_mix': dict(Counter(cpu_class(row.get('environment.json', {})) for row in physical)),
            'attempts_observed': len(physical), 'read_errors': errors,
            'read_only_sources': ['saved stage2-run-registry.json', 'Modal volume budget-ledger.json', 'Modal volume runs/**'],
            'limitations': ['Only complete six-run candidates and complete ten-candidate CMA generations are plotted.',
                           'Fitness is the registered mean success-curve area across two tasks and three seeds. No uncertainty interval is estimated.',
                           'Per-task views use all three seeds for that task; generation lines require all ten task-specific candidates. They are descriptive and do not alter registered fitness.',
                           'Task input guards include retained retry history but exclude shared setup, diagnostic allowances and overhead. Task billing and exact preemption causes are not available.',
                           'Additional attempts prove restarts, not their cause. Missing termination evidence cannot establish an exact preemption count.',
                           'Guard includes pre-lock work and live reservations. Compute and completed-hour billing are different estimates, not an invoice.',
                           'Registry and volume are asynchronous snapshots. Read failures or unvalidated completed inputs are disclosed.',
                           'Progress only: no held-out inference, winner conclusion, new fitting, or Stage 3 evaluation.']}


def render(data, output, interval):
    views = []
    fits = []
    for field, name, color in (('best', 'CMA best in generation', '#0072B2'), ('mean', 'CMA generation mean', '#E69F00')):
        points = [{'step': g['generation'], 'fitness': g[field]} for g in data['generations']]
        if points:
            fits.append({'name': name, 'color': color, 'points': points})
    points = [{'step': c['group'] + 1 + ((c['candidate'] % 10 + .5) / 10 - .5) * .7, 'fitness': c['fitness']}
              for c in data['candidates'] if c['method'] == 'random']
    if points:
        fits.append({'name': 'Random candidates (jittered within batch)', 'color': '#009E73', 'scatter': True, 'points': points})
    if fits:
        views.append({'key': 'fitness', 'label': 'Search fitness', 'type': 'line', 'unit': 'mean curve area',
                      'xLabel': 'CMA generation / random-search batch', 'series': fits})

    def bars(key, label, unit, items):
        views.append({'key': key, 'label': label, 'type': 'bar', 'unit': unit, 'xLabel': label,
                      'series': [{'name': label, 'points': [{'step': i, key: value, 'label': name}
                                  for i, (name, value) in enumerate(items) if value is not None]}]})
    bars('runs', 'Runs completed out of 1,560', 'logical runs', [('Completed', data['completed_runs']),
         ('In flight', data['running_runs']), ('Awaiting retry', data.get('retry_pending_runs', 0)),
         ('Uncertain', data['uncertain_runs']), ('Not submitted', data['not_submitted_runs'])])
    bars('spend', f"Spend against ${data['cap_usd']:g} cap", 'USD (not additive)', [('Guard', data['guard_usd']),
         ('Compute estimate', data['measured_compute_estimate_usd']), ('Billed intervals', data['billed_complete_intervals_usd']), ('Cap', data['cap_usd'])])
    bars('failures', 'Failures and restarts', 'observed count', [('Candidate failures', data['candidate_failures']),
         ('Infrastructure attempts', data['infrastructure_failure_attempts']), ('All restarts', data['observed_restarts']),
         ('Unclassified restarts', data['unclassified_restarts']), ('Confirmed preemptions', data['confirmed_preemptions'])])
    bars('cpu', 'CPU class mix', 'container attempts', sorted(data['cpu_attempt_mix'].items()) or [('No CPU metadata yet', 0)])
    for task, detail in data.get('per_task', {}).items():
        key = 'task_' + task.replace('-', '_')
        series = []
        for field, name, color in (('best', 'CMA best in generation', '#0072B2'), ('mean', 'CMA generation mean', '#E69F00')):
            points = [{'step': g['generation'], 'area': g[field]} for g in detail['generations']]
            if points:
                series.append({'name': name, 'color': color, 'points': points})
        points = [{'step': c['group'] + 1 + ((c['candidate'] % 10 + .5) / 10 - .5) * .7, 'area': c['fitness']}
                  for c in detail['candidates'] if c['method'] == 'random']
        if points:
            series.append({'name': 'Random candidates (three seeds)', 'color': '#009E73', 'scatter': True, 'points': points})
        if series:
            # Series metric names must match the view key expected by the renderer.
            for item in series:
                for point in item['points']:
                    point[key] = point.pop('area')
            views.append({'key': key, 'label': task + ' · success-curve area', 'type': 'line',
                          'unit': 'mean area over three seeds', 'xLabel': 'CMA generation / random-search batch', 'series': series})
        bars(key + '_runs', task + ' · runs out of 780', 'logical runs',
             [('Completed', detail['completed_runs']), ('In flight', detail['running_runs']),
              ('Awaiting retry', detail['retry_pending_runs']), ('Uncertain', detail['uncertain_runs']),
              ('Not submitted', detail['not_submitted_runs'])])
        bars(key + '_cost', task + ' · training costs', 'USD (not additive; shared costs excluded)',
             [('Input guards incl. retry history', detail['input_guard_usd']), ('Completed compute estimate', detail['measured_compute_estimate_usd'])])
        bars(key + '_failures', task + ' · failures and restarts', 'observed count',
             [('Candidate failures', detail['candidate_failures']), ('Infrastructure attempts', detail['infrastructure_failure_attempts']),
              ('Restarts (cause may be unknown)', detail['observed_restarts'])])
        bars(key + '_cpu', task + ' · CPU mix', 'container attempts',
             sorted(detail['cpu_attempt_mix'].items()) or [('No CPU metadata yet', 0)])
    render_chart(output, title=TITLE, subtitle='Stage 2 search · read-only observer · choose a view · no inferential claims', metrics=views)
    doc = output.read_text()
    cards = [('Completed', f"{data['completed_runs']:,} / 1,560"), ('Guard / cap', f"${data['guard_usd']:.2f} / ${data['cap_usd']:g}"),
             ('Candidate failures', str(data['candidate_failures'])), ('Confirmed preemptions',
              'Not recorded' if data['confirmed_preemptions'] is None else str(data['confirmed_preemptions']))]
    cards += [(task + ' completed', f"{detail['completed_runs']:,} / {detail['planned_runs']:,}") for task, detail in data.get('per_task', {}).items()]
    panel = '<section class="cards">' + ''.join(f'<div><small>{html.escape(k)}</small><strong>{html.escape(v)}</strong></div>' for k, v in cards) + '</section>'
    notes = '<section class="notes"><p>Updated ' + html.escape(data['observed_utc']) + f' · local refresh every {interval // 60} minutes.</p>'
    notes += '<p>Fitness: six complete task/seed runs per candidate; best/mean only after all ten candidates in a generation finish. Lines are not smoothed; random points share the axis with within-batch jitter.</p>'
    notes += '<ul>' + ''.join('<li>' + html.escape(s) + '</li>' for s in data['limitations']) + '</ul>'
    notes += f'<p>Validated result files: {data["validated_completed_runs"]} of {data["completed_runs"]} ledger completions. Read errors: {len(data["read_errors"])}. Snapshot status differences: {data["snapshot_status_differences"]}.</p>'
    notes += '<p>Tracking attachment omitted: publishing metrics is a write. This observer never initializes a tracking run or edits its metrics. <a href="stage2-progress.json">Download snapshot JSON</a></p></section>'
    style = '<style>html{overflow:auto}body{max-width:1180px;margin:24px auto;padding:0 16px}#chart{height:420px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 14px}.cards div{border:1px solid #e7e9ec;padding:16px;border-radius:8px}.cards small{display:block;color:#6b7078}.cards strong{display:block;font-size:23px;margin-top:8px}.notes{margin:20px 14px;line-height:1.6;color:#454a51}</style>'
    doc = doc.replace('</head>', f'<meta http-equiv="refresh" content="{interval}">' + style + '</head>')
    doc = doc.replace('<div class="toolbar">', panel + '<div class="toolbar">', 1).replace('</main>', notes + '</main>')
    atomic(output, doc)


def refresh(args):
    import modal
    volume = modal.Volume.from_name(args.volume, create_if_missing=False, version=2)
    reader = Reader(volume, args.out / '.cache')
    registry = json.loads(args.registry.read_text())
    data = collect(reader, registry, args.readers)
    atomic(args.out / 'stage2-progress.json', data)
    with (args.out / 'observations.jsonl').open('a') as stream:
        stream.write(json.dumps({k: data[k] for k in ('observed_utc', 'completed_runs', 'guard_usd', 'candidate_failures', 'observed_restarts')}) + '\n')
    render(data, args.out / 'stage2-dashboard.html', args.interval)
    for name in ('stage2-dashboard.html', 'stage2-progress.json'):
        atomic(args.mirror / name, (args.out / name).read_text())
    atomic(args.out / 'observer-status.json', {'pid': os.getpid(), 'last_success_utc': now(), 'status': 'ok', 'read_errors': len(data['read_errors'])})
    print(json.dumps({'event': 'dashboard_refreshed', 'utc': now(), 'completed': data['completed_runs'], 'read_errors': len(data['read_errors'])}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--volume', default='evolved-intrinsic-rewards-stage2')
    p.add_argument('--registry', type=Path, default=SHARED / 'stage2-run-registry.json')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--mirror', type=Path, default=SHARED / 'stage2-progress')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--interval', type=int, default=600)
    p.add_argument('--readers', type=int, default=4)
    args = p.parse_args()
    if args.interval < 600 or not 1 <= args.readers <= 4:
        p.error('Refresh interval must be at least 600 seconds; read concurrency must be 1..4')
    args.out.mkdir(parents=True, exist_ok=True)
    # Separate observer lock: never acquire, inspect or mutate controller locks.
    guard = (args.out / '.observer.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.watch:
        atomic(SHARED / 'stage2-dashboard-process.json', {'pid': os.getpid(), 'started_utc': now(),
               'command': sys.argv, 'interval_seconds': args.interval, 'output': str(args.out / 'stage2-dashboard.html'),
               'role': 'read-only observer; no controller or volume writes'})
    while True:
        started = time.monotonic()
        try:
            refresh(args)
        except Exception as exc:
            atomic(args.out / 'observer-status.json', {'pid': os.getpid(), 'error_utc': now(), 'status': 'stale', 'error': f'{type(exc).__name__}: {exc}'})
            print(json.dumps({'event': 'dashboard_error', 'error': f'{type(exc).__name__}: {exc}'}), flush=True)
            if not args.watch:
                raise
        if not args.watch:
            break
        time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
