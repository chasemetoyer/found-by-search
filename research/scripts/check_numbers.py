"""Independently cross-check generated Stage 1 numbers against retained records."""
import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import median, fmean
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = Path('/Users/chasemetoyer/.local/share/openresearch/files/evolved-intrinsic-rewards/stage1')


def read(p):
    return json.loads(Path(p).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numbers', type=Path, default=ART / 'stage1-numbers.json')
    parser.add_argument('--out', type=Path, default=ART / 'number-check.json')
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--volume-attempts', type=Path, default=Path(
        '/Users/chasemetoyer/Desktop/evolved-intrinsic-rewards/handoff-cross-platform/stage1-container-cpu/volume-attempts'))
    args = parser.parse_args()
    data = read(args.numbers)
    manifest = read(ROOT / 'stage1/manifest.json')
    specs = {s['id']: s for s in manifest['runs']}
    failures, checks = [], []

    def check(name, observed, expected):
        if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
            ok = math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
        else:
            ok = observed == expected
        checks.append(name)
        if not ok:
            failures.append(dict(check=name, observed=observed, expected=expected))

    def raw_score(directory, nominal):
        summary = read(directory / 'summary.json')
        rows = [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines() if s.strip()]
        y = [r.get('train/success', 0.) for r in rows]
        actual = nominal // 2048 * 2048
        check(str(directory) + ':steps', summary['total_timesteps'], actual)
        end = float(np.mean(y[-max(1, len(y)//10):]))
        return dict(curve_area=float(np.mean(y)), solved=end >= .9, final_success=end,
                    first_solved_censored=summary.get('first_solved_step') or nominal,
                    success_ever=summary.get('first_goal_step') is not None or max(y) > 0,
                    seed=summary['seed'])

    raw = {}
    for rid, score in data['scores_by_run'].items():
        spec = specs[rid]
        directory = Path(score['directory'])
        values = raw_score(directory, spec['total_timesteps'])
        for key, expected in values.items():
            check(rid + ':' + key, score[key], expected)
        for name, field in [('process_usage.json', 'process_core_seconds')]:
            usage = read(directory / name)
            check(rid + ':' + field, score[field], usage[field])
        usage = read(directory / 'usage.json')
        check(rid + ':wall', score['wall_seconds'], usage['wall_seconds'])
        check(rid + ':cost', score['cost_estimate_usd'], float(usage['compute_cost_estimate_usd']))
        values.update(cost_estimate_usd=float(usage['compute_cost_estimate_usd']))
        raw[rid] = values
        marker = read(directory.parent.parent / 'complete.json')
        for relative, expected in marker['files'].items():
            p = directory.parent.parent.joinpath(*Path(relative).parts[2:])
            check(rid + ':hash:' + relative, hashlib.sha256(p.read_bytes()).hexdigest(), expected)

    cells = {}
    for cell in data['calibration']['cells']:
        ids = [s['id'] for s in manifest['runs'] if s['purpose'] == 'calibration'
               and (s['task'],s['method'],s['total_timesteps']) == (cell['task'],cell['method'],cell['budget'])]
        observed = [raw[rid] for rid in ids if rid in raw]
        key = (cell['task'], cell['method'], cell['budget'])
        complete = len(observed) == 5
        check(str(key)+':n',cell['n'],len(observed))
        check(str(key)+':complete',cell['complete'],complete)
        check(str(key)+':solves',cell['solve_count'],sum(x['solved'] for x in observed))
        mean_area = fmean(x['curve_area'] for x in observed) if observed else None
        check(str(key)+':area',cell['mean_curve_area'],mean_area)
        check(str(key)+':median',cell['median_first_solved_step'],median(x['first_solved_censored'] for x in observed) if observed else None)
        cells[key] = dict(complete=complete, area=mean_area, success=any(x['success_ever'] for x in observed))
    for selection in data['calibration']['selection']:
        task = selection['task']
        training = selection['role'] == 'training'
        needed = [(task,method,budget) for method in ('none','episodic_count')
                  for budget in ((1000000,2000000) if training else (2000000,))]
        if not all(cells[k]['complete'] for k in needed):
            expected = 'incomplete'
        else:
            best = max(cells[(task,m,1000000 if training else 2000000)]['area'] for m in ('none','episodic_count'))
            keep = (best <= .85 and any(cells[(task,m,2000000)]['success'] for m in ('none','episodic_count'))) if training else .05 <= best <= .85
            expected = 'kept' if keep else 'rejected'
        check(task+':selection',selection['status'],expected)

    # There are only 252 labelings for two groups of five. Enumerate U directly,
    # independently of the report's SciPy exact/permutation implementation.
    calculated_p = []
    for comparison in data['calibration']['separation_at_1M']:
        task = comparison['task']
        groups = [[raw[s['id']]['curve_area'] for s in manifest['runs']
                   if s['purpose']=='calibration' and s['task']==task
                   and s['method']==method and s['total_timesteps']==1000000 and s['id'] in raw]
                  for method in ('episodic_count','none')]
        if any(len(g)!=5 for g in groups):
            calculated_p.append(None)
            continue
        def u(a,b):return sum((x>y)+.5*(x==y) for x in a for y in b)
        observed=u(*groups); pooled=groups[0]+groups[1]; values=[]
        for selected in combinations(range(10),5):
            subset=set(selected)
            values.append(u([pooled[i] for i in subset],[pooled[i] for i in range(10) if i not in subset]))
        p=min(1.,2*min(sum(v<=observed for v in values),sum(v>=observed for v in values))/len(values))
        calculated_p.append(p)
        check(task+':MWU',comparison['p'],p)
    adjusted=[None]*len(calculated_p);previous=0.
    for rank,(i,p) in enumerate(sorted(((i,p) for i,p in enumerate(calculated_p) if p is not None),key=lambda t:t[1])):
        previous=max(previous,min(1.,(3-rank)*p));adjusted[i]=previous
    for comparison,p in zip(data['calibration']['separation_at_1M'],adjusted):
        check(comparison['task']+':Holm',comparison['p_holm'],p)

    projection=data['cost_projection']
    if projection['selection_complete']:
        unit={}
        for task in projection['all_kept_tasks']:
            unit[task]={}
            for budget in (1000000,2000000):
                costs=[raw[s['id']]['cost_estimate_usd'] for s in manifest['runs'] if s['purpose']=='calibration'
                       and s['task']==task and s['total_timesteps']==budget and s['id'] in raw]
                unit[task][budget]=fmean(costs) if costs else None
                check(task+':unitcost:'+str(budget),projection['per_task'][task][str(budget)]['mean_cost_estimate_usd'],unit[task][budget])
        for row in projection['stage2']:
            check('stage2:callcount:'+str(row['fitness_budget']),row['training_runs'],720*len(projection['training_tasks']))
            check('stage2:cost:'+str(row['fitness_budget']),row['cost_estimate_usd'],720*sum(unit[t][row['fitness_budget']] for t in projection['training_tasks']))
        stage3unit=sum(unit[t][2000000] for t in projection['all_kept_tasks'])
        for row in projection['stage3']:
            check('stage3:cost:'+str(row['method_count_scenario']),row['cost_estimate_usd'],row['method_count_scenario']*20*stage3unit)
        for row in projection['tamper_extra']:
            check('tamper:cost:'+str(row['method_count_scenario'])+':'+str(row['additional_training_conditions']),row['additional_cost_estimate_usd'],row['method_count_scenario']*20*row['additional_training_conditions']*stage3unit)

    check('descriptive_no_inferential_tests',data['cross_platform']['inferential_tests'],False)
    for row in data['cross_platform']['seeds']:
        directory = ROOT / 'handoff-cross-platform/scored-results/a1b-run-3629f9a7' / f'seed{row["seed"]}'
        expected = raw_score(directory,2000000)
        check('Mac:'+str(row['seed'])+':area',row['mac']['curve_area'],expected['curve_area'])
        check('Mac:'+str(row['seed'])+':solve',row['mac']['solved'],expected['solved'])
    for record in data['validation']:
        if not record.get('complete'):
            continue
        suites = list(ET.parse(Path(record['directory'])/'validation.xml').getroot().iter('testsuite'))
        for k in ('tests','failures','errors','skipped'):
            check('validation:'+k,record[k],sum(int(s.get(k,0)) for s in suites))
        cases = list(ET.parse(Path(record['directory'])/'validation.xml').getroot().iter('testcase'))
        check('validation:testcase_nodes', record['testcase_nodes'], len(cases))
        import re
        counts = re.search(r'(\d+) passed, (\d+) subtests passed', (Path(record['directory'])/'stdout.log').read_text())
        if counts:
            check('validation:passed_cases', record['passed_cases'], int(counts.group(1)))
            check('validation:passed_subtests', record['passed_subtests'], int(counts.group(2)))
        check('validation:passing_targeted_tests', record.get('passing_targeted_tests'),
              [case.attrib['name'] for case in cases
               if not any(child.tag in ('failure','error','skipped') for child in case)])
        for name, value in record.get('unit_data',{}).items():
            check('validation:unit:'+name,value,read(Path(record['directory'])/'unit-tests'/name))
    for record in data['determinism']:
        candidates = [spec for spec in manifest['runs'] if spec['purpose']=='determinism' and spec['seed']==record['seed']]
        fingerprints=[]
        for spec in sorted(candidates,key=lambda x:x['replicate']):
            if spec['id'] not in raw:
                continue
            directory=Path(data['scores_by_run'][spec['id']]['directory'])
            rows=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines() if s.strip()]
            for row in rows:row.pop('sps',None)
            def canonical(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            import torch
            tensor_hash=hashlib.sha256()
            for name,tensor in sorted(torch.load(directory/'agent.pt',map_location='cpu',weights_only=True).items()):
                tensor_hash.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)]).encode())
                tensor_hash.update(tensor.detach().contiguous().numpy().tobytes())
            fingerprints.append(dict(metrics_without_sps_sha256=canonical(rows),summary_sha256=canonical(read(directory/'summary.json')),checkpoint_tensors_sha256=tensor_hash.hexdigest()))
        if len(fingerprints)==2:
            check('determinism:'+str(record['seed']),record['bit_identical'],fingerprints[0]==fingerprints[1])

    # Container CPU identity: enumerate attempt folders independently and
    # re-derive vendor/family/model/flags from the raw saved /proc/cpuinfo.
    import re
    attempts = {}
    for env in args.results.glob('*/attempts/*/environment.json'):
        attempts[env.parent.name] = (env.parent.parent.parent.name, env)
    if args.volume_attempts.exists():
        for env in args.volume_attempts.glob('*/*/environment.json'):
            attempts.setdefault(env.parent.name, (env.parent.parent.name, env))
    attempts = {a: v for a, v in attempts.items() if v[0] in specs}
    containers = {c['attempt']: c for c in data['containers']}
    check('containers:attempt_set', sorted(containers), sorted(attempts))
    for attempt, (key, env) in sorted(attempts.items()):
        c = containers.get(attempt)
        if c is None:
            continue
        env_bytes = env.read_bytes()
        check('container:'+attempt+':sha256', c['environment_sha256'], hashlib.sha256(env_bytes).hexdigest())
        block = (json.loads(env_bytes).get('cpuinfo') or '').split('\n\n')[0]
        def field(name):
            m = re.search(r'^' + re.escape(name) + r'\s*:\s*(.*?)\s*$', block, re.M)
            return m.group(1) if m else None
        vendor, fam, mod = field('vendor_id'), field('cpu family'), field('model')
        fam = int(fam) if fam and fam.isdigit() else None
        mod = int(mod) if mod and mod.isdigit() else None
        flags = set((field('flags') or '').split())
        tier = 'AVX512' if {'avx512vl','avx512bw','avx512dq','fma'} <= flags else ('AVX2' if {'avx2','fma'} <= flags else 'DEFAULT')
        for name, observed, expected in (('run_key', c['run_key'], key), ('vendor', c['vendor_id'], vendor),
                                         ('family', c['cpu_family'], fam), ('model', c['model'], mod),
                                         ('avx512f', c['avx512f'], 'avx512f' in flags),
                                         ('aten_tier', c['inferred_aten_cpu_capability'], tier)):
            check('container:'+attempt+':'+name, observed, expected)
        label = c['microarchitecture']
        if vendor == 'AuthenticAMD' and fam == 25 and mod is not None and mod <= 0x0F:
            consistent = label.startswith('AMD Zen 3')
        elif vendor == 'AuthenticAMD' and fam == 25 and mod is not None and 0x10 <= mod <= 0x1F:
            consistent = label.startswith('AMD Zen 4')
        elif vendor == 'GenuineIntel' and fam == 6 and mod == 0x55:
            consistent = 'model 55h' in label and (('Cascade Lake class' in label) == ('avx512_vnni' in flags and 'avx512_bf16' not in flags))
        else:
            consistent = label.startswith('unmapped')
        check('container:'+attempt+':label_consistent', consistent, True)
        marker = args.results / key / 'complete.json'
        role = (('final' if read(marker)['attempt'] == attempt else 'superseded') if marker.exists()
                else 'in progress or superseded')
        check('container:'+attempt+':role', c['role'], role)

    # Determinism explanation: recompute divergence location and replicate CPUs.
    final_cpu = {c['run_key']: c for c in data['containers'] if c['role'] == 'final'}
    for e in data['determinism_explanation']:
        if not e.get('complete'):
            continue
        seed = str(e['seed'])
        pair = sorted((s for s in manifest['runs'] if s['purpose']=='determinism' and s['seed']==e['seed']), key=lambda s: s['replicate'])
        dirs = [Path(data['scores_by_run'][s['id']]['directory']) for s in pair]
        rows = [[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines() if x.strip()] for d in dirs]
        n = min(len(r) for r in rows)
        same = lambda i: {k: v for k, v in rows[0][i].items() if k != 'sps'} == {k: v for k, v in rows[1][i].items() if k != 'sps'}
        diff = [i for i in range(n) if not same(i)]
        check('explain:'+seed+':bit_identical', e['bit_identical'], next(r['bit_identical'] for r in data['determinism'] if r['seed']==e['seed']))
        check('explain:'+seed+':logged_rows', e['logged_rows'], [len(r) for r in rows])
        check('explain:'+seed+':differing_rows', e['differing_rows'], len(diff))
        if diff:
            i = diff[0]
            check('explain:'+seed+':first_row', e['first_difference']['row'], i)
            check('explain:'+seed+':first_step', e['first_difference']['global_step'], rows[0][i]['global_step'])
            check('explain:'+seed+':first_fields', e['first_difference']['fields'],
                  sorted(k for k in rows[0][i] if k != 'sps' and rows[0][i][k] != rows[1][i].get(k)))
            for k, value in e['first_difference']['relative_differences'].items():
                a, b = rows[0][i][k], rows[1][i][k]
                check('explain:'+seed+':relative:'+k, value, abs(a-b)/max(abs(a), abs(b)))
        else:
            check('explain:'+seed+':first_difference_absent', e['first_difference'], None)
        rollout = [i for i in range(n) if any(rows[0][i].get(k) != rows[1][i].get(k) for k in rows[0][i] if k.startswith('train/'))]
        check('explain:'+seed+':first_rollout_row', (e['first_rollout_difference'] or {}).get('row'), rollout[0] if rollout else None)
        for j, r in enumerate(rows):
            for k in ('global_step', 'train/success', 'train/ret', 'train/len'):
                check('explain:'+seed+':final:'+str(j)+':'+k, e['final_rows'][j][k], r[-1].get(k))
        envs = [read(d/'environment.json') for d in dirs]
        check('explain:'+seed+':packages_identical', e['packages_identical'], envs[0]['packages'] == envs[1]['packages'])
        recorded = [read(d/'spec.json') for d in dirs]
        check('explain:'+seed+':spec_fields_differing', e['spec_fields_differing'],
              sorted(k for k in recorded[0] if recorded[0][k] != recorded[1].get(k)))
        for replicate, s in zip(e['replicates'], pair):
            for k in ('attempt', 'vendor_id', 'cpu_family', 'model', 'microarchitecture', 'avx512f', 'inferred_aten_cpu_capability'):
                check('explain:'+seed+':r'+str(s['replicate'])+':'+k, replicate.get(k), final_cpu.get(s['id'], {}).get(k))
    report = args.numbers.parent / 'stage1-report.md'
    if report.exists():
        text = report.read_text()
        check('report:open_stage2_cpu_isa_decision', all(s in text for s in ('ATEN_CPU_CAPABILITY=avx2', 'MKL_CBWR=AVX2', 'unpaired')), True)
    check('stage2_never_started',manifest['no_evolution_or_random_search'],True)
    output=dict(status='pass' if not failures else 'fail',checks=len(checks),failures=failures,
                completed_training_records=len(raw),containers_checked=len(attempts),
                scope='Independent raw-output arithmetic, complete-cell selection, source hashes, descriptive Mac rows, validation XML, numerical determinism, container CPU identity from raw cpuinfo and determinism divergence location; no new policies executed.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
