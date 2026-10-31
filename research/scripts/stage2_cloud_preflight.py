"""Read-only replay and submission audit: never imports or invokes a training backend."""
from __future__ import annotations
import collections, hashlib, json, os, platform, sys, time, resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import modal
import numpy as np
from stage2.search import REGISTERED_DRAFT
from stage2.cloud import ResumeSearchPlan as SearchPlan
from stage2.common import canonical_sha,parse_cpuinfo,cpu_class
from stage1.score import score_seed
from stage2.transport import retry_io


def run(out, volume=None, previous=None):
    started=time.monotonic()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    volume=volume or modal.Volume.from_name('evolved-intrinsic-rewards-stage2',create_if_missing=False,version=2)
    def read(path):return retry_io(lambda:b''.join(volume.read_file(path)))
    ledger=json.loads(read('/budget-ledger.json'))
    matched=0
    # Completion markers from already-audited uncertain successes are also usable,
    # without changing authoritative ledger state or retraining them.
    plan=SearchPlan(REGISTERED_DRAFT);rows=[];mismatches=[];areas={};candidates={}
    existing=sorted((k for k in ledger['calls'] if plan.owns(k)),key=plan.reconcile_sort_key)
    audit=json.loads((Path(__file__).resolve().parents[1]/'handoff-cross-platform/stage2-interruption/second-interruption/interruption-audit.json').read_text())
    saved_markers={r['id']:r['marker'] for r in audit['unresolved'] if r['durable_completion']}
    for key in existing:
        record=ledger['calls'][key]
        expected=plan.spec_for_key(key)
        if canonical_sha(expected)!=record['spec_sha256']:
            mismatches.append({'id':key,'reason':'Regenerated specification hash differs','expected':canonical_sha(expected),'recorded':record['spec_sha256']})
            # Stop before scoring this generation with incompatible candidates.
            break
        matched+=1
        if expected['purpose']=='search_cma':
            candidates[(expected['group'],expected['candidate'])]=expected['latent']
        marker=None
        if record['status']=='complete':
            local=previous/key/'complete.json' if previous else None
            marker=json.loads(local.read_text()) if local and local.exists() else json.loads(read('/runs/'+key+'/complete.json'))
        elif key in saved_markers:marker=saved_markers[key]
        if marker is None:continue
        if marker['id']!=key or marker['spec_sha256']!=record['spec_sha256']:raise ValueError('Saved output identity differs')
        directory=out/'evidence'/key;directory.mkdir(parents=True,exist_ok=True)
        for name in ['spec.json','metrics.jsonl','summary.json']:
            relative=f'runs/{key}/attempts/{marker["attempt"]}/{name}'
            old=previous/key/'attempts'/marker['attempt']/name if previous else None
            raw=old.read_bytes() if old and old.exists() else read('/'+relative)
            if hashlib.sha256(raw).hexdigest()!=marker['files'][relative]:raise ValueError('Evidence hash mismatch:'+key+'/'+name)
            (directory/name).write_bytes(raw)
        plan.record(key,marker,directory)
        areas[key]=plan.areas[key]
    # Only explicitly reconciled terminated inputs may be retried; all other
    # existing keys stay excluded, including already completed/successful inputs.
    ready=[]
    if not mismatches:
        for spec in plan.ready_specs(ledger):
            key=spec['id'];r=ledger['calls'].get(key)
            if r is None:ready.append({'id':key,'kind':'new'})
            elif r['status']=='retry_pending':ready.append({'id':key,'kind':'previously_authorized_retry'})
            elif r['status']=='uncertain' and key not in saved_markers:
                ready.append({'id':key,'kind':'requires_terminal_reconciliation_before_retry'})
    saved=set(areas)
    assert not (saved&{r['id'] for r in ready})
    counts=collections.Counter(ledger['calls'][k]['status'] for k in existing)
    highest=max((g for g,c in candidates),default=None)
    cpu=parse_cpuinfo(Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else None)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result={'status':'FAIL' if mismatches else 'PASS','platform':platform.platform(),'machine':platform.machine(),
            'run_id':os.environ.get('ORX_RUN_ID'),'source_commit':os.environ.get('ORX_COMMIT'),
            'cpu':cpu,'cpu_class':cpu_class(cpu),'wall_seconds':time.monotonic()-started,
            'process_cpu_seconds':usage.ru_utime+usage.ru_stime,
            'peak_rss_bytes':usage.ru_maxrss*(1024 if platform.system()=='Linux' else 1),
            'registered_input_keys':len(existing),'matched_input_keys':matched,
            'saved_completed_inputs':len(areas),'candidate_vectors_rebuilt':len(candidates),
            'latest_recorded_generation_zero_based':highest,'mismatches':mismatches,
            'dry_run_saved_keys_resubmitted':sorted(saved&{r['id'] for r in ready}),
            'dry_run_ready_keys':ready,'ledger_status_counts':dict(counts),
            'recorded_fitness_areas':areas,'rebuilt_latent_candidates':{f'{g}:{c}':z for (g,c),z in candidates.items()},
            'provider_submissions':0,'controller_or_ledger_mutated':False,
            'replay_uses_frozen_recorded_fitness':True,
            'note':'The current recorded ask panel is checked; future unasked generations have no recorded candidates to compare.'}
    (out/'replay-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['recorded_fitness_areas','rebuilt_latent_candidates','dry_run_ready_keys']},indent=2),flush=True)
    return result

if __name__=='__main__':
    result=run(Path('results/cloud-preflight'))
    if result['status']!='PASS':raise SystemExit(3)
