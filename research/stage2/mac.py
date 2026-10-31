"""Approved Mac-only resume with a hard50USD cap and exact current-ledger preflight."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,platform,sys
from pathlib import Path
from types import SimpleNamespace
from stage2.common import ROOT,canonical_sha
from stage2.transport import retry_io

BUDGET_AMENDMENT='preregistration/stage2-budget50-amendment-20261028.json'
AMENDMENT='preregistration/stage2-mac-resume-amendment-20261028.json'
RECEIPT='stage2/mac-replay-receipt.json'
REPLACEMENTS={'stage2/controller.py','stage2/registration.py','scripts/run_stage1.sh'}
EXTERNAL_RUNS=('4e1c708d-d0da-4898-8c9d-1b9d27f9f523','fc91e753-e0e1-467a-af65-44566804fb09')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def authorized_hashes(lock,root):
    from stage2.cloud import authorized_hashes as previous,AMENDMENT as cloud_manifest
    effective=previous(lock,root)
    if not (root/AMENDMENT).exists():return effective
    a=json.loads((root/AMENDMENT).read_text());b=json.loads((root/BUDGET_AMENDMENT).read_text())
    if a['approval_seq']!=32 or a['scientific_changes'] or b['approval_seq']!=32 or b['limit_usd']!='50':
        raise ValueError('Mac resume/hard50 cap not authorized')
    if a['prior_cloud_amendment_sha256']!=sha((root/cloud_manifest).read_bytes()) or a['budget_amendment_sha256']!=sha((root/BUDGET_AMENDMENT).read_bytes()):
        raise ValueError('Mac amendment chain changed')
    for name,expected in b['implementation_sha256'].items():
        if name!='stage2/budget.py':raise ValueError('Budget-only amendment changed scope')
        if sha((root/'preregistration/frozen-pre-mac-stage2'/name).read_bytes())!=effective[name]:raise ValueError('Prior budget version not preserved')
        effective[name]=expected
    if set(a['implementation_sha256'])!=REPLACEMENTS:raise ValueError('Mac runtime scope changed')
    for name,expected in a['implementation_sha256'].items():
        if sha((root/'preregistration/frozen-pre-mac-stage2'/name).read_bytes())!=effective[name]:raise ValueError('Prior Mac route version not preserved')
        effective[name]=expected
    for name,expected in a['added_file_sha256'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or sha((root/p).read_bytes())!=expected:raise ValueError('Mac helper changed:'+name)
    return effective


def identity():
    a=json.loads((ROOT/AMENDMENT).read_text());b=json.loads((ROOT/BUDGET_AMENDMENT).read_text())
    return canonical_sha({**a['implementation_sha256'],**b['implementation_sha256'],**a['added_file_sha256']})


def account_external(ledger):
    for run in EXTERNAL_RUNS:
        ledger.data.setdefault('controller_sessions',{}).setdefault(run,{
            'guard_charge_usd':'0.10','host':'completed_modal_sandbox_diagnostic','run_id':run,
            'accounted_by_approval':32,'cost_basis':'Retained setup allowance, not provider invoice; no training'})
    ledger.save()


def normalized_prebudget(data,root=ROOT):
    """Verify only declared accounting fields changed; preserve all input/status/cost history."""
    a=json.loads((root/AMENDMENT).read_text())
    if data.get('limit_usd')!='50' or data.get('cap_restore_approval')!=32 or data.get('monetary_cap_enabled') is not True:
        raise ValueError('Hard50 cap is not restored')
    original=a['original_budget_fields']
    old_amendments=original.get('limit_amendments',[])
    amendments=data.get('limit_amendments',[])
    if len(amendments)!=len(old_amendments)+1 or amendments[:-1]!=old_amendments or amendments[-1].get('approval')!=32 or amendments[-1].get('limit_usd')!='50':
        raise ValueError('Unexpected budget amendment history')
    sessions=data.get('controller_sessions',{})
    if set(sessions)!=set(original.get('controller_sessions',{}))|set(EXTERNAL_RUNS):
        raise ValueError('Unexpected controller accounting records')
    for run in EXTERNAL_RUNS:
        row=sessions.get(run,{})
        if row.get('guard_charge_usd')!='0.10' or row.get('accounted_by_approval')!=32:
            raise ValueError('External diagnostic cost missing or changed')
    result=copy.deepcopy(data)
    for field in a['allowed_accounting_fields']:
        if field in original:result[field]=copy.deepcopy(original[field])
        else:result.pop(field,None)
    if canonical_sha(result)!=a['original_ledger_canonical_sha256']:
        raise ValueError('Non-budget ledger changed; reconcile explicitly before restart')
    return result


def reconcile_current(ledger,volume,results,query=None):
    from stage2.recovery_v2 import reconcile_interrupted,AMENDMENT as recovery_manifest
    recovery_id=sha((ROOT/recovery_manifest).read_bytes())
    if recovery_id in ledger.data.get('recovery_amendments',[]):return
    normalized=normalized_prebudget(ledger.data)
    preview=SimpleNamespace(data=normalized,save=lambda:None)
    reconcile_interrupted(preview,volume,results,query=query)
    # Only reconcile input states; preserve restored cap and accounting exactly.
    ledger.data['calls']=preview.data['calls']
    ledger.data['recovery_amendments']=preview.data['recovery_amendments']
    ledger.save()


def restore_budget(volume,out):
    from stage2.budget import Stage2Ledger
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    before=json.loads(retry_io(lambda:b''.join(volume.read_file('/budget-ledger.json'))))
    a=json.loads((ROOT/AMENDMENT).read_text())
    if canonical_sha(before)!=a['original_ledger_canonical_sha256']:
        raise ValueError('Budget restoration base ledger changed')
    path=out/'budget-ledger-restored.json';path.write_text(json.dumps(before))
    ledger=Stage2Ledger(path);account_external(ledger)
    assert ledger.data['calls']==before['calls']
    assert ledger.data.get('infrastructure_history')==before.get('infrastructure_history')
    normalized_prebudget(ledger.data)
    def upload():
        with volume.batch_upload(force=True) as batch:batch.put_file(path,'/budget-ledger.json')
    retry_io(upload)
    confirmed=json.loads(retry_io(lambda:b''.join(volume.read_file('/budget-ledger.json'))))
    assert confirmed==ledger.data and confirmed['limit_usd']=='50'
    return ledger


def preflight(out):
    import modal
    from stage2.budget import Stage2Ledger
    from scripts.stage2_mac_preflight import run
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    volume=modal.Volume.from_name('evolved-intrinsic-rewards-stage2',create_if_missing=False,version=2)
    raw=json.loads(retry_io(lambda:b''.join(volume.read_file('/budget-ledger.json'))))
    normalized_prebudget(raw)
    r=run(out/'replay',volume=volume,previous=Path(json.loads((ROOT/AMENDMENT).read_text())['previous_results_directory']))
    frozen=json.loads((ROOT/'handoff-cross-platform/cloud-controller/recorded-fitness.json').read_text())
    assert r['recorded_fitness_areas']==frozen['run_areas'],'Original Mac scoring differs from saved fitness'
    path=out/'preview-ledger.json';path.write_text(json.dumps(raw));ledger=Stage2Ledger(path)
    reconcile_current(ledger,volume,out)
    # Old pending retries were already approved, but confirm their current terminal state too.
    for key,row in ledger.data['calls'].items():
        if row['status']=='retry_pending':
            graph=retry_io(lambda:modal.FunctionCall.from_id(row['function_call_id']).get_call_graph())
            if not graph or any(g.status.name!='TERMINATED' for g in graph):raise ValueError('Retry handle not terminal:'+key)
    assert sum(row['status']=='retry_pending' for row in ledger.data['calls'].values())==36
    assert sum(row['status']=='complete' for row in ledger.data['calls'].values())==684
    completed={k for k,row in ledger.data['calls'].items() if row['status']=='complete'}
    planned={row['id'] for row in r['dry_run_ready_keys']}
    assert not completed&planned,'Completed input would be resubmitted'
    ledger.assert_can_launch()
    assert r['status']=='PASS' and r['saved_completed_inputs']==492
    r.update(source_identity=identity(),provider_ledger_canonical_sha256=canonical_sha(raw),hard_cap_usd='50',
             recovered_successes_included=4,original_scoring_matches_recorded_fitness=True,
             retry_handles_confirmed_terminal=36,completed_inputs_rescheduled=0,
             reconstructed_ledger_completed=sum(row['status']=='complete' for row in ledger.data['calls'].values()),
             restored_guard_usd=str(ledger.guard_total()))
    (out/'mac-replay-receipt.json').write_text(json.dumps(r,indent=2)+'\n')
    return r


def main():
    if platform.system()!='Darwin' or platform.machine()!='arm64':raise SystemExit('Resume requires the original arm64 Mac controller platform')
    if not os.environ.get('ORX_RUN_ID') or not os.environ.get('ORX_COMMIT'):raise SystemExit('Use tracked project runner')
    from stage2.registration import authorize,mode_record
    authorize(mode_record())
    import modal
    r=json.loads((ROOT/RECEIPT).read_text())
    if r['status']!='PASS' or r['source_identity']!=identity() or r['hard_cap_usd']!='50' or r['completed_inputs_rescheduled']!=0 or r['recovered_successes_included']!=4:
        raise ValueError('Current Mac replay/dedup/hard50 preflight has not passed')
    volume=modal.Volume.from_name('evolved-intrinsic-rewards-stage2',create_if_missing=False,version=2)
    current=json.loads(retry_io(lambda:b''.join(volume.read_file('/budget-ledger.json'))))
    if canonical_sha(current)!=r['provider_ledger_canonical_sha256']:raise ValueError('Provider ledger changed after Mac preflight')
    from stage2.controller import main as controller_main
    controller_main()

if __name__=='__main__':main()
