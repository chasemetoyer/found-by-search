"""Approved cloud entry point. Verification never starts a controller or training input."""
from __future__ import annotations
import hashlib,json,os,platform,time
from pathlib import Path
from decimal import Decimal
from stage2.common import ROOT
from stage2.transport import retry_io
from stage2.search import SearchPlan

AMENDMENT='preregistration/stage2-cloud-amendment-20261028.json'
CONFIG='stage2/cloud-controller.json'
FITNESS='handoff-cross-platform/cloud-controller/recorded-fitness.json'
REPLACEMENTS={'stage2/controller.py','stage2/budget.py','stage2/registration.py','stage2/common.py','scripts/run_stage1.sh','stage2/modal_app.py'}


def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return digest(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def configuration(root=ROOT):
    c=json.loads((root/CONFIG).read_text())
    if c not in ({'phase':p,'controller_timeout_seconds':43200,'monetary_cap_enabled':False,'auto_restart':False} for p in ['verify','search']):
        raise ValueError('Unapproved cloud configuration')
    return c


def authorized_hashes(lock,root):
    from stage2.recovery_v2 import authorized_hashes as previous,AMENDMENT as prior
    effective=previous(lock,root)
    path=root/AMENDMENT
    if not path.exists():return effective
    a=json.loads(path.read_text())
    if a['approval_seq']!=30 or a['scientific_changes'] is not False:
        raise ValueError('Cloud migration is not authorized')
    if a['prior_amendment_sha256']!=digest((root/prior).read_bytes()):
        raise ValueError('Prior amendment changed')
    if set(a['implementation_sha256'])!=REPLACEMENTS:raise ValueError('Unexpected cloud replacement scope')
    for name in REPLACEMENTS:
        if digest((root/'preregistration/frozen-pre-cloud-stage2'/name).read_bytes())!=effective[name]:
            raise ValueError('Pre-cloud runtime not preserved: '+name)
    for name,expected in a['added_file_sha256'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or digest((root/p).read_bytes())!=expected:
            raise ValueError('Cloud helper/data changed: '+name)
    c=configuration(root)
    if canonical(c)!=a['allowed_config_sha256'][c['phase']]:raise ValueError('Cloud profile changed')
    effective.update(a['implementation_sha256'])
    return effective


def source_identity(root=ROOT):
    a=json.loads((root/AMENDMENT).read_text())
    files={**a['implementation_sha256'],**a['added_file_sha256']}
    return canonical({p:digest((root/p).read_bytes()) for p in sorted(files)})


class ResumeSearchPlan(SearchPlan):
    """Preserve already-recorded fitness during replay; new inputs use original scoring."""
    def __init__(self,cfg,recorded=None):
        super().__init__(cfg)
        self.recorded=json.loads((ROOT/FITNESS).read_text()) if recorded is None else recorded
    def record(self,key,result,directory):
        super().record(key,result,directory)  # unchanged identity/evidence/schedule checks
        if key in self.recorded['run_areas']:
            self.areas[key]=self.recorded['run_areas'][key]
    def cma_fitness(self,g,i):
        computed=super().cma_fitness(g,i)  # still requires every actual observation
        return self.recorded['candidate_fitness'].get(f'{g}:{g*self.cfg.population+i}',computed)


def require_receipt(volume,root=ROOT):
    r=json.loads(retry_io(lambda:b''.join(volume.read_file('/controller/preflight/latest.json'))))
    if r['status']!='PASS' or r['source_identity']!=source_identity(root):raise ValueError('Cloud replay did not pass for this runtime')
    if r['machine']!='x86_64' or r['provider_submissions']!=0 or r['dry_run_saved_keys_resubmitted']:
        raise ValueError('Cloud replay/dry-run receipt is invalid')
    raw=json.loads(retry_io(lambda:b''.join(volume.read_file('/budget-ledger.json'))))
    if canonical(raw)!=r['ledger_canonical_sha256']:raise ValueError('Shared ledger changed after cloud replay')
    return r


def apply_uncapped_accounting(ledger,owner,volume):
    from stage1.budget import utc_now
    if ledger.data.get('limit_usd') is not None:
        ledger.data.setdefault('limit_amendments',[]).append({'utc':utc_now(),'previous_limit_usd':ledger.data['limit_usd'],
            'limit_usd':None,'approval':28,'threshold_waiver_approval':30})
        ledger.data.update(historical_limit_usd=ledger.data['limit_usd'],limit_usd=None,cap_removal_approval=28)
    rate=Decimal('0.00003942')+2*Decimal('0.00000667')
    ledger.data.setdefault('controller_sessions',{}).setdefault(owner,{
        'guard_charge_usd':str(rate*43200),'host':'modal_sandbox','timeout_seconds':43200,
        'started_utc':utc_now(),'cost_basis':'Protective 1physical-core/2GiB Sandbox envelope for12h; not invoice or actual limit',
        'cpuinfo':Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else None,
        'restart_after_stop_authorized':False})
    # Preflight costs are separate from scientific inputs and cannot disappear.
    receipt=json.loads(retry_io(lambda:b''.join(volume.read_file('/controller/preflight/latest.json'))))
    ledger.data['controller_sessions'].setdefault(receipt['run_id'],{
        'guard_charge_usd':'0.10','host':'modal_sandbox_preflight','run_id':receipt['run_id'],
        'cost_basis':'Retained verification/setup allowance; measured/billed cost may be delayed'})
    ledger.save()


def main():
    if not os.environ.get('ORX_RUN_ID') or not os.environ.get('ORX_COMMIT'):raise SystemExit('Use the tracked project runner')
    if platform.system()!='Linux' or os.environ.get('STAGE2_CLOUD_CONTROLLER')!='1':
        raise SystemExit('Cloud controller must execute in the approved remote Linux Sandbox')
    from stage2.registration import authorize,mode_record
    authorize(mode_record())
    import modal
    volume=modal.Volume.from_name('evolved-intrinsic-rewards-stage2',create_if_missing=False,version=2)
    c=configuration()
    if c['phase']=='verify':
        from scripts.stage2_cloud_preflight import run
        out=ROOT/'results/cloud-preflight'
        r=run(out,volume=volume)
        r['source_identity']=source_identity()
        r['ledger_canonical_sha256']=canonical(json.loads(retry_io(lambda:b''.join(volume.read_file('/budget-ledger.json')))))
        # No ledger/controller mutation; only this verification receipt is uploaded.
        path=out/'replay-check.json';path.write_text(json.dumps(r,indent=2)+'\n')
        def upload():
            with volume.batch_upload(force=True) as batch:
                batch.put_file(path,'/controller/preflight/latest.json')
                batch.put_file(path,'/controller/preflight/'+os.environ['ORX_RUN_ID']+'.json')
        retry_io(upload)
        print(json.dumps({'event':'cloud_preflight_complete','status':r['status'],'provider_submissions':0}),flush=True)
        if r['status']!='PASS':raise SystemExit(3)
        return
    require_receipt(volume)
    from stage2.controller import main as controller_main
    controller_main()

if __name__=='__main__':main()
