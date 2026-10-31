"""Cloud migration safety checks; fake SDK only, no training or cloud calls."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from stage2.budget import Stage2Ledger,BudgetStop
from stage2 import cloud


def test_uncapped_accounting_retains_costs_and_uncertainty_checks(tmp_path):
    ledger=Stage2Ledger(tmp_path/'ledger.json');ledger.reserve('old','hash',3600)
    ledger.data['calls']['old'].update(status='complete',guard_charge_usd='40')
    volume=SimpleNamespace(read_file=lambda p:[json.dumps({'run_id':'probe'}).encode()])
    cloud.apply_uncapped_accounting(ledger,'controller',volume)
    assert ledger.data['limit_usd'] is None and ledger.data['cap_removal_approval']==28
    assert ledger.guard_total()>40
    before=ledger.guard_total();cloud.apply_uncapped_accounting(ledger,'controller',volume)
    assert ledger.guard_total()==before and len(ledger.data['limit_amendments'])==1
    ledger.reserve('new','hash2',3600)  # exceeds the old35cap but scientific bounds unchanged
    with pytest.raises(BudgetStop,match='already exists'):ledger.reserve('old','hash',3600)
    ledger.uncertain('new','unknown submission')
    with pytest.raises(BudgetStop,match='Uncertain'):ledger.reserve('third','hash3',3600)
    reopened=Stage2Ledger(tmp_path/'ledger.json');assert reopened.data['limit_usd'] is None


def test_only_approved_cloud_profiles(tmp_path):
    p=tmp_path/cloud.CONFIG;p.parent.mkdir(parents=True)
    for phase in ['verify','search']:
        c={'phase':phase,'controller_timeout_seconds':43200,'monetary_cap_enabled':False,'auto_restart':False}
        p.write_text(json.dumps(c));assert cloud.configuration(tmp_path)==c
    c['auto_restart']=True;p.write_text(json.dumps(c))
    with pytest.raises(ValueError):cloud.configuration(tmp_path)


@pytest.mark.parametrize('status',['PASS','FAIL'])
def test_verification_never_starts_controller_or_training(tmp_path,monkeypatch,status):
    import modal,stage2.registration,stage2.controller,scripts.stage2_cloud_preflight
    monkeypatch.setattr(cloud,'ROOT',tmp_path)
    monkeypatch.setattr(cloud.platform,'system',lambda:'Linux')
    for k,v in [('ORX_RUN_ID','probe-id'),('ORX_COMMIT','commit'),('STAGE2_CLOUD_CONTROLLER','1')]:monkeypatch.setenv(k,v)
    monkeypatch.setattr(cloud,'configuration',lambda:{'phase':'verify'})
    monkeypatch.setattr(cloud,'source_identity',lambda:'identity')
    monkeypatch.setattr(stage2.registration,'authorize',lambda r:None)
    monkeypatch.setattr(stage2.controller,'main',lambda:pytest.fail('Verification cannot start controller'))
    uploaded=[]
    class Batch:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def put_file(self,p,remote):uploaded.append(remote)
    volume=SimpleNamespace(read_file=lambda p:[b'{"calls":{}}'],batch_upload=lambda **kw:Batch())
    monkeypatch.setattr(modal.Volume,'from_name',lambda *a,**kw:volume)
    def probe(out,volume):
        out.mkdir(parents=True)
        return {'status':status,'provider_submissions':0,'machine':'x86_64'}
    monkeypatch.setattr(scripts.stage2_cloud_preflight,'run',probe)
    if status=='FAIL':
        with pytest.raises(SystemExit) as error:cloud.main()
        assert error.value.code==3
    else:cloud.main()
    assert uploaded==['/controller/preflight/latest.json','/controller/preflight/probe-id.json']


def test_bad_receipt_blocks_search_before_mutation(monkeypatch):
    volume=SimpleNamespace(read_file=lambda p:[b'{"status":"FAIL","source_identity":"same"}'])
    with pytest.raises(ValueError,match='did not pass'):cloud.require_receipt(volume)


def test_frozen_fitness_is_not_a_license_to_skip_missing_observations():
    from stage2.search import SEARCH_PATH_SMOKE
    p=cloud.ResumeSearchPlan(SEARCH_PATH_SMOKE,recorded={'run_areas':{},'candidate_fitness':{'0:0':.7}})
    with pytest.raises(KeyError):p.cma_fitness(0,0)


def test_worker_bundle_contains_frozen_fitness_and_registered_cloud_runtime():
    from stage2.modal_app import excluded
    from stage2.cloud import FITNESS,REPLACEMENTS
    assert not excluded(Path('handoff-cross-platform/cloud-controller'))
    assert not excluded(Path(FITNESS))
    assert excluded(Path('handoff-cross-platform/cloud-controller/terminated-provider-calls.csv'))
    assert all(not excluded(Path(p)) for p in REPLACEMENTS if p!='scripts/run_stage1.sh')
