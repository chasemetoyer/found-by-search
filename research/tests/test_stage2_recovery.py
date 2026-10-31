"""Infrastructure recovery checks with fake providers and tiny synthetic state; no training."""
import copy
import json
from decimal import Decimal
from types import SimpleNamespace
import pytest
from stage2.budget import Stage2Ledger, BudgetStop
from stage2.recovery import retry_io, canonical, reconcile_interrupted, AMENDMENT


def test_retry_transport_then_success():
    calls=[]; waits=[]
    def operation():
        calls.append(1)
        if len(calls)<3: raise ConnectionError('dns')
        return 'ok'
    assert retry_io(operation, sleep=waits.append)=='ok'
    assert waits==[1,2]

@pytest.mark.parametrize('error',[ValueError('hash'), FileNotFoundError('missing')])
def test_never_retry_validation_or_missing(error):
    waits=[]
    with pytest.raises(type(error)): retry_io(lambda: (_ for _ in ()).throw(error),sleep=waits.append)
    assert not waits


def test_transport_retry_exhaustion():
    waits=[]
    with pytest.raises(ConnectionError):
        retry_io(lambda: (_ for _ in ()).throw(ConnectionError()),sleep=waits.append)
    assert waits==[1,2,4,8,16]


def fixture(tmp_path):
    ledger=Stage2Ledger(tmp_path/'ledger.json')
    for k in ['saved','terminated']:
        ledger.reserve(k,k,3600);ledger.associate(k,'fc-'+k,'ap-old')
    for k in ['saved','terminated']: ledger.uncertain(k,'stopped')
    a={'interrupted_ledger_canonical_sha256':canonical(ledger.data),'interrupted_calls':{
        'saved':{'function_call_id':'fc-saved','provider_status':'SUCCESS'},
        'terminated':{'function_call_id':'fc-terminated','provider_status':'TERMINATED'}}}
    p=tmp_path/AMENDMENT;p.parent.mkdir(parents=True)
    marker={'id':'saved','spec_sha256':'saved','outcome':'success','usage':{'compute_cost_estimate_usd':'0.01'}}
    a['interrupted_calls']['saved']['completion_marker_canonical_sha256']=canonical(marker)
    p.write_text(json.dumps(a))
    class Volume:
        def read_file(self,p): return [json.dumps(marker).encode()]
        def listdir(self,*args,**kw): return []
    def query(call): return [SimpleNamespace(status=SimpleNamespace(name='SUCCESS' if call=='fc-saved' else 'TERMINATED'))]
    return ledger,Volume(),query,p,marker


def test_recovery_preserves_charges_and_retries_only_terminated(tmp_path):
    ledger,volume,query,p,_=fixture(tmp_path);guard=ledger.guard_total();old=copy.deepcopy(ledger.data['calls']['terminated'])
    reconcile_interrupted(ledger,volume,tmp_path,root=tmp_path,query=query)
    assert ledger.data['calls']['saved']['status']=='complete'
    assert ledger.data['calls']['terminated']['status']=='retry_pending'
    assert ledger.guard_total()==guard
    with pytest.raises(BudgetStop): ledger.reserve('saved','saved',3600)
    new=ledger.reserve('terminated','terminated',3600)
    assert ledger.data['infrastructure_history'][0]['function_call_id']=='fc-terminated'
    assert ledger.guard_total()==guard+Decimal(new['reservation_usd'])
    ledger.associate('terminated','fc-new','ap-new');ledger.complete('terminated',10,{'compute_cost_estimate_usd':'0.001'})
    assert Decimal(ledger.data['infrastructure_history'][0]['guard_charge_usd'])==Decimal(old['guard_charge_usd'])
    assert ledger.guard_total()>guard
    # Idempotency and no expansion to future failed calls.
    ledger.uncertain('terminated','future outage');before=copy.deepcopy(ledger.data)
    reconcile_interrupted(ledger,volume,tmp_path,root=tmp_path,query=query)
    assert ledger.data==before


def test_recovery_refuses_changed_ledger(tmp_path):
    ledger,v,q,p,_=fixture(tmp_path);ledger.data['limit_usd']='34'
    with pytest.raises(ValueError,match='Ledger changed'):reconcile_interrupted(ledger,v,tmp_path,root=tmp_path,query=q)


def test_recovery_refuses_active_provider_and_makes_no_mutation(tmp_path):
    ledger,v,q,p,_=fixture(tmp_path);before=copy.deepcopy(ledger.data)
    active=lambda _: [SimpleNamespace(status=SimpleNamespace(name='RUNNING'))]
    with pytest.raises(ValueError,match='Provider state'):reconcile_interrupted(ledger,v,tmp_path,root=tmp_path,query=active)
    assert ledger.data==before


def test_recovery_refuses_output_identity_change(tmp_path):
    ledger,v,q,p,m=fixture(tmp_path);before=copy.deepcopy(ledger.data);m['spec_sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):reconcile_interrupted(ledger,v,tmp_path,root=tmp_path,query=q)
    assert ledger.data==before


def test_retry_requires_identical_spec_and_preserves_state(tmp_path):
    ledger,v,q,p,_=fixture(tmp_path);reconcile_interrupted(ledger,v,tmp_path,root=tmp_path,query=q)
    before=copy.deepcopy(ledger.data)
    with pytest.raises(BudgetStop,match='specification'):ledger.reserve('terminated','changed',3600)
    assert ledger.data==before


def test_retry_respects_cap_including_old_costs(tmp_path):
    ledger,v,q,p,_=fixture(tmp_path);reconcile_interrupted(ledger,v,tmp_path,root=tmp_path,query=q)
    ledger.data['calls']['saved']['guard_charge_usd']='34.40'
    before=copy.deepcopy(ledger.data)
    with pytest.raises(BudgetStop,match='cross'): ledger.reserve('terminated','terminated',3600)
    assert ledger.data==before


def test_real_scheduler_recovers_plan_without_retraining_completed_inputs(tmp_path, monkeypatch):
    from stage2.search import SearchPlan, SEARCH_PATH_SMOKE
    from stage2.scheduler import run_plan
    from stage2.common import canonical_sha
    from test_stage2_search import FakeBackend, hooks_for
    import stage2.registration
    monkeypatch.setattr(stage2.registration, 'mode_record', lambda: {'mode':'search','profile':'smoke'})
    plan=SearchPlan(SEARCH_PATH_SMOKE); backend=FakeBackend();store={}
    ledger=Stage2Ledger(tmp_path/'ledger.json');hooks=hooks_for(tmp_path,backend,store)
    # Seed one completed input and one explicitly reconciled retry-pending input.
    first,second=plan.ready_specs(ledger)[:2]
    for spec in (first,second):ledger.reserve(spec['id'],canonical_sha(spec),3600)
    call=backend.spawn(first,3600)
    result=None
    while result is None:result=backend.poll(call)
    directory=hooks.sync(result)
    ledger.complete(first['id'],1,result['usage']);plan.record(first['id'],result,directory)
    ledger.data['recovery_amendments']=['test-amendment']
    ledger.data['calls'][second['id']].update(status='retry_pending',function_call_id='fc-terminated',
        provider_terminal_status='TERMINATED',recovery_amendment='test-amendment')
    before=copy.deepcopy(ledger.data['calls'][second['id']])
    resumed=SearchPlan(SEARCH_PATH_SMOKE)
    run_plan(resumed,ledger,backend,hooks,32)
    assert len(backend.spawned)==len(set(backend.spawned))==72
    assert backend.spawned.count(first['id'])==1 and backend.spawned.count(second['id'])==1
    assert resumed.done(ledger)
    assert ledger.data['infrastructure_history'][0]['function_call_id']==before['function_call_id']


def test_arbitrary_retry_pending_is_rejected(tmp_path):
    ledger=Stage2Ledger(tmp_path/'ledger.json');ledger.reserve('x','x',3600)
    ledger.data['calls']['x']['status']='retry_pending'
    with pytest.raises(BudgetStop,match='authorization'):ledger.reserve('x','x',3600)
