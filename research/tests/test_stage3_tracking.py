from copy import deepcopy
import pytest
from stage3.tracking import LABEL,TrackioSink,curve_series,panel_html,snapshot_payload


def record(seed=5000,method='E',success=.5,**extra):
    r=dict(method=method,task='DoorKey-8x8',seed=seed,area=success,solved=False,cpu_class='zen3',
           metrics=[{'global_step':2048,'train/success':success,'entropy':.8}]);r.update(extra);return r


def test_task_mean_and_faint_seed_lines_only_use_complete_records():
    p=snapshot_payload([record(success=.2),record(5001,success=.8)],{'limit_usd':'15'})
    ss=curve_series(p['records'],'success');assert [s for s in ss if s['mean']][0]['points']==[(2048,.5)]
    assert len([s for s in ss if s['opacity']<1])==2
    assert p['completed_by_cell']['E/DoorKey-8x8']==2
    assert p['completed_by_cell']['R/UnlockPickup']==0


def test_duplicate_results_rejected_and_missing_telemetry_stays_missing():
    with pytest.raises(ValueError,match='Duplicate'):snapshot_payload([record(),record()],{})
    p=snapshot_payload([record(method='tamper_on')],{})
    assert curve_series(p['records'],'presses_step')==[]
    assert 'No completed-seed telemetry' in panel_html(p)['tamper/presses per step']


def test_every_required_panel_has_progress_label_and_intrinsic_reference():
    r=record(method='tamper_on');r['metrics'][0].update({'tamper/presses_per_step':.25,'train/tampers':2.,'train/int_ret':.6})
    p=snapshot_payload([r],{'guard_total_usd':'1.25','limit_usd':'15'})
    panels=panel_html(p)
    assert len(panels)==18
    assert all(LABEL in v for v in panels.values())
    assert 'P0 = 0.64' in panels['tamper/intrinsic episode return']
    assert all('learning/'+t in panels for t in ('DoorKey-8x8','UnlockPickup','KeyCorridorS3R2'))


class Client:
    def __init__(self,fail=False):self.fail=fail;self.kwargs=None;self.logged=None
    def init(self,**kwargs):self.kwargs=kwargs;return self
    def Html(self,doc,caption):return {'doc':doc,'caption':caption}
    def log(self,metrics,step):
        if self.fail:raise RuntimeError('credential=SECRET must not escape')
        self.logged=(metrics,step)
    def finish(self):pass


def environment():return {'TRACKIO_PROJECT':'evolved-intrinsic-rewards','ORX_RUN_ID':'smoke-run','ORX_EXPERIMENT':'stage3','TRACKIO_SPACE_ID':'owner/private-space'}


def test_native_visualize_identity_and_private_remote_logging_without_mutating_inputs():
    p=snapshot_payload([record()],{'limit_usd':'15'});before=deepcopy(p);c=Client();sink=TrackioSink(environ=environment(),client=c)
    assert sink.publish(p,7)['ok']
    assert c.kwargs['name']=='smoke-run' and c.kwargs['group']=='stage3' and c.kwargs['private'] is True
    assert c.kwargs['embed'] is False and c.kwargs['resume']=='allow'
    assert c.logged[1]==7 and len(c.logged[0])>18 and p==before


def test_tracking_failure_cannot_raise_or_expose_credentials():
    p=snapshot_payload([],{});sink=TrackioSink(environ=environment(),client=Client(fail=True))
    result=sink.publish(p,1);assert result=={'ok':False,'error_type':'RuntimeError','step':1}
    assert 'SECRET' not in str(result)


@pytest.mark.parametrize('env',[{},dict(environment(),TRACKIO_SPACE_ID='',TRACKIO_SERVER_URL='http://localhost:7860')])
def test_no_local_fallback_when_remote_is_unconfigured(env):
    assert TrackioSink(environ=env,client=Client()).publish(snapshot_payload([],{}),0)['ok'] is False


def test_private_prefix_gates_all_reads_and_writes_and_keeps_rootpath():
    import asyncio
    import httpx
    from stage3.tracking_server import CapabilityPrefix
    calls=[];commits=[]
    async def app(scope,receive,send):
        calls.append(scope)
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':b'ok'})
    token='x'*48
    wrapped=CapabilityPrefix(app,token,after_write=lambda:commits.append(True))
    async def test():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=wrapped),base_url='https://test') as c:
            for path in ['/','/api/project','/file?path=/tracking/data','/private/'+token+'-extra/api/project']:
                assert (await c.get(path)).status_code==404
            assert (await c.get('/private/'+token+'/api/project')).status_code==200
            assert calls[-1]['path']=='/api/project' and calls[-1]['root_path']=='/private/'+token
            assert not commits
            assert (await c.post('/private/'+token+'/api/log')).status_code==200
            assert commits==[True]
    asyncio.run(test())


def test_private_server_refuses_empty_or_short_read_credentials():
    from stage3.tracking_server import CapabilityPrefix
    with pytest.raises(ValueError):CapabilityPrefix(None,'')
    with pytest.raises(ValueError):CapabilityPrefix(None,'short')


def test_library_stdout_and_stderr_cannot_echo_private_endpoint(capsys,caplog):
    import sys
    class Noisy(Client):
        def init(self,**kwargs):
            print('https://host/private/SECRET');print('SECRET',file=sys.stderr)
            return super().init(**kwargs)
        def log(self,metrics,step):
            import logging
            logging.warning('https://host/private/SECRET')
            print('SECRET');return super().log(metrics,step)
        def finish(self):print('SECRET')
    sink=TrackioSink(environ=environment(),client=Noisy())
    assert sink.publish(snapshot_payload([],{}),0)['ok'];assert sink.finish()['ok']
    output=capsys.readouterr();assert 'SECRET' not in output.out+output.err+caplog.text


def test_modal_server_factory_declares_serializable_singleton_without_launch(monkeypatch):
    import modal
    from stage3.tracking_server import APP_NAME,VOLUME_NAME,make_tracking_app
    monkeypatch.setenv('TRACKIO_ACCESS_PATH_TOKEN','fixture'*8)
    monkeypatch.setenv('TRACKIO_WRITE_TOKEN','fixture'*8)
    app,function,volume=make_tracking_app(modal.Image.debian_slim())
    assert app.name==APP_NAME and volume.name==VOLUME_NAME
    assert function is not None  # Definition construction catches unsupported local-function closures.


def test_missing_success_is_zero_for_mean_but_other_telemetry_remains_missing():
    a=record(seed=5000,success=.8);b=record(seed=5001,success=0)
    b['metrics']=[{'global_step':2048}]
    ss=curve_series(snapshot_payload([a,b],{})['records'],'success')
    assert [s for s in ss if s['mean']][0]['points']==[(2048,.4)]
    assert len([s for s in ss if not s['mean']])==2
    assert curve_series([b],'presses_episode')==[]
    assert curve_series([b],'entropy')==[]
    assert 'train/success' not in b['metrics'][0]


def test_numerical_failure_panel_covers_all_training_methods():
    p=snapshot_payload([],{'candidate_failures':3})
    panel=panel_html(p)['progress/failures and restarts']
    assert 'Numerical training failures' in panel and 'Candidate failures' not in panel


def test_remote_readback_checks_exact_final_cell_counts_and_all_panels():
    from stage3.tracking import remote_readback
    payload=snapshot_payload([record()],{})
    panels={f'{LABEL}/{name}' for name in panel_html(payload)}
    scalars={f'{LABEL}/progress/completed/{k}':v for k,v in payload['completed_by_cell'].items()}
    scalars[f'{LABEL}/progress/completed_total']=1
    class Remote:
        def __init__(self,wrong=False):self.wrong=wrong
        def predict(self,**kw):
            if kw['api_name']=='/get_run_summary':return {'num_logs':1,'metrics':list(panels)+list(scalars),'last_step':42}
            assert kw['api_name']=='/get_snapshot' and kw['step']==42
            result={name:[{'value':{'_type':'trackio.html','caption':LABEL}}] for name in panels}
            result.update({k:[{'value':v}] for k,v in scalars.items()})
            if self.wrong:result[f'{LABEL}/progress/completed/E/DoorKey-8x8']=[{'value':0}]
            return result
    good=remote_readback(Remote(),'project','run',payload,42)
    assert good['ok'] and good['completed_total']==1 and good['panel_count']==18
    bad=remote_readback(Remote(True),'project','run',payload,42)
    assert not bad['ok'] and bad['completed_scalar_mismatches']==[f'{LABEL}/progress/completed/E/DoorKey-8x8']


def test_observer_counts_physical_failures_and_restarts_including_active_calls():
    from types import SimpleNamespace
    import json
    from stage3.observer import collect
    a='stage3_smoke-E-DoorKey-8x8-8192-seed5000';b='stage3_smoke-R-DoorKey-8x8-8192-seed5001'
    ledger={'calls':{a:{'status':'running','attempt_count':1},b:{'status':'complete','attempt_count':1}}}
    files={'/budget-ledger.json':json.dumps(ledger).encode()};entries=[]
    env=json.dumps({'cpu':{'vendor_id':'AuthenticAMD','cpu_family':25,'model':1}}).encode()
    for key,attempt,names in [(a,'a1',['attempt.json','environment.json','infrastructure_failure.json']),
                              (a,'a2',['attempt.json','environment.json']),(a,'a3',['attempt.json']),
                              (b,'b1',['attempt.json','environment.json','infrastructure_failure.json']),
                              (b,'b2',['attempt.json','environment.json'])]:
        for name in names:
            path=f'/runs/{key}/attempts/{attempt}/{name}';entries.append(SimpleNamespace(path=path))
            if name=='environment.json':files[path]=env
    class Volume:
        def read_file(self,path):return [files[path]]
        def listdir(self,*args,**kwargs):return entries
    cache={b:record(seed=5001,method='R')}
    rows,observed,errors=collect(Volume(),'smoke',cache)
    assert observed['infrastructure_failures']==2
    assert observed['observed_restarts']==3 and observed['physical_attempts_observed']==5
    assert observed['physical_cpu_mix']=={'zen3':4,'unknown':1}
    assert json.loads(files['/budget-ledger.json'])==ledger and not errors


def test_final_flush_receipt_is_safe_and_does_not_depend_on_training(monkeypatch):
    from stage3 import observer
    records=[record()];ledger={'calls':{},'limit_usd':'15','physical_attempts_observed':1,
                              'physical_cpu_mix':{'zen3':1},'infrastructure_failures':0,'observed_restarts':0}
    monkeypatch.setattr(observer,'collect',lambda *args:(records,ledger,[]))
    class Sink:
        def publish(self,payload,step):return {'ok':True,'queued':True}
        def finish(self):return {'ok':True}
        def verify_remote(self,payload,step):return {'ok':True,'completed_total':1,'panel_count':18}
    receipt=observer.observe_once(None,Sink(),'smoke',2,finish=True,verify=True,step=42)
    assert receipt['completed_records']==1 and receipt['status']=='FAIL'  # incomplete matrix cannot pass.
    assert receipt['scientific_results_mutated'] is False and receipt['budget_ledger_mutated'] is False
    class Fail(Sink):
        def verify_remote(self,payload,step):return {'ok':False,'error_type':'TimeoutError'}
    failed=observer.observe_once(None,Fail(),'smoke',2,finish=True,verify=True,step=42)
    assert failed['status']=='FAIL' and failed['training_dependency'] is False


def test_final_flush_complete34_counts_pass_and_only_writes_control_receipt(monkeypatch,tmp_path):
    from stage3 import observer
    records=[]
    for method in ('E','R','T','N','S','tamper_on','tamper_off'):
        tasks=('DoorKey-8x8',) if method.startswith('tamper_') else ('DoorKey-8x8','UnlockPickup','KeyCorridorS3R2')
        for task in tasks:
            records.extend(record(seed=seed,method=method,task=task) for seed in (5000,5001))
    ledger={'calls':{},'limit_usd':'15','physical_attempts_observed':34,'physical_cpu_mix':{'zen3':34},'infrastructure_failures':0,'observed_restarts':0}
    monkeypatch.setattr(observer,'collect',lambda *args:(records,ledger,[]))
    order=[]
    class Sink:
        def publish(self,payload,step):order.append('publish');return {'ok':True}
        def finish(self):order.append('finish');return {'ok':True}
        def verify_remote(self,payload,step):
            order.append('verify');assert sum(payload['completed_by_cell'].values())==34
            return {'ok':True,'completed_total':34,'panel_count':18}
    receipt=observer.observe_once(None,Sink(),'smoke',2,finish=True,verify=True,step=42)
    assert receipt['status']=='PASS' and receipt['complete_snapshot'] and order==['publish','finish','verify']
    writes=[]
    class Batch:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def put_file(self,local,remote):writes.append((local,remote))
    class Volume:
        def batch_upload(self,force):return Batch()
    observer.save_flush_receipt(Volume(),receipt,tmp_path)
    assert len(writes)==1 and writes[0][1]=='/controller/tracking-final-flush.json'
    assert 'budget-ledger' not in str(writes) and receipt['budget_ledger_mutated'] is False


def test_old_tracking_lease_cannot_stop_a_new_deployment():
    from stage3.tracking_server import stop_if_current_lease
    stops=[]
    assert stop_if_current_lease('old',{'lease_id':'new'},lambda:stops.append(1))['status']=='superseded'
    assert stops==[]
    assert stop_if_current_lease('new',{'lease_id':'new'},lambda:stops.append(1))['status']=='stop_requested'
    assert stops==[1]


def test_tracking_guardian_targets_the_deployed_monitor_and_preserves_volume():
    import ast
    from pathlib import Path
    from stage3 import tracking_server
    root=Path(__file__).resolve().parents[1]
    controller=ast.parse((root/'stage3/controller.py').read_text())
    deployed=[keyword.value.value for call in ast.walk(controller)
              if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
              and call.func.attr=='deploy' for keyword in call.keywords
              if keyword.arg=='name' and isinstance(keyword.value,ast.Constant)]
    assert deployed==[tracking_server.APP_NAME]
    assert tracking_server.APP_NAME=='evolved-intrinsic-rewards-stage3-monitor'
    assert tracking_server.VOLUME_NAME=='evolved-intrinsic-rewards-stage3-tracking'
    # Verify the real expiry function resolves that same deployment constant,
    # not an unrelated app name that would leave the warm service running.
    server=ast.parse(Path(tracking_server.__file__).read_text())
    expiry=next(node for node in ast.walk(server) if isinstance(node,ast.FunctionDef)
                and node.name=='expire_tracking')
    lookups=[node for node in ast.walk(expiry) if isinstance(node,ast.Call)
             and isinstance(node.func,ast.Attribute) and node.func.attr=='lookup']
    assert len(lookups)==1
    assert isinstance(lookups[0].args[0],ast.Name) and lookups[0].args[0].id=='APP_NAME'
