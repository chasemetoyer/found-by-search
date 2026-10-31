from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pytest
from scripts.stage3_dashboard_sync_v2 import (LABEL,PROGRESS,Publisher,archive_legacy,completion_history,
    curated_panels,legacy_metrics,parent_flush_seen,step_path,synchronize)


def data():
    a='stage3_smoke-E-DoorKey-8x8-8192-seed5000'
    b='stage3_smoke-R-DoorKey-8x8-8192-seed5001'
    records=[{'method':m,'task':'DoorKey-8x8','seed':seed,'area':area,'solved':False,'cpu_class':'zen3',
              'metrics':[{'global_step':2048,'train/success':area}]} for m,seed,area in [('E',5000,.25),('R',5001,.75)]]
    ledger={'calls':{b:{'status':'complete','completed_utc':'2026-10-29T12:00:02+00:00'},
                     a:{'status':'complete','completed_utc':'2026-10-29T12:00:01+00:00'},
                     'stage3_science-T-DoorKey-8x8-2000000-seed2000':{'status':'complete'}},'guard_total_usd':'1.5','limit_usd':'15'}
    return records,ledger,a,b


def test_completion_axis_is_real_order_and_right_angle_without_epoch_steps():
    rows,ledger,a,b=data();series,counts,total=completion_history(ledger,rows,'smoke')
    assert total==2 and series['E/DoorKey-8x8']==[(0,0),(1,1),(2,1)]
    assert series['R/DoorKey-8x8']==[(0,0),(1,0),(2,1)]
    path=step_path(series['E/DoorKey-8x8'],lambda x:x,lambda y:y)
    assert path=='M0.000,0.000 H1.000 V1.000 H2.000 V1.000'
    assert ' L' not in path
    panels,proof=curated_panels(ledger,rows,'smoke',2)
    assert proof['x_extent']==[0,34] and proof['data_x_extent']==[0,2]
    assert 'Total runs completed' in panels[PROGRESS] and '1793' not in panels[PROGRESS]
    assert len(panels)==18 and '2048' in panels['learning/DoorKey-8x8']


def test_equal_timestamps_are_one_completion_group_no_invented_order():
    rows,ledger,a,b=data();ledger['calls'][b]['completed_utc']=ledger['calls'][a]['completed_utc']
    series,_,_=completion_history(ledger,rows,'smoke')
    assert series['E/DoorKey-8x8']==[(0,0),(2,1)]
    assert series['R/DoorKey-8x8']==[(0,0),(2,1)]


def test_wrong_or_unvalidated_results_fail_without_changing_source_data():
    rows,ledger,a,b=data();before=deepcopy((rows,ledger))
    with pytest.raises(ValueError):completion_history(ledger,rows[:1],'smoke')
    assert (rows,ledger)==before
    del ledger['calls'][a]['completed_utc']
    with pytest.raises(KeyError):completion_history(ledger,rows,'smoke')


def test_only_exact_legacy_scalar_names_detected():
    assert legacy_metrics([LABEL+'/progress/completed/E/DoorKey-8x8',LABEL+'/'+PROGRESS])==[LABEL+'/progress/completed/E/DoorKey-8x8']
    assert not legacy_metrics([LABEL+'/learning/DoorKey-8x8',LABEL+'/'+PROGRESS])


class Client:
    def __init__(self,remote):self.remote=remote
    def init(self,**kwargs):self.kwargs=kwargs;return self
    def Html(self,doc,caption):return {'_type':'trackio.html','caption':caption,'document':doc}
    def log(self,metrics,step):self.remote.logged=(metrics,step)
    def finish(self):pass


class Remote:
    def __init__(self,legacy=True):self.legacy=legacy;self.calls=[];self.logged=None
    def predict(self,**kwargs):
        self.calls.append(kwargs);api=kwargs['api_name']
        if api=='/get_run_summary':
            return {'num_logs':2,'metrics':[LABEL+'/progress/completed/E/DoorKey-8x8'] if self.legacy else list(self.logged[0]) if self.logged else []}
        if api=='/rename_run':self.legacy=False;return True
        if api=='/get_snapshot':return {k:[{'value':v}] for k,v in self.logged[0].items()}
        raise AssertionError(api)


def test_preserve_archive_then_media_only_under_original_identity(monkeypatch):
    monkeypatch.setenv('TRACKIO_PROJECT','evolved-intrinsic-rewards');monkeypatch.setenv('ORX_EXPERIMENT','stage3')
    monkeypatch.setenv('TRACKIO_SERVER_URL','https://fixture.invalid/private/fixture')
    rows,ledger,_,_=data();panels,proof=curated_panels(ledger,rows,'smoke',2)
    remote=Remote();client=Client(remote);publisher=Publisher('existing-run',True,remote,client)
    receipt=publisher.publish(panels,proof)
    assert receipt['status']=='PASS' and receipt['completed']==2 and receipt['legacy_numeric_metrics']==0
    assert len(remote.logged[0])==18 and remote.logged[1]==2
    assert all(isinstance(v,dict) and v['_type']=='trackio.html' for v in remote.logged[0].values())
    archive=next(c for c in remote.calls if c['api_name']=='/rename_run')
    assert archive['old_name']=='existing-run' and archive['new_name'].startswith('existing-run--legacy-ms-')
    assert client.kwargs['name']=='existing-run'
    assert not any('delete' in c['api_name'] for c in remote.calls)


def test_archive_requires_flag_and_parent_flush_requires_exact_identity():
    with pytest.raises(ValueError):archive_legacy(Remote(),'project','run',False)
    class Volume:
        def read_file(self,path):
            assert path=='/controller/tracking-final-flush.json'
            return [json.dumps({'run_id':'right','profile':'science'}).encode()]
    assert parent_flush_seen(Volume(),'right','science')
    assert not parent_flush_seen(Volume(),'wrong','science')
    assert not parent_flush_seen(Volume(),'right','smoke')


def test_sync_uses_only_read_sources_and_leaves_registered_files_unchanged(monkeypatch,tmp_path):
    import scripts.stage3_dashboard_sync_v2 as mod
    paths=[Path(p) for p in ('stage3/tracking.py','stage3/observer.py','stage3/controller.py','stage3/modal_app.py')]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows,ledger,_,_=data();frozen=deepcopy((rows,ledger))
    monkeypatch.setattr(mod,'collect',lambda *args:(rows,ledger,[]))
    class Publish:
        def publish(self,panels,proof):return {'status':'PASS','panels':len(panels)}
    receipt,_=synchronize(object(),Publish(),'smoke',2,{},tmp_path)
    assert receipt['panels']==18 and (rows,ledger)==frozen
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    source=Path(mod.__file__).read_text()
    assert '.batch_upload(' not in source and '.spawn(' not in source and '.terminate(' not in source
