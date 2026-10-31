"""No training: simulate transport failures at actual polling and persistence boundaries."""
import json
from types import SimpleNamespace
import pytest
import modal
from grpclib.exceptions import GRPCError
from grpclib.const import Status
from stage2.transport import retry_io, transient
from stage1.budget import BudgetStop


def test_retry_polling_same_handle(monkeypatch):
    import stage2.transport as t
    from stage2.controller import ModalBackend
    original=t.retry_io
    sleeps=[]
    monkeypatch.setattr(t,'retry_io',lambda op,**kw:original(op,**kw,sleep=sleeps.append,clock=lambda:10))
    class Call:
        object_id='fc-original'
        def __init__(self):self.n=0
        def get(self,timeout):
            assert timeout==0
            self.n+=1
            if self.n<=2:raise modal.exception.ConnectionError('Deadline exceeded')
            return {'result':'unchanged'}
        def cancel(self,*a,**kw):pytest.fail('Polling must not cancel or respawn')
    call=Call();b=ModalBackend('same-commit','same-run');b.deadlines[call.object_id]=100
    assert b.poll(call)=={'result':'unchanged'}
    assert call.n==3 and sleeps==[1,2] and b.deadlines[call.object_id]==100


def test_unready_poll_is_not_transport_failure():
    from stage2.controller import ModalBackend
    call=SimpleNamespace(object_id='fc-unready',get=lambda **kw:(_ for _ in ()).throw(TimeoutError()))
    assert ModalBackend('commit','run').poll(call) is None


def test_poll_retries_enforce_original_deadline(monkeypatch):
    import stage2.transport as t
    from stage2.controller import ModalBackend
    original=t.retry_io;clock=[10];calls=[]
    def sleep(seconds):clock[0]+=seconds
    monkeypatch.setattr(t,'retry_io',lambda op,**kw:original(op,**kw,sleep=sleep,clock=lambda:clock[0]))
    def get(**kw):calls.append(1);raise modal.exception.ConnectionError('offline')
    b=ModalBackend('commit','run');b.deadlines['fc']=12
    with pytest.raises(BudgetStop,match='deadline'):b.poll(SimpleNamespace(object_id='fc',get=get))
    assert clock[0]==12 and len(calls)==2 and b.deadlines['fc']==12


def test_persistence_retries_identical_ledger_without_new_submission(tmp_path,monkeypatch):
    import stage2.transport as t
    import stage2.controller as c
    original=t.retry_io
    monkeypatch.setattr(t,'retry_io',lambda op,**kw:original(op,**kw,sleep=lambda s:None))
    path=tmp_path/'ledger.json';payload={'calls':{'input':{'function_call_id':'fc-original','guard_charge_usd':'1'}}}
    saves=[];uploads=[]
    def save():saves.append(1);path.write_text(json.dumps(payload))
    class Batch:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def put_file(self,local,remote):
            uploads.append((local.read_bytes(),remote))
            if len(uploads)==1:raise modal.exception.ConnectionError('write response lost')
    monkeypatch.setattr(c,'volume',SimpleNamespace(batch_upload=lambda **kw:Batch()))
    controller=c.Controller.__new__(c.Controller)
    controller.ledger=SimpleNamespace(path=path,save=save)
    controller.publish=lambda:None
    controller.persist()
    assert len(saves)==1 and len(uploads)==2 and uploads[0]==uploads[1]
    assert payload['calls']['input']['function_call_id']=='fc-original'

@pytest.mark.parametrize('error',[ValueError('hash'),PermissionError(),FileNotFoundError(),
                                  modal.exception.RemoteError('worker error'),GRPCError(Status.INVALID_ARGUMENT,'bad')])
def test_permanent_and_worker_errors_fail_closed(error):
    waits=[]
    with pytest.raises(type(error)):retry_io(lambda:(_ for _ in ()).throw(error),sleep=waits.append)
    assert not waits


def test_bounded_poll_failure_exhaustion():
    waits=[]
    with pytest.raises(modal.exception.ConnectionError):
        retry_io(lambda:(_ for _ in ()).throw(modal.exception.ConnectionError('offline')),sleep=waits.append)
    assert waits==[1,2,4,8,16]


def test_only_transient_grpc_codes_retry():
    assert transient(GRPCError(Status.UNAVAILABLE,'network'))
    assert transient(GRPCError(Status.DEADLINE_EXCEEDED,'network'))
    assert not transient(GRPCError(Status.PERMISSION_DENIED,'auth'))
