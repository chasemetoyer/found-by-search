"""Verify new-outage authorization composes with previous charged/pending retries."""
import copy,json,shutil
from pathlib import Path
import pytest
from stage2.recovery_v2 import AMENDMENT, reconcile_interrupted, canonical, authorized_hashes
from test_stage2_recovery import fixture


def test_second_recovery_preserves_previous_pending_and_history(tmp_path):
    ledger,volume,query,p,marker=fixture(tmp_path)
    prior=copy.deepcopy(ledger.data['calls']['terminated'])
    prior.update(status='retry_pending',function_call_id='fc-prior',recovery_amendment='prior',provider_terminal_status='TERMINATED')
    ledger.data['calls']['previous-pending']=prior
    ledger.data['infrastructure_history']=[dict(prior,logical_key='historical')]
    ledger.data['recovery_amendments']=['prior']
    a=json.loads(p.read_text());a['interrupted_ledger_canonical_sha256']=canonical(ledger.data)
    (tmp_path/AMENDMENT).write_text(json.dumps(a))
    guard=ledger.guard_total()
    reconcile_interrupted(ledger,volume,tmp_path,root=tmp_path,query=query)
    assert ledger.data['calls']['previous-pending']==prior
    assert len(ledger.data['infrastructure_history'])==1 and ledger.guard_total()==guard
    assert ledger.data['calls']['saved']['status']=='complete'
    assert ledger.data['calls']['terminated']['status']=='retry_pending'
    assert len(ledger.data['recovery_amendments'])==2


def test_chained_authorization_requires_prior_versions_and_helpers(tmp_path):
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'preregistration/stage2-lock.json').read_text())
    old=json.loads((root/'preregistration/stage2-recovery-amendment-20261028.json').read_text())
    new=json.loads((root/AMENDMENT).read_text())
    names={'preregistration/stage2-lock.json','preregistration/stage2-recovery-amendment-20261028.json',AMENDMENT}
    names |= set(old['added_file_sha256'])|set(new['added_file_sha256'])
    names |= {'preregistration/frozen-original-stage2/'+p for p in old['implementation_sha256']}
    names |= {'preregistration/frozen-recovery1-stage2/'+p for p in new['implementation_sha256']}
    for name in names:
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
    assert authorized_hashes(lock,tmp_path)['stage2/controller.py']==new['implementation_sha256']['stage2/controller.py']
    (tmp_path/'preregistration/frozen-recovery1-stage2/stage2/controller.py').write_text('changed')
    with pytest.raises(ValueError,match='not preserved'):authorized_hashes(lock,tmp_path)
