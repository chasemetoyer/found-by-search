"""Hard50USD cap and budget-only ledger normalization checks. No training/provider calls."""
import copy,json
from pathlib import Path
from decimal import Decimal
import pytest
from stage2.budget import Stage2Ledger,BudgetStop,LIMIT_USD
from stage2 import mac
from stage2.common import canonical_sha

FIELDS=['limit_usd','limit_amendments','updated_utc','guard_total_usd','cap_restore_approval','monetary_cap_enabled','controller_sessions']


def fixture(tmp_path):
    p=tmp_path/'ledger.json';l=Stage2Ledger(p)
    l.data['limit_usd']='35';l.data.pop('cap_restore_approval',None);l.data.pop('monetary_cap_enabled',None)
    l.save();original=copy.deepcopy(l.data)
    a={'original_budget_fields':{k:original[k] for k in FIELDS if k in original},'allowed_accounting_fields':FIELDS,'original_ledger_canonical_sha256':canonical_sha(original)}
    q=tmp_path/mac.AMENDMENT;q.parent.mkdir(parents=True);q.write_text(json.dumps(a))
    l=Stage2Ledger(p);mac.account_external(l)
    return l,original


def test_hard50_restores_legacy_cap_without_changing_inputs(tmp_path):
    l,old=fixture(tmp_path)
    assert LIMIT_USD==Decimal(50) and l.data['limit_usd']=='50'
    assert l.data['calls']==old['calls'] and l.data['cap_restore_approval']==32
    assert mac.normalized_prebudget(l.data,tmp_path)==old
    assert l.guard_total()==Decimal(old['guard_total_usd'])+Decimal('.20')
    assert Stage2Ledger(l.path).data['limit_usd']=='50'


def test_legacy_uncapped_ledger_restores_hard50(tmp_path):
    p=tmp_path/'ledger.json';l=Stage2Ledger(p);l.data['limit_usd']=None;l.data['cap_removal_approval']=28;l.save()
    l=Stage2Ledger(p);assert l.data['limit_usd']=='50'
    assert l.data['limit_amendments'][-1]['previous_limit_usd'] is None


def test_no_reservation_can_cross50(tmp_path):
    l=Stage2Ledger(tmp_path/'ledger.json');l.data['controller_sessions']={'past':{'guard_charge_usd':'49.49'}}
    before=copy.deepcopy(l.data)
    with pytest.raises(BudgetStop,match='cross'):l.reserve('new','hash',3600)
    assert l.data==before


def test_at50_stops_and_does_not_reset_costs(tmp_path):
    l=Stage2Ledger(tmp_path/'ledger.json');l.data['controller_sessions']={'past':{'guard_charge_usd':'49.50'}}
    with pytest.raises(BudgetStop,match='reached'):l.assert_can_launch()


def test_uncapped_metadata_cannot_disable_runtime50(tmp_path):
    l=Stage2Ledger(tmp_path/'ledger.json');l.data['limit_usd']=None;l.data['controller_sessions']={'past':{'guard_charge_usd':'49.50'}}
    with pytest.raises(BudgetStop,match='reached'):l.assert_can_launch()


@pytest.mark.parametrize('change',[lambda d:d['calls'].update({'foreign':{'status':'complete'}}),
 lambda d:d.update(stage=3),lambda d:d.update(infrastructure_history=[{'new':'unknown'}])])
def test_budget_normalization_rejects_nonbudget_changes(tmp_path,change):
    l,old=fixture(tmp_path);change(l.data)
    with pytest.raises(ValueError,match='Non-budget'):mac.normalized_prebudget(l.data,tmp_path)


def test_normalization_rejects_hidden_cost_records(tmp_path):
    l,old=fixture(tmp_path);l.data['controller_sessions']['other']={'guard_charge_usd':'1'}
    with pytest.raises(ValueError,match='Unexpected controller'):mac.normalized_prebudget(l.data,tmp_path)


def test_duplicate_and_uncertain_guards_remain(tmp_path):
    l=Stage2Ledger(tmp_path/'ledger.json');l.reserve('x','hash',3600)
    with pytest.raises(BudgetStop,match='already exists'):l.reserve('x','hash',3600)
    l.uncertain('x','provider outcome unknown')
    with pytest.raises(BudgetStop,match='Uncertain'):l.reserve('y','hash2',3600)
