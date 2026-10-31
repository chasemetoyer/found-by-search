"""Synthetic provider rows only: no live queries or compute launches."""
from dataclasses import dataclass
from datetime import datetime,timezone
from decimal import Decimal
from types import SimpleNamespace
import json
import pytest
from stage3.billing import PROJECT_ID,attribution_report,query_billing,effective_guard,tracking_lease_reservation


def row(cost, *, project=PROJECT_ID, stage='3',role='training'):
    return {'cost':cost,'tags':{'project_id':project,'stage':stage,'role':role},'object_id':'ap-fixture'}


def test_only_current_project_and_stage_including_tracking_are_charged():
    data=attribution_report([row('2'),row('.2',role='tracking'),row('999',stage='2'),row('888',project='other'),row('777',stage=None)])
    assert data['provider_reported_complete_intervals_cost_usd']=='2.2'
    assert data['cost_by_role_usd']=={'tracking':'.2','training':'2'} or data['cost_by_role_usd']=={'tracking':'0.2','training':'2'}
    assert len(data['rows'])==2
    assert data['excluded_row_counts']=={'different_project':1,'different_or_missing_stage':2}
    assert '999' not in json.dumps(data) and '888' not in json.dumps(data)


def test_dataclass_rows_and_exact_decimal_sum():
    @dataclass
    class BillingRow:
        cost:Decimal
        tags:dict
        interval_start:datetime
    data=attribution_report([BillingRow(Decimal('.1'),row(0)['tags'],datetime(2026,1,1,tzinfo=timezone.utc)),row('.2')])
    assert data['provider_reported_complete_intervals_cost_usd']=='0.3'
    json.dumps(data)


def test_floor_does_not_fall_or_double_count_controller_proxy():
    data=attribution_report([row('3')],previous_total_usd='4',native_controller_proxy_usd='2.279232')
    assert data['retained_billed_floor_usd']=='4'
    assert data['native_controller_invoice_usd'] is None
    assert effective_guard('10',data)==10
    high=attribution_report([row('15.1')])
    assert effective_guard('10',high)==Decimal('15.60')


def test_exact_api_parameters_and_failures_retain_old_floor_without_leaking_messages():
    calls=[]
    def report(**kwargs):calls.append(kwargs);return [row('.4',role='tracking')]
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    workspace=SimpleNamespace(billing=SimpleNamespace(report=report))
    data=query_billing(start,workspace=workspace)
    assert data['status']=='ok' and calls==[{'start':start,'resolution':'h','tag_names':['project_id','stage','role']}]
    def fail(**kwargs):raise RuntimeError('token=DO_NOT_SAVE')
    bad=query_billing(start,workspace=SimpleNamespace(billing=SimpleNamespace(report=fail)),previous_total_usd='1.7')
    assert bad['status']=='error' and bad['retained_billed_floor_usd']=='1.7'
    assert 'DO_NOT_SAVE' not in json.dumps(bad)


@pytest.mark.parametrize('value',['-1','NaN','Infinity',True,None])
def test_invalid_selected_cost_retains_floor(value):
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    data=query_billing(start,workspace=SimpleNamespace(billing=SimpleNamespace(report=lambda **kw:[row(value)])),previous_total_usd='2')
    assert data['status']=='error' and data['retained_billed_floor_usd']=='2'


def test_unknown_or_other_stage_cost_is_not_silently_accepted_as_stage3():
    data=attribution_report([row('3',stage=None),row('4',stage='2')],native_controller_proxy_usd='1')
    assert data['provider_reported_complete_intervals_cost_usd']=='0'
    assert data['native_controller_invoice_usd'] is None
    assert data['native_controller_proxy_usd']=='1'


def test_tracking_lease_forecast_is_finite_and_invalid_durations_fail():
    assert tracking_lease_reservation()==Decimal('.189432000')
    for invalid in (0,-1,True,1.5):
        with pytest.raises(ValueError):tracking_lease_reservation(invalid)
    with pytest.raises(ValueError):query_billing(datetime(2026,1,1))
