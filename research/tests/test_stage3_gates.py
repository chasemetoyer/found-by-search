"""Pure receipt checks: never launch a worker or inspect a live provider ledger."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from stage3 import gates
from stage3.specs import ROOT, EXPECTED_HASHES, specs, decode_record
from stage3.tracking import LABEL, panel_html, snapshot_payload


def h(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return h(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
def dump(root,name,value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2)+'\n');return {'path':name,'sha256':h(p.read_bytes())}

def bind_approval(root,receipt):
    dump(root,gates.RECEIPT,receipt)
    approval=json.loads((root/gates.APPROVAL).read_text())
    approval['prelaunch_receipt_sha256']=h((root/gates.RECEIPT).read_bytes())
    dump(root,gates.APPROVAL,approval)


def replace_proof(root,receipt,name,value):
    receipt['proofs'][name]=dump(root,receipt['proofs'][name]['path'],value)
    bind_approval(root,receipt)


@pytest.fixture
def complete_receipts(tmp_path,monkeypatch):
    import stage2.registration
    # The inherited registration verifier has its own mandatory-file tests;
    # these pure fixtures isolate the new Stage3 proof gate only.
    monkeypatch.setattr(stage2.registration,'authorize',lambda *a,**k:None)
    monkeypatch.setattr(stage2.registration,'mode_record',lambda *a,**k:{})
    root=tmp_path
    for filename in gates.IMPLEMENTATION_FILES:
        p=root/filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('tested fixture '+filename)
    for method in EXPECTED_HASHES:
        name=f'stage3/frozen/frozen-winner-{method}.json';p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/name).read_bytes())
    amendment_paths=['preregistration/stage3-budget15-amendment-20261029.json',
                     'preregistration/stage3-descriptive-amendment-20261029.md']
    for name in amendment_paths:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('published approved amendment')
    hashes={name:h((root/name).read_bytes()) for name in gates.IMPLEMENTATION_FILES|set(amendment_paths)}
    inventory={'schema':'stage3-prelaunch-source-inventory','sha256':hashes}
    source_ref=dump(root,'preregistration/stage3-runtime-manifest.json',inventory)
    old_ref=dump(root,'stage3/evidence/original-worker-inventory.json',inventory)
    inputs=[]
    for spec in specs('smoke'):
        key=spec['id'];attempt='fixtureattempt'
        prefix=f'runs/{key}/attempts/{attempt}/'
        marker={'id':key,'spec_sha256':canonical(spec),'attempt':attempt,'outcome':'success',
                'source_commit':'original-worker-commit',
                'files':{prefix+n:'0'*64 for n in ('spec.json','metrics.jsonl','summary.json','environment.json',
                          'descriptive-episodes.jsonl','descriptive-windows.jsonl','descriptive-status.json')}}
        marker_ref=dump(root,f'stage3/evidence/markers/{key}.json',marker)
        inputs.append({'id':key,'spec_sha256':canonical(spec),'marker':marker_ref})
    worker={'status':'PASS','profile':'smoke','completed':34,'expected':34,'run_id':'worker-smoke-run',
            'source_commit':'original-worker-commit','platform':'Linux-6.8-x86_64',
            'finished_utc':'2026-10-29T14:38:30.039544+00:00',
            'implementation_inventory_sha256':old_ref['sha256']}
    controller={'status':'PASS','profile':'smoke','completed':34,'expected':34,'run_id':'controller-validation-run',
                'platform':'Linux-6.8-x86_64','worker_smoke_run_id':worker['run_id'],
                'implementation_inventory_sha256':source_ref['sha256'],'submitted':0,'replayed':34,
                'completed_inputs':inputs}
    record=decode_record()
    visual={'panel_names':[f'{LABEL}/{name}' for name in panel_html(snapshot_payload([],{}))],
            'project_id':gates.PROJECT_ID,'run_id':controller['run_id'],'completed_runs':34,
            'native_run_listed':True,'native_visualize_button_verified':True,'browser_verified':True,
            'remote_delivery_verified':True,'browser_verified_utc':'2026-10-29T14:55:00+00:00'}
    lid={'explicit_user_confirmation':True,'user_quote':'My lid was closed from10:35 to10:40 EDT.',
         'worker_smoke_run_id':worker['run_id'],'approval_seq':35,
         'closed_at':'2026-10-29T10:35:00-04:00','opened_at':'2026-10-29T10:40:00-04:00'}
    ledger={'stage':3,'limit_usd':'15','guard_total_usd':'1.00','calls':{'fixture-input':{'status':'complete'}}}
    cost={'hard_cap_usd':'15','projected_guard_peak_usd':'11.50',
          'final_evaluation_runs':300,'held_out_runs':100,'tamper_runs':40,'max_in_flight':32,
          'includes_retries':True,'includes_controller_container':True,'includes_tracking_container':True,
          'includes_prelaunch_spend':True,'includes_in_flight_reservations':True,
          'in_flight_reservation_usd':'2.05428480','scientific_input_plan_sha256':canonical(specs('science')),
          'provider_ledger_canonical_sha256':canonical(ledger)}
    values={'worker_inventory':old_ref,'source_inventory':source_ref}
    amendment={'status':'PASS','commit':'a'*40,'github_ref_commit':'a'*40,'repository_visibility':'PUBLIC',
               'implementation_inventory_sha256':source_ref['sha256'],
               'budget_amendment_sha256':h((root/amendment_paths[0]).read_bytes()),
               'descriptive_amendment_sha256':h((root/amendment_paths[1]).read_bytes()),
               'verified_utc':'2026-10-29T14:50:00+00:00'}
    values['amendment_commit']=dump(root,'stage3/evidence/amendment-commit.json',amendment)
    for name,value in [('worker_smoke',worker),('controller_validation',controller),
                       ('mac_decode',{'platform':'macOS-26-arm64','record':record}),
                       ('linux_decode',{'platform':'Linux-6.8-x86_64','record':record}),
                       ('visualize',visual),('lid_closed',lid),('cost_projection',cost)]:
        values[name]=dump(root,f'stage3/evidence/{name}.json',value)
    summary=root/'stage3/prelaunch-summary.md'
    summary.write_text('Six-item prelaunch summary: hashes, budget, decode, lid smoke, amendment commit, native Visualize.')
    values['summary']={'path':'stage3/prelaunch-summary.md','sha256':h(summary.read_bytes())}
    receipt={'schema':gates.RECEIPT_SCHEMA,'proofs':values,
             'summary_delivered_utc':'2026-10-29T15:00:00+00:00',
             'summary_items':['hash_verification','cost_and_cap','mac_linux_decode','lid_closed_smoke','amendment_commit','native_visualize']}
    approvallog=root/'stage3/evidence/user-approvals.jsonl';approvallog.parent.mkdir(parents=True,exist_ok=True)
    approvallog.write_text('\n'.join(json.dumps(x) for x in [
        {'seq':35,'quote':lid['user_quote'],'recorded_utc':'2026-10-29T14:41:00+00:00'},
        {'seq':36,'quote':'Go ahead and launch Stage3.', 'recorded_utc':'2026-10-29T15:01:00+00:00',
         'approval_scope':'stage3-scientific-launch','scientific_launch_authorized':True}])+'\n')
    dump(root,gates.APPROVAL,{'schema':gates.APPROVAL_SCHEMA,'prelaunch_receipt_sha256':'pending',
                             'approval_log':{'path':'stage3/evidence/user-approvals.jsonl','sha256':h(approvallog.read_bytes())},
                             'approval_seq':36,'scientific_launch_authorized':True})
    bind_approval(root,receipt)
    return root,receipt,ledger


def test_all_science_gates_can_pass_without_any_training(complete_receipts):
    root,_,ledger=complete_receipts
    result=gates.require_science_gate(root=root,current_ledger=ledger)
    assert result['status']=='PASS' and result['provider_mutations']==0
    assert result['smoke_inputs_verified']==34 and result['native_panels_verified']==18


@pytest.mark.parametrize('changed_file,allowed',[
    ('stage3/tracking_server.py', True), ('stage3/specs.py', False),
    ('stage3/budget.py', False), ('stage3/controller.py', False),
])
def test_later_repair_preserves_prior_approval_only_for_infrastructure(complete_receipts,changed_file,allowed):
    root,receipt,ledger=complete_receipts
    old_inventory=json.loads((root/receipt['proofs']['source_inventory']['path']).read_text())
    prior_inventory=dump(root,'stage3/evidence/pre-summary-inventory.json',old_inventory)
    amendment=json.loads((root/receipt['proofs']['amendment_commit']['path']).read_text())
    prior_amendment=dump(root,'stage3/evidence/pre-summary-amendment.json',amendment)
    (root/changed_file).write_text('later infrastructure or forbidden scientific change')
    inventory=copy.deepcopy(old_inventory)
    inventory['sha256'][changed_file]=h((root/changed_file).read_bytes())
    receipt['proofs']['source_inventory']=dump(root,'preregistration/stage3-runtime-manifest.json',inventory)
    amendment.update(verified_utc='2026-10-29T15:02:00+00:00',scope='post-summary-infrastructure-only',
                     prior_amendment_commit=prior_amendment,prior_source_inventory=prior_inventory,
                     implementation_inventory_sha256=receipt['proofs']['source_inventory']['sha256'])
    receipt['proofs']['amendment_commit']=dump(root,receipt['proofs']['amendment_commit']['path'],amendment)
    validation=json.loads((root/receipt['proofs']['controller_validation']['path']).read_text())
    validation['implementation_inventory_sha256']=receipt['proofs']['source_inventory']['sha256']
    replace_proof(root,receipt,'controller_validation',validation)
    if allowed:
        assert gates.require_science_gate(root=root,current_ledger=ledger)['status']=='PASS'
    else:
        with pytest.raises(gates.LaunchGateError,match='outside the infrastructure allowlist'):
            gates.require_science_gate(root=root,current_ledger=ledger)


def test_later_repair_requires_real_pre_summary_provenance(complete_receipts):
    root,receipt,ledger=complete_receipts
    amendment=json.loads((root/receipt['proofs']['amendment_commit']['path']).read_text())
    amendment.update(verified_utc='2026-10-29T15:02:00+00:00',scope='post-summary-infrastructure-only')
    replace_proof(root,receipt,'amendment_commit',amendment)
    with pytest.raises(gates.LaunchGateError,match='Invalid proof reference'):
        gates.require_science_gate(root=root,current_ledger=ledger)


def test_boolean_only_launch_approval_cannot_bypass_proofs(complete_receipts):
    root,_,ledger=complete_receipts
    dump(root,gates.APPROVAL,{'explicit_user_goahead':True,'scientific_launch_authorized':True,
                             'smoke_pass':True,'projected_guard_peak_usd':0})
    with pytest.raises(gates.LaunchGateError):gates.require_science_gate(root=root,current_ledger=ledger)


@pytest.mark.parametrize('field',['browser_verified','native_visualize_button_verified','remote_delivery_verified'])
def test_pending_browser_or_tracking_proof_blocks(complete_receipts,field):
    root,receipt,ledger=complete_receipts
    visual=json.loads((root/receipt['proofs']['visualize']['path']).read_text());visual[field]=False
    replace_proof(root,receipt,'visualize',visual)
    with pytest.raises(gates.LaunchGateError,match='Visualize/browser'):
        gates.require_science_gate(root=root,current_ledger=ledger)


def test_missing_panel_blocks(complete_receipts):
    root,receipt,ledger=complete_receipts
    visual=json.loads((root/receipt['proofs']['visualize']['path']).read_text());visual['panel_names'].pop()
    replace_proof(root,receipt,'visualize',visual)
    with pytest.raises(gates.LaunchGateError,match='omits'):
        gates.require_science_gate(root=root,current_ledger=ledger)


@pytest.mark.parametrize('closed,opened',[
    ('2026-10-29T14:39:00+00:00','2026-10-29T14:40:00+00:00'),
    ('2026-10-29T14:35:00','2026-10-29T14:40:00+00:00'),
])
def test_lid_witness_must_have_timezone_and_enclose_original_worker_end(complete_receipts,closed,opened):
    root,receipt,ledger=complete_receipts
    lid=json.loads((root/receipt['proofs']['lid_closed']['path']).read_text());lid.update(closed_at=closed,opened_at=opened)
    replace_proof(root,receipt,'lid_closed',lid)
    with pytest.raises(gates.LaunchGateError):gates.require_science_gate(root=root,current_ledger=ledger)


def test_one_ulp_decode_difference_blocks(complete_receipts):
    root,receipt,ledger=complete_receipts
    linux=json.loads((root/receipt['proofs']['linux_decode']['path']).read_text())
    first=next(iter(linux['record']['configuration_sha256']));linux['record']['configuration_sha256'][first]='f'*64
    replace_proof(root,receipt,'linux_decode',linux)
    with pytest.raises(gates.LaunchGateError,match='configurations differ'):
        gates.require_science_gate(root=root,current_ledger=ledger)


def test_completed_input_hash_or_duplicate_marker_cannot_pass(complete_receipts):
    root,receipt,ledger=complete_receipts
    c=json.loads((root/receipt['proofs']['controller_validation']['path']).read_text());c['completed_inputs'][0]['spec_sha256']='0'*64
    replace_proof(root,receipt,'controller_validation',c)
    with pytest.raises(gates.LaunchGateError,match='identity/hash differs'):
        gates.require_science_gate(root=root,current_ledger=ledger)


def test_worker_change_requires_new_real_training_smoke(complete_receipts):
    root,receipt,ledger=complete_receipts
    current=json.loads((root/'preregistration/stage3-runtime-manifest.json').read_text())
    path='stage3/run_training.py';(root/path).write_text('changed worker')
    current['sha256'][path]=h((root/path).read_bytes())
    receipt['proofs']['source_inventory']=dump(root,'preregistration/stage3-runtime-manifest.json',current)
    c=json.loads((root/receipt['proofs']['controller_validation']['path']).read_text())
    c['implementation_inventory_sha256']=receipt['proofs']['source_inventory']['sha256']
    amendment=json.loads((root/receipt['proofs']['amendment_commit']['path']).read_text())
    amendment['implementation_inventory_sha256']=receipt['proofs']['source_inventory']['sha256']
    receipt['proofs']['amendment_commit']=dump(root,receipt['proofs']['amendment_commit']['path'],amendment)
    replace_proof(root,receipt,'controller_validation',c)
    with pytest.raises(gates.LaunchGateError,match='Worker implementation changed'):
        gates.require_science_gate(root=root,current_ledger=ledger)


@pytest.mark.parametrize('cap,peak,heldout',[('14','11',100),('15','13.00001',100),('15','11',0)])
def test_budget_cap_review_threshold_and_heldout_matrix_are_mandatory(complete_receipts,cap,peak,heldout):
    root,receipt,ledger=complete_receipts
    c=json.loads((root/receipt['proofs']['cost_projection']['path']).read_text())
    c.update(hard_cap_usd=cap,projected_guard_peak_usd=peak,held_out_runs=heldout)
    replace_proof(root,receipt,'cost_projection',c)
    with pytest.raises(gates.LaunchGateError):gates.require_science_gate(root=root,current_ledger=ledger)


def test_ledger_change_or_unresolved_charge_blocks(complete_receipts):
    root,receipt,ledger=complete_receipts
    ledger=copy.deepcopy(ledger);ledger['guard_total_usd']='1.01'
    with pytest.raises(gates.LaunchGateError,match='ledger changed'):
        gates.require_science_gate(root=root,current_ledger=ledger)


def test_goahead_before_summary_is_not_reused(complete_receipts):
    root,receipt,ledger=complete_receipts
    approval=json.loads((root/gates.APPROVAL).read_text());path=root/approval['approval_log']['path']
    records=[json.loads(line) for line in path.read_text().splitlines()]
    records[-1]['recorded_utc']='2026-10-29T14:59:00+00:00';path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    approval['approval_log']['sha256']=h(path.read_bytes());dump(root,gates.APPROVAL,approval)
    with pytest.raises(gates.LaunchGateError,match='predates'):
        gates.require_science_gate(root=root,current_ledger=ledger)


def test_explicit_lid_only_waiver_stays_unverified_and_needs_later_launch_approval(complete_receipts):
    root,receipt,ledger=complete_receipts
    approval=json.loads((root/gates.APPROVAL).read_text());logpath=root/approval['approval_log']['path']
    records=[json.loads(line) for line in logpath.read_text().splitlines()]
    waiver_quote="it wasn’t, but it doesnt matter just go ahead and proceed it proboaly worked"
    records[0].update(quote=waiver_quote,approval_scope='lid-closed-smoke-waiver')
    logpath.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    approval['approval_log']['sha256']=h(logpath.read_bytes());dump(root,gates.APPROVAL,approval)
    lid={'status':'WAIVED_UNVERIFIED','waiver_scope':'lid_closed_smoke_witness_only',
         'actual_lid_closed_verified':False,'explicit_user_confirmation':False,
         'approval_seq':35,'user_quote':waiver_quote,'worker_smoke_run_id':'worker-smoke-run'}
    replace_proof(root,receipt,'lid_closed',lid)
    result=gates.require_science_gate(root=root,current_ledger=ledger)
    assert result['status']=='PASS' and result['lid_closed_status']=='WAIVED_UNVERIFIED'
    # The waiver row itself cannot substitute for the later full launch approval.
    approval=json.loads((root/gates.APPROVAL).read_text());approval['approval_seq']=35;dump(root,gates.APPROVAL,approval)
    with pytest.raises(gates.LaunchGateError,match='full launch go-ahead'):
        gates.require_science_gate(root=root,current_ledger=ledger)


def test_lid_waiver_cannot_claim_that_actual_lid_closure_passed(complete_receipts):
    root,receipt,ledger=complete_receipts
    lid=json.loads((root/receipt['proofs']['lid_closed']['path']).read_text())
    lid.update(status='WAIVED_UNVERIFIED',waiver_scope='lid_closed_smoke_witness_only',
               actual_lid_closed_verified=True,explicit_user_confirmation=False)
    replace_proof(root,receipt,'lid_closed',lid)
    with pytest.raises(gates.LaunchGateError,match='explicitly unverified'):
        gates.require_science_gate(root=root,current_ledger=ledger)


def test_projected_peak_cannot_hide_missing_inflight_reservations(complete_receipts):
    root,receipt,ledger=complete_receipts
    cost=json.loads((root/receipt['proofs']['cost_projection']['path']).read_text())
    cost['in_flight_reservation_usd']='0'
    replace_proof(root,receipt,'cost_projection',cost)
    with pytest.raises(gates.LaunchGateError,match='in-flight reservations'):
        gates.require_science_gate(root=root,current_ledger=ledger)


def post_summary_infra_proof(root,receipt,*,changed='stage3/tracking_server.py'):
    """Preserve genuine old public receipts, then bind a later narrow repair."""
    old_inventory=json.loads((root/'preregistration/stage3-runtime-manifest.json').read_text())
    old_amendment=json.loads((root/receipt['proofs']['amendment_commit']['path']).read_text())
    prior_inventory=dump(root,'stage3/evidence/prior-public-inventory.json',old_inventory)
    prior_amendment=dump(root,'stage3/evidence/prior-public-amendment.json',old_amendment)
    (root/changed).write_text('new post-summary infrastructure fixture')
    new_inventory=copy.deepcopy(old_inventory);new_inventory['sha256'][changed]=h((root/changed).read_bytes())
    receipt['proofs']['source_inventory']=dump(root,'preregistration/stage3-runtime-manifest.json',new_inventory)
    new_amendment=dict(old_amendment,commit='b'*40,github_ref_commit='b'*40,
                       implementation_inventory_sha256=receipt['proofs']['source_inventory']['sha256'],
                       verified_utc='2026-10-29T15:02:00+00:00',scope='post-summary-infrastructure-only',
                       prior_source_inventory=prior_inventory,prior_amendment_commit=prior_amendment)
    receipt['proofs']['amendment_commit']=dump(root,'stage3/evidence/amendment-commit.json',new_amendment)
    controller=json.loads((root/receipt['proofs']['controller_validation']['path']).read_text())
    controller['implementation_inventory_sha256']=receipt['proofs']['source_inventory']['sha256']
    replace_proof(root,receipt,'controller_validation',controller)


def test_post_summary_narrow_infrastructure_repair_keeps_prior_scientific_approval(complete_receipts):
    root,receipt,ledger=complete_receipts
    post_summary_infra_proof(root,receipt)
    result=gates.require_science_gate(root=root,current_ledger=ledger)
    assert result['status']=='PASS'
    assert result['post_summary_infrastructure_changes']==['stage3/tracking_server.py']
    assert result['approval_seq']==36  # The explicit original post-summary go-ahead.


def test_post_summary_repair_without_preserved_public_provenance_blocks(complete_receipts):
    root,receipt,ledger=complete_receipts
    amendment=json.loads((root/receipt['proofs']['amendment_commit']['path']).read_text())
    amendment.update(verified_utc='2026-10-29T15:02:00+00:00',scope='post-summary-infrastructure-only')
    replace_proof(root,receipt,'amendment_commit',amendment)
    with pytest.raises(gates.LaunchGateError,match='Invalid proof reference'):
        gates.require_science_gate(root=root,current_ledger=ledger)


@pytest.mark.parametrize('path',['stage3/specs.py','stage3/controller.py','stage3/budget.py','stage3/run_training.py'])
def test_post_summary_repair_cannot_change_scientific_or_dispatch_code(complete_receipts,path):
    root,receipt,ledger=complete_receipts
    post_summary_infra_proof(root,receipt,changed=path)
    with pytest.raises(gates.LaunchGateError,match='scientific, worker or dispatch'):
        gates.require_science_gate(root=root,current_ledger=ledger)


def test_post_summary_repair_cannot_rewrite_budget_or_descriptive_amendment(complete_receipts):
    root,receipt,ledger=complete_receipts
    post_summary_infra_proof(root,receipt,changed='preregistration/stage3-descriptive-amendment-20261029.md')
    amendment=json.loads((root/receipt['proofs']['amendment_commit']['path']).read_text())
    amendment['descriptive_amendment_sha256']=h((root/'preregistration/stage3-descriptive-amendment-20261029.md').read_bytes())
    replace_proof(root,receipt,'amendment_commit',amendment)
    with pytest.raises(gates.LaunchGateError,match='approved budget/descriptive'):
        gates.require_science_gate(root=root,current_ledger=ledger)
