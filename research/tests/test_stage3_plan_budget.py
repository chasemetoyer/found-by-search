import copy
from decimal import Decimal
import pytest
from stage3.specs import specs,winners,configuration,decode_record
from stage3.budget import Ledger,BudgetStop

def test_roster_and_seeds():
 panel=specs('science');assert len(panel)==340 and len({s['id'] for s in panel})==340
 for s in panel:
  assert s['total_timesteps']==2_000_000
  assert s['config']['env']['eval_episodes']==64
  assert s['config']['ppo']['norm_adv']
  assert s['seed'] in (range(2020,2040) if s['method'].startswith('tamper_') else range(2000,2020))
 assert len(specs('smoke'))==34 and all(s['seed']>=5000 for s in specs('smoke'))

def test_frozen_rewards_consumed_literally():
 w=winners()
 for method,index in [('E',113),('R',57)]:
  assert w[method]['candidate']==index
  cfg=configuration(method,'DoorKey-8x8',2000,2_000_000)
  for feature in ['weights','relief','growth']:assert cfg['intrinsic'][feature]==w[method]['genome'][feature]
 assert configuration('S','KeyCorridorS3R2',2000,2_000_000)['intrinsic']['event_mode']=='locked_door_key'
 assert configuration('S','DoorKey-8x8',2000,2_000_000)['intrinsic']['kind']=='homeostatic'

def test_no_resubmit_and_hard_cap(tmp_path):
 ledger=Ledger(tmp_path/'budget.json');ledger.reserve('key','sha',3600);ledger.associate('key','fc-test','ap-test')
 with pytest.raises(BudgetStop):ledger.reserve('key','sha',3600)
 ledger.data['controller_sessions']['fixture']={'guard_charge_usd':'15'}
 with pytest.raises(BudgetStop):ledger.reserve('next','sha',3600)

def test_unknown_submission_blocks_launch(tmp_path):
 ledger=Ledger(tmp_path/'budget.json');ledger.reserve('key','sha',3600)
 with pytest.raises(BudgetStop):ledger.reserve('next','sha',3600)


def test_provider_billed_floor_blocks_even_when_model_is_lower(tmp_path):
    ledger=Ledger(tmp_path/'budget.json');ledger.data['provider_billing_floor_usd']='15'
    with pytest.raises(BudgetStop):ledger.reserve('new-input','sha',3600)


def test_reservations_accumulate_above_a_larger_billing_floor(tmp_path):
    ledger=Ledger(tmp_path/'budget.json');ledger.data['provider_billing_floor_usd']='14'
    count=0
    for i in range(32):
        try:ledger.reserve(str(i),'sha',3600)
        except BudgetStop:break
        ledger.associate(str(i),'fc-'+str(i),'app');count+=1
    assert count==7
    assert Decimal('14.9') < ledger.guard_total() <= Decimal('15')
