"""Independent hard15USD Stage3 guard. No changes to frozen Stage2 ledger."""
from decimal import Decimal
import json,time
from pathlib import Path
from stage1.budget import estimate_cost,BudgetStop,atomic_json,utc_now,TERMINATION_ALLOWANCE_SECONDS
CAP=Decimal('15');OVERHEAD=Decimal('.50')
# Conservative controller envelope, not a claim that native Sandbox maxima exist.
CONTROLLER_SECOND=Decimal('.00003942')+2*Decimal('.00000667')
TRACKER_SERVER_12H=(Decimal('.25')*Decimal('.0000131')+Decimal('.5')*Decimal('.00000222'))*43200
class Ledger:
 def __init__(self,path):
  self.path=Path(path)
  self.data=json.loads(self.path.read_text()) if self.path.exists() else {'stage':3,'limit_usd':'15','calls':{},'controller_sessions':{},'overhead_usd':str(OVERHEAD),'tracking_server_reservation_usd':str(TRACKER_SERVER_12H),'created_utc':utc_now()}
  if self.data.get('stage')!=3 or self.data.get('limit_usd')!='15':raise BudgetStop('Stage3 hard cap must be15USD')
 def guard_total(self):
  estimate=sum((Decimal(r['guard_charge_usd']) for r in self.data['calls'].values()),Decimal(0))+sum((Decimal(r['guard_charge_usd']) for r in self.data['controller_sessions'].values()),Decimal(0))+Decimal(self.data['overhead_usd'])+Decimal(self.data['tracking_server_reservation_usd'])+sum((Decimal(r['guard_charge_usd']) for r in self.data.get('tracking_guardian_calls',{}).values()),Decimal(0))
  overhead=Decimal(self.data['overhead_usd'])
  base=estimate-overhead
  # Retain any observed billing/model discrepancy. Future reservations must
  # add to this correction rather than disappear beneath a max() floor.
  correction=max(Decimal(self.data.get('billing_adjustment_usd','0')),
                 Decimal(self.data.get('provider_billing_floor_usd','0'))-base)
  self.data['billing_adjustment_usd']=str(correction)
  return estimate+correction
 def save(self):
  self.data.update(updated_utc=utc_now(),guard_total_usd=str(self.guard_total()));atomic_json(self.path,self.data)
 def assert_can_launch(self,amount=Decimal(0)):
  if any(c['status'] in ('reserved','uncertain','numerical_failure') for c in self.data['calls'].values()):raise BudgetStop('Unresolved submission or failure; stop for reconciliation')
  if self.guard_total()+amount>CAP:raise BudgetStop('Stage3 hard15USD guard blocks next reservation')
 def session(self,run_id,seconds):
  if run_id in self.data['controller_sessions']:raise BudgetStop('Controller session already recorded')
  amount=CONTROLLER_SECOND*seconds;self.assert_can_launch(amount)
  self.data['controller_sessions'][run_id]={'started_unix':time.time(),'reserved_seconds':seconds,'guard_charge_usd':str(amount),'cost_basis':'Sandbox forecast proxy1core2GiB atnon-preemptiblerates; actual billedusage mayvary'};self.save()
 def finish_session(self,run_id):
  row=self.data['controller_sessions'][run_id];elapsed=time.time()-row['started_unix']+60
  row.update(guard_charge_usd=str(CONTROLLER_SECOND*Decimal(str(min(elapsed,row['reserved_seconds'])))),elapsed_seconds=elapsed);self.save()
 def reserve(self,key,sha,deadline):
  if key in self.data['calls']:raise BudgetStop('Recorded input cannot be submitted again:'+key)
  if sum(c['status']=='running' for c in self.data['calls'].values())>=32:raise BudgetStop('All32slotsreserved')
  amount=estimate_cost(deadline+TERMINATION_ALLOWANCE_SECONDS);self.assert_can_launch(amount)
  self.data['calls'][key]={'status':'reserved','spec_sha256':sha,'guard_charge_usd':str(amount),'reservation_usd':str(amount),'submitted_unix':time.time(),'absolute_deadline':time.time()+deadline,'function_call_id':None};self.save()
 def associate(self,key,call,app):
  self.data['calls'][key].update(status='running',function_call_id=call,app_id=app);self.save()
 def complete(self,key,marker):
  row=self.data['calls'][key]
  if marker['id']!=key or marker['spec_sha256']!=row['spec_sha256']:raise ValueError('Result identity mismatch')
  elapsed=time.time()-row['submitted_unix']+TERMINATION_ALLOWANCE_SECONDS
  charged=estimate_cost(elapsed)
  if charged>Decimal(row['reservation_usd']):row['status']='uncertain';self.save();raise BudgetStop('Call exceeded reserved envelope')
  row.update(status='complete' if marker['outcome']=='success' else 'numerical_failure',guard_charge_usd=str(charged),measured=marker['usage'],attempt_count=marker['usage'].get('attempt_count',1),completed_utc=utc_now());self.save()
