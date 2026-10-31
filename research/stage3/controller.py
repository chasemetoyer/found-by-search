"""Single Linux Stage3 controller; no local service or scientific launch implied."""
from __future__ import annotations
import json,os,platform,time,subprocess,sys,hashlib,resource
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
import modal
from stage3.common import *
from stage3.budget import Ledger,BudgetStop
from stage2.transport import retry_io
from stage3.specs import specs,decode_record,winners
VOLUME=modal.Volume.from_name(VOLUME_NAME,create_if_missing=True,version=2)
OUT=ROOT/'results'

def upload(path,name):
 def put():
  with VOLUME.batch_upload(force=True) as b:b.put_file(path,name)
 retry_io(put)

def save_json(name,data,remote=None):
 p=OUT/name;atomic_json(p,data);upload(p,remote or '/controller/'+name);return p

def read_remote(name):return json.loads(retry_io(lambda:b''.join(VOLUME.read_file(name))))

def persist(ledger):ledger.save();upload(ledger.path,'/budget-ledger.json')

def main():
 if platform.system()!='Linux':raise SystemExit('Stage3 controller must start on Modal Linux')
 if not os.environ.get('ORX_RUN_ID') or not os.environ.get('ORX_COMMIT'):raise SystemExit('Use tracked project runner')
 from stage3.registration import verify_sources
 verify_sources()
 OUT.mkdir(exist_ok=True);profile=json.loads((ROOT/'stage3/mode.json').read_text())['profile']
 if profile not in ('preflight','smoke','science'):raise ValueError('Unknown profile')
 winners()
 if profile=='science':
  from stage3.gates import require_science_gate
  require_science_gate()
 lock=modal.Dict.from_name(CONTROLLER_LOCK_NAME,create_if_missing=True);run=os.environ['ORX_RUN_ID']
 if not lock.put('owner',run,skip_if_exists=True):raise BudgetStop('AnotherStage3controllerownsledger')
 ledger=None;started=time.time();observer=None
 try:
  try:atomic_json(OUT/'budget-ledger.json',read_remote('/budget-ledger.json'))
  except (FileNotFoundError,modal.exception.NotFoundError):pass
  ledger=Ledger(OUT/'budget-ledger.json');ledger.session(run,43200 if profile=='science' else 1800);persist(ledger)
  def billing():
   from stage3.billing import query_billing
   evidence=query_billing(datetime.fromisoformat(ledger.data['created_utc']),previous_total_usd=ledger.data.get('provider_billing_floor_usd','0'))
   ledger.data['provider_billing_floor_usd']=evidence['retained_billed_floor_usd']
   save_json('billing-latest.json',evidence);persist(ledger)
  billing();last_billing=time.time()
  proof=decode_record();proof.update(platform=platform.platform(),machine=platform.machine(),run_id=run,source_commit=os.environ['ORX_COMMIT'])
  save_json('linux-decode.json',proof)
  mac=json.loads((ROOT/'stage3/mac-decode.json').read_text())
  if proof['configuration_sha256']!=mac['configuration_sha256'] or proof['winner_sha256']!=mac['winner_sha256']:raise ValueError('Mac/Linux frozen configuration hashes differ')
  print(json.dumps({'event':'decode_check','status':'PASS','configuration_sha256':proof['combined_sha256']}),flush=True)
  from stage3.modal_app import app,image,train_one
  from stage3.tracking_server import make_tracking_app
  tracking_app,tracking_fn,tracking_volume=make_tracking_app(image)
  tracking_app.deploy(name='evolved-intrinsic-rewards-stage3-monitor')
  # Keep the native dashboard warm inside an explicitly bounded paid lease.
  lease_id=run;expires=time.time()+43200
  lease_path=OUT/'tracking-service-lease.json';atomic_json(lease_path,{'lease_id':lease_id,'expires_unix':expires,'started_utc':utc_now()})
  with tracking_volume.batch_upload(force=True) as batch:batch.put_file(lease_path,'/service-lease.json')
  guardian_reservation=(Decimal('.125')*Decimal('.0000131')+Decimal('.125')*Decimal('.00000222'))*43320
  ledger.assert_can_launch(guardian_reservation)
  ledger.data.setdefault('tracking_guardian_calls',{})[lease_id]={'status':'reserved','guard_charge_usd':str(guardian_reservation),'expires_unix':expires}
  persist(ledger)
  from stage3.tracking_server import expiry_function
  guardian=expiry_function(tracking_app).spawn(lease_id,expires)
  ledger.data['tracking_guardian_calls'][lease_id].update(status='running',function_call_id=guardian.object_id);persist(ledger)
  save_json('tracking-lease.json',{'lease_id':lease_id,'expires_unix':expires,'guardian_call_id':guardian.object_id,'reservation_usd':str(guardian_reservation),'service_warm':True})
  base=tracking_fn.get_web_url()
  if not base:raise RuntimeError('Tracking endpoint has no deployed URL')
  private_url=base.rstrip('/')+'/private/'+os.environ['TRACKIO_ACCESS_PATH_TOKEN']
  os.environ['TRACKIO_SERVER_URL']=private_url
  # SecretURL never enters logs, checkpoints, Git or scientific results.
  from stage3.tracking import TrackioSink,snapshot_payload
  sink=TrackioSink();payload=snapshot_payload([],ledger.data,expected=2 if profile=='smoke' else 20)
  result=sink.publish(payload,0)
  if not result.get('ok'):raise RuntimeError('Remote tracking bootstrap did not initialize:'+result.get('error_type','unknown'))
  sink.finish()
  verified=sink.verify_remote()
  save_json('tracking-bootstrap-verification.json',verified)
  if not verified.get('ok'):raise RuntimeError('Tracking delivery not verified')
  if profile=='preflight':
   save_json('preflight-result.json',{'status':'PASS','decode':'PASS','tracking_initialized':True,'run_id':run,'scientific_submissions':0,'endpoint_base_public':base,'controller_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'controller_cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,'controller_wall_seconds':time.time()-started,'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text() if Path('/sys/fs/cgroup/cpu.max').exists() else None,'memory_max':Path('/sys/fs/cgroup/memory.max').read_text() if Path('/sys/fs/cgroup/memory.max').exists() else None})
   return
  # Observer is a different process; SDK/logging delays and failures cannot block training.
  env=dict(os.environ)
  observer=subprocess.Popen([sys.executable,'-m','stage3.observer','--expected','2' if profile=='smoke' else '20'],cwd=ROOT,env=env,stdout=(OUT/'tracking-observer.log').open('a'),stderr=subprocess.STDOUT)
  panel=specs(profile);save_json('plan.json',{'profile':profile,'specs':panel})
  active={};completed=set();submitted_count=0;replayed_count=0;blocked=None
  with modal.enable_output(),app.run():
   for spec in panel:
    key=spec['id'];row=ledger.data['calls'].get(key)
    if row:
     if row['spec_sha256']!=canonical_sha(spec):raise ValueError('Saved input changed')
     if row['status']=='complete':completed.add(key);replayed_count+=1
     elif row['status']=='running' and row['function_call_id']:active[key]=modal.FunctionCall.from_id(row['function_call_id'])
     else:raise BudgetStop('Existing unresolved input blocks resume')
   pending=[s for s in panel if s['id'] not in ledger.data['calls']]
   while pending or active:
    while pending and len(active)<32 and not blocked:
     spec=pending.pop(0);key=spec['id']
     try:ledger.reserve(key,canonical_sha(spec),900 if profile=='smoke' else 3600)
     except BudgetStop as error:
      pending.insert(0,spec);blocked=str(error);break
     persist(ledger)
     row=ledger.data['calls'][key]
     try:call=train_one.spawn(spec,row['absolute_deadline'],os.environ['ORX_COMMIT'],run)
     except BaseException:
      row['status']='uncertain';persist(ledger);raise
     submitted_count+=1;active[key]=call;ledger.associate(key,call.object_id,app.app_id);persist(ledger)
     print(json.dumps({'event':'submitted','id':key,'call_id':call.object_id}),flush=True)
    for key,call in list(active.items()):
     row=ledger.data['calls'][key]
     def poll():
      try:return call.get(timeout=0)
      except TimeoutError:return None
     result=retry_io(poll,deadline=row['absolute_deadline'])
     if result is None:continue
     ledger.complete(key,result);persist(ledger);active.pop(key);completed.add(key)
     print(json.dumps({'event':'completed','id':key,'outcome':result['outcome']}),flush=True)
     if result['outcome']!='success':raise BudgetStop('Numerical failure: stop before interpreting endpoints')
    if blocked and not active:raise BudgetStop(blocked)
    if time.time()-last_billing>1800:billing();last_billing=time.time()
    time.sleep(3)
  # Keep a remotely timed observation interval for the user's lid-closed witness.
  if profile=='smoke':
   save_json('smoke-training-complete.json',{'run_id':run,'completed':len(completed),'utc':utc_now()})
   remaining=max(0,(300 if submitted_count else 0)-(time.time()-started))
   if remaining:time.sleep(remaining)
  if observer:
   observer.terminate();observer.wait(timeout=30);observer=None
  # Separate telemetry flush AFTER training. Failures never rescore or retrain.
  try:
   with (OUT/'tracking-final-flush.log').open('a') as log:
    flush=subprocess.run([sys.executable,'-m','stage3.observer','--expected','2' if profile=='smoke' else '20','--once','--finish','--verify'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
   tracking_status='PASS' if flush.returncode==0 else 'FAIL'
  except Exception:tracking_status='FAIL'
  billing()
  complete_inputs=[]
  for key in sorted(completed):
   marker=read_remote('/runs/'+key+'/complete.json')
   complete_inputs.append({'id':key,'spec_sha256':marker['spec_sha256'],'marker':{'path':'stage3/proofs/smoke-markers/'+key+'.json','sha256':canonical_sha(marker)}})
  save_json('controller-validation.json',{'status':'PASS','profile':profile,'completed':len(completed),'expected':len(panel),'run_id':run,'source_commit':os.environ['ORX_COMMIT'],'worker_smoke_run_id':'f23220df-d501-4cde-8dbd-4c9d1bf8840e','platform':platform.platform(),'implementation_inventory_sha256':hashlib.sha256((ROOT/'preregistration/stage3-runtime-manifest.json').read_bytes()).hexdigest(),'submitted':submitted_count,'replayed':replayed_count,'completed_inputs':complete_inputs,'tracking_status':tracking_status})
  save_json('completion.json',{'status':'PASS','profile':profile,'run_id':run,'completed':len(completed),'expected':len(panel),'finished_utc':utc_now(),'guard_total_usd':str(ledger.guard_total()),'platform':platform.platform()})
 except BaseException as error:
  if ledger:
   for row in ledger.data['calls'].values():
    if row['status'] in ('running','reserved'):row.update(status='uncertain',error_type=type(error).__name__)
   persist(ledger)
   save_json('controller-stop.json',{'error_type':type(error).__name__,'run_id':run,'profile':profile,'automatic_restart_authorized':False})
  raise
 finally:
  if observer:
   observer.terminate();observer.wait(timeout=30)
  if ledger:
   ledger.finish_session(run);persist(ledger)
  if lock.get('owner')==run:lock.pop('owner')
if __name__=='__main__':main()
