"""Independent volume reader and tracking publisher. Never imports training code.

The only volume write is an observer-owned telemetry receipt under /controller/.
It never edits the budget ledger, completion markers, specifications or results.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import modal
from stage3.common import ROOT,VOLUME_NAME,canonical_sha,cpu_class,atomic_json,utc_now
from stage3.tracking import TrackioSink,snapshot_payload,quiet_tracking_output
from stage2.transport import retry_io


def collect(volume,profile,cache=None,disk_cache=None):
    cache={} if cache is None else cache
    disk_cache=Path(disk_cache) if disk_cache is not None else None
    if disk_cache is not None:disk_cache.mkdir(parents=True,exist_ok=True)
    def raw(path):return retry_io(lambda:b''.join(volume.read_file(path)))
    ledger=json.loads(raw('/budget-ledger.json'));records=[];errors=[]
    def load_record(key,row):
        if key in cache:return cache[key]
        path=(disk_cache/(hashlib.sha256(key.encode()).hexdigest()+'.json')) if disk_cache else None
        if path and path.exists():
            try:
                saved=json.loads(path.read_text());record=saved['record']
                if saved['spec_sha256']==row['spec_sha256'] and saved['record_sha256']==canonical_sha(record):
                    return record
            except (ValueError,KeyError,OSError):pass  # Re-read authoritative results.
        marker=json.loads(raw('/runs/'+key+'/complete.json'))
        prefix=f'/runs/{key}/attempts/{marker["attempt"]}/'
        def checked(name):
            data=raw(prefix+name);expected=marker['files'].get((prefix+name).lstrip('/'))
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Result hash mismatch')
            return data
        spec=json.loads(checked('spec.json'))
        if canonical_sha(spec)!=row['spec_sha256'] or marker['id']!=key:
            raise ValueError('Recorded input mismatch')
        metrics=[json.loads(line) for line in checked('metrics.jsonl').decode().splitlines() if line.strip()]
        windows={w['global_step']:w for w in [json.loads(line) for line in checked('descriptive-windows.jsonl').decode().splitlines() if line.strip()]}
        for metric in metrics:
            window=windows.get(metric['global_step'],{})
            for source,target in [('presses_per_step','tamper/presses_per_step'),
                                  ('presses_per_episode','tamper/presses_per_episode'),
                                  ('episode_intrinsic_return_mean','intrinsic/episode_return')]:
                if window.get(source) is not None:metric[target]=window[source]
        env=json.loads(checked('environment.json'))
        success=[m.get('train/success',0.) for m in metrics]
        import numpy as np
        area=float(np.mean(success));solved=float(np.mean(success[-max(1,len(success)//10):]))>=.9
        record={'method':spec['method'],'task':spec['task'],'seed':spec['seed'],
                'metrics':metrics,'area':area,'solved':solved,'cpu_class':cpu_class(env['cpu']),
                '_tracking_key':key,'_tracking_attempt':marker['attempt']}
        if path:atomic_json(path,{'spec_sha256':row['spec_sha256'],'record_sha256':canonical_sha(record),'record':record})
        return record
    completed={key:row for key,row in ledger['calls'].items()
               if key.startswith('stage3_'+profile+'-') and row['status']=='complete'}
    # Fixed read concurrency insulates tracker latency from training throughput.
    with ThreadPoolExecutor(max_workers=4) as readers:
        jobs={readers.submit(load_record,key,row):key for key,row in completed.items()}
        for future in as_completed(jobs):
            key=jobs[future]
            try:
                record=future.result();records.append(record);cache[key]=record
                if record.get('_tracking_attempt'):
                    directory=f"runs/{key}/attempts/{record['_tracking_attempt']}"
                    cache['cpu',directory]=record['cpu_class']
            except Exception as error:errors.append({'id':key,'error_type':type(error).__name__})
    records.sort(key=lambda r:(r['method'],r['task'],r['seed']))
    physical={}
    for entry in volume.listdir('/runs',recursive=True):
        parts=Path(entry.path.lstrip('/')).parts
        if parts and parts[0]!='runs':parts=('runs',)+parts
        if len(parts)==5 and parts[1].startswith('stage3_'+profile+'-') and parts[2]=='attempts':
            physical.setdefault('/'.join(parts[:4]),set()).add(parts[4])
    mix=Counter();per_key=Counter()
    for directory,names in physical.items():
        per_key[directory.split('/')[1]]+=1
        cachekey=('cpu',directory)
        try:
            if cachekey not in cache and 'environment.json' in names:
                cache[cachekey]=cpu_class(json.loads(raw('/'+directory+'/environment.json'))['cpu'])
            mix[cache.get(cachekey,'unknown')]+=1
        except Exception as error:
            mix['unknown']+=1
            errors.append({'id':directory,'error_type':type(error).__name__,'kind':'cpu_metadata'})
    # Count physical observations, including additional attempts of active calls.
    # Ledger history alone misses SDK/container restarts and partial attempts.
    ledger['physical_cpu_mix']=dict(mix)
    ledger['physical_attempts_observed']=len(physical)
    ledger['candidate_failures']=sum(r['status']=='numerical_failure' for r in ledger['calls'].values())
    ledger['infrastructure_failures']=sum('infrastructure_failure.json' in names for names in physical.values())
    ledger['observed_restarts']=sum(max(0,n-1) for n in per_key.values())
    return records,ledger,errors


def observe_once(volume,sink,profile,expected,*,finish=False,verify=False,step=None,cache=None,disk_cache=None):
    """Publish a snapshot; final verification checks every cell's actual server value."""
    records,ledger,errors=collect(volume,profile,cache,disk_cache)
    payload=snapshot_payload(records,ledger,expected)
    step=int(time.time()*1000) if step is None else step
    published=sink.publish(payload,step)
    finished=sink.finish() if finish else None
    verified=sink.verify_remote(payload,step) if verify else None
    complete=all(value==expected for value in payload['completed_by_cell'].values())
    good=published.get('ok') and (not finish or finished.get('ok')) and (not verify or verified.get('ok'))
    final_requested=finish and verify
    status=('PASS' if complete and good and not errors else 'FAIL') if final_requested else ('QUEUED' if good else 'FAIL')
    return {'schema':'stage3-tracking-observer-receipt','updated_utc':utc_now(),'status':status,
            'run_id':os.environ.get('ORX_RUN_ID'),'source_commit':os.environ.get('ORX_COMMIT'),
            'profile':profile,'completed_records':len(records),'expected_per_cell':expected,
            'expected_records':expected*len(payload['completed_by_cell']),
            'completed_by_cell':payload['completed_by_cell'],'complete_snapshot':complete,
            'read_errors':errors,'publish':published,'finish':finished,'remote_verification':verified,
            'physical_attempts_observed':ledger['physical_attempts_observed'],
            'physical_cpu_mix':ledger['physical_cpu_mix'],'infrastructure_failures':ledger['infrastructure_failures'],
            'observed_restarts':ledger['observed_restarts'],'tracking_step':step,
            'scientific_results_mutated':False,'budget_ledger_mutated':False,
            'training_dependency':False}


def save_flush_receipt(volume,receipt,directory=None):
    directory=ROOT/'results' if directory is None else Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'tracking-final-flush.json';atomic_json(path,receipt)
    def upload():
        with volume.batch_upload(force=True) as batch:
            batch.put_file(path,'/controller/tracking-final-flush.json')
    retry_io(upload)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expected',type=int,default=20)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--finish',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.expected<=0:parser.error('--expected must be positive')
    if (args.finish or args.verify) and not args.once:parser.error('--finish/--verify require --once')
    profile=json.loads((ROOT/'stage3/mode.json').read_text())['profile']
    volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=False,version=2)
    sink=TrackioSink();iteration=0;cache={}
    while True:
        try:
            with quiet_tracking_output():
                receipt=observe_once(volume,sink,profile,args.expected,finish=args.finish,
                                     verify=args.verify,cache=cache,disk_cache=ROOT/'results'/'tracking-cache')
        except Exception as error:
            receipt={'status':'FAIL','run_id':os.environ.get('ORX_RUN_ID'),
                     'error_type':type(error).__name__,'scientific_results_mutated':False,
                     'budget_ledger_mutated':False,'training_dependency':False}
        if args.once:
            # The receipt is telemetry only. Its failure cannot rescore or resubmit PPO.
            save_flush_receipt(volume,receipt)
            return 0 if receipt['status'] in ('PASS','QUEUED') else 1
        print(json.dumps({'event':'tracking_observer','iteration':iteration,
                          'completed':receipt.get('completed_records'),
                          'read_errors':receipt.get('read_errors',[]),
                          'tracking':receipt.get('publish'),
                          'error_type':receipt.get('error_type')}),flush=True)
        iteration+=1;time.sleep(60)


if __name__=='__main__':
    with quiet_tracking_output():raise SystemExit(main())
