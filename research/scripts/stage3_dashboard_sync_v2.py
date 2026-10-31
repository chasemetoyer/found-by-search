"""Unregistered, independent presentation repair. Never launches or alters PPO.

Reads frozen result/ledger sources and writes only local observer files plus
Trackio presentation metadata. Stop the old observer before using this writer.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,html,json,os,re,sys,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage3.observer import collect
from stage3.tracking import (LABEL,METHODS,TASKS,NAMES,COLORS,TrackioSink,
                             snapshot_payload,panel_html,quiet_tracking_output)
from stage3.common import ROOT,VOLUME_NAME,canonical_sha,atomic_json

PROGRESS='progress/completed by method and task'
KEY=re.compile(r'^stage3_(smoke|science)-(E|R|T|N|S|tamper_on|tamper_off)-(DoorKey-8x8|UnlockPickup|KeyCorridorS3R2)-[0-9]+-seed([0-9]+)$')


def legacy_metrics(metrics):
    prefix=LABEL+'/progress/completed/'
    exact={LABEL+'/progress/completed_total',LABEL+'/budget/guard_usd',LABEL+'/budget/cap_usd'}
    return sorted(m for m in metrics if m.startswith(prefix) or m in exact)


def completion_history(ledger,records,profile):
    """Genuine completion order; equal timestamps are simultaneous groups."""
    valid={(r['method'],r['task'],r['seed']) for r in records}
    grouped=defaultdict(list)
    for key,row in ledger['calls'].items():
        match=KEY.fullmatch(key)
        if not match or match[1]!=profile or row['status']!='complete':continue
        identity=(match[2],match[3],int(match[4]))
        if identity not in valid:raise ValueError('Completed result missing from validated observations')
        stamp=datetime.fromisoformat(row['completed_utc'].replace('Z','+00:00'))
        grouped[stamp].append(match[2]+'/'+match[3])
    cells=[m+'/'+task for m in METHODS for task in TASKS]+[m+'/DoorKey-8x8' for m in ('tamper_on','tamper_off')]
    current={cell:0 for cell in cells};series={cell:[(0,0)] for cell in cells};total=0
    for stamp in sorted(grouped):
        for cell in grouped[stamp]:current[cell]+=1
        total+=len(grouped[stamp])
        for cell in cells:series[cell].append((total,current[cell]))
    if total!=len(records):raise ValueError('Validated result count differs from completion history')
    return series,current,total


def step_path(points,X,Y):
    if not points:return ''
    path=f'M{X(points[0][0]):.3f},{Y(points[0][1]):.3f}'
    for x,y in points[1:]:path+=f' H{X(x):.3f} V{Y(y):.3f}'
    return path


def progress_panel(ledger,records,profile,expected):
    series,current,total=completion_history(ledger,records,profile)
    planned=expected*len(current);X=lambda x:75+700*x/max(1,planned);Y=lambda y:300-230*y/max(1,expected)
    out=['<!doctype html><html><head><meta charset="utf-8"><style>body{font:14px system-ui;margin:12px}svg{width:100%;height:auto}table{border-collapse:collapse}td,th{padding:5px 12px;border-bottom:1px solid #ddd}.note{font-size:12px;color:#60666d}</style></head><body>',
         '<h3>Completed runs by method and task</h3>',f'<p>{LABEL}</p>',
         f'<p><strong>{total} / {planned}</strong> validated completed runs. Each cell: out of {expected}.</p>',
         '<svg viewBox="0 0 830 375" role="img" aria-label="Cumulative completions, right-angle steps">',
         '<path d="M75 55V300H780" fill="none" stroke="#777"/>',
         '<text x="425" y="362" text-anchor="middle">Total runs completed (all method/task cells)</text>',
         '<text x="10" y="30">Completed seeds in each cell</text>']
    for x in sorted({0,planned//4,planned//2,3*planned//4,planned}):out.append(f'<text x="{X(x):.2f}" y="323" text-anchor="middle">{x}</text>')
    for y in sorted({0,expected//2,expected}):out.append(f'<text x="65" y="{Y(y)+4:.2f}" text-anchor="end">{y}</text>')
    dashes={TASKS[0]:'none',TASKS[1]:'6 3',TASKS[2]:'2 3'}
    for cell,points in series.items():
        method,task=cell.split('/')
        out.append(f'<path d="{step_path(points,X,Y)}" fill="none" stroke="{COLORS[method]}" stroke-width="1.8" stroke-dasharray="{dashes[task]}"><title>{html.escape(NAMES[method]+" / "+task)}</title></path>')
    out.extend(['</svg>','<p class="note">No smoothing or linear interpolation. Order is from recorded completion timestamps; simultaneous timestamps form one group. Blank future x-range contains no observations.</p>',
                '<table><thead><tr><th>Method</th><th>Task</th><th>Completed</th></tr></thead><tbody>'])
    for cell,value in current.items():
        method,task=cell.split('/');out.append(f'<tr><td style="color:{COLORS[method]}">{html.escape(NAMES[method])}</td><td>{html.escape(task)}</td><td>{value} / {expected}</td></tr>')
    out.append('</tbody></table></body></html>')
    return ''.join(out),{'planned':planned,'completed':total,'counts':current,'x_extent':[0,planned],'data_x_extent':[0,total]}


def curated_panels(ledger,records,profile,expected):
    payload=snapshot_payload(records,ledger,expected)
    panels=panel_html(payload)
    panels[PROGRESS],proof=progress_panel(ledger,records,profile,expected)
    if len(panels)!=18:raise ValueError('Curated dashboard must have exactly eighteen panels')
    return panels,proof


def archive_legacy(remote,project,run_id,enabled,now=None):
    summary=remote.predict(project=project,run=run_id,api_name='/get_run_summary')
    bad=legacy_metrics(summary.get('metrics',[]))
    if not bad:return None
    if not enabled:raise ValueError('Legacy presentation metrics require explicit archival permission')
    stamp=(now or datetime.now(timezone.utc)).strftime('%Y%m%dT%H%M%S%fZ')
    archive=run_id+'--legacy-ms-'+stamp
    ok=remote.predict(project=project,old_name=run_id,new_name=archive,api_name='/rename_run')
    if ok is not True:raise RuntimeError('Tracking metadata archival did not succeed')
    return {'archive_name':archive,'original_name':run_id,'log_rows_preserved':summary.get('num_logs'),
            'legacy_metric_count':len(bad),'delete_performed':False}


class Publisher:
    def __init__(self,run_id,archive=False,remote=None,client=None):
        self.env=dict(os.environ);self.env['ORX_RUN_ID']=run_id
        self.run_id=run_id;self.archive=archive;self.remote=remote;self.client=client;self.archives=[]
    def connect(self):
        if self.remote is None:
            from trackio.remote_client import RemoteClient
            source=self.env.get('TRACKIO_SPACE_ID') or self.env.get('TRACKIO_SERVER_URL')
            if not source:raise ValueError('Remote tracking is not configured')
            self.remote=RemoteClient(source,hf_token=self.env.get('HF_TOKEN'),write_token=self.env.get('TRACKIO_WRITE_TOKEN'))
        return self.remote
    def publish(self,panels,proof):
        remote=self.connect();project=self.env['TRACKIO_PROJECT']
        archived=archive_legacy(remote,project,self.run_id,self.archive)
        if archived:self.archives.append(archived)
        # A fresh SDK context resolves the authoritative post-rename run identity.
        sink=TrackioSink(environ=self.env,client=self.client);sink._init()
        metrics={LABEL+'/'+name:sink.client.Html(doc,caption=LABEL) for name,doc in panels.items()}
        sink.run.log(metrics,step=proof['completed'])
        if not sink.finish().get('ok'):raise RuntimeError('Tracking flush failed')
        summary=remote.predict(project=project,run=self.run_id,api_name='/get_run_summary')
        if legacy_metrics(summary.get('metrics',[])):raise RuntimeError('Concurrent legacy publisher detected')
        snapshot=remote.predict(project=project,run=self.run_id,step=proof['completed'],api_name='/get_snapshot')
        missing=sorted(set(metrics)-set(snapshot))
        invalid=[]
        for name in set(metrics)-set(missing):
            values=snapshot[name];value=values[-1].get('value') if values else None
            if not isinstance(value,dict) or value.get('_type')!='trackio.html' or value.get('caption')!=LABEL:invalid.append(name)
        if missing or invalid:raise RuntimeError('Curated remote media verification failed')
        return {'status':'PASS','run_id':self.run_id,'panel_count':len(metrics),'completed':proof['completed'],
                'counts':proof['counts'],'x_extent':proof['x_extent'],'data_x_extent':proof['data_x_extent'],
                'tracking_step':proof['completed'],'legacy_numeric_metrics':0,'archives':list(self.archives),
                'scientific_volume_mutations':0,'controller_or_worker_changes':False,
                'only_tracking_presentation_changed':True}


def parent_flush_seen(volume,run_id,profile):
    try:row=json.loads(b''.join(volume.read_file('/controller/tracking-final-flush.json')))
    except (FileNotFoundError,KeyError):return False
    return row.get('run_id')==run_id and row.get('profile')==profile


def synchronize(volume,publisher,profile,expected,cache,disk_cache):
    records,ledger,errors=collect(volume,profile,cache,disk_cache)
    if errors:raise ValueError('Authoritative result read errors; preserve previous display')
    panels,proof=curated_panels(ledger,records,profile,expected)
    return publisher.publish(panels,proof),proof


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['smoke','science'],required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--expected',type=int,choices=[2,20],required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--once',action='store_true');mode.add_argument('--loop',action='store_true')
    parser.add_argument('--archive-legacy',action='store_true');args=parser.parse_args(argv)
    if not re.fullmatch('[a-fA-F0-9-]{36}',args.run_id):parser.error('--run-id must be an existing tracked UUID')
    if (args.profile,args.expected) not in [('smoke',2),('science',20)]:parser.error('Profile/quota mismatch')
    out=ROOT/'results'/'dashboard-sync-v2';out.mkdir(parents=True,exist_ok=True)
    lock=(out/(args.run_id+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['TRACKIO_DIR']=str(out/('client-'+args.run_id))
    import modal
    volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=False,version=2)
    publisher=Publisher(args.run_id,args.archive_legacy);cache={}
    state_path=out/(args.run_id+'-receipt.json')
    while True:
        try:
            records,ledger,errors=collect(volume,args.profile,cache,out/'cache')
            if errors:raise ValueError('Authoritative result read errors; preserve previous display')
            panels,proof=curated_panels(ledger,records,args.profile,args.expected)
            complete=proof['completed']==proof['planned']
            waited=False
            if complete and args.loop:
                # Frozen controller's own final flush may append legacy counters once.
                # Wait only by reading its receipt, then preserve/archive that history too.
                while not parent_flush_seen(volume,args.run_id,args.profile):time.sleep(.5)
                records,ledger,errors=collect(volume,args.profile,cache,out/'cache')
                if errors:raise ValueError('Final authoritative result read errors')
                panels,proof=curated_panels(ledger,records,args.profile,args.expected)
                waited=True
            receipt=publisher.publish(panels,proof)
            receipt.update(profile=args.profile,parent_final_flush_observed=waited or
                (complete and parent_flush_seen(volume,args.run_id,args.profile)))
            atomic_json(state_path,receipt)
            if args.once or complete:return 0
        except Exception as error:
            receipt={'status':'FAIL','run_id':args.run_id,'profile':args.profile,'error_type':type(error).__name__,
                     'scientific_volume_mutations':0,'controller_or_worker_changes':False}
            atomic_json(state_path,receipt)
            if args.once:return 1
        time.sleep(60)


if __name__=='__main__':
    with quiet_tracking_output():raise SystemExit(main())
