"""Isolated Stage3 tracking projection: reads supplied observations, never learner state.

This module has no training/controller imports. Trackio is imported lazily by the
separate observer process. Native Trackio media panels appear inside Visualize;
no standalone HTML dashboard is produced or substituted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import html
import contextlib
import math
import logging
import os
from urllib.parse import urlparse

LABEL = 'in progress, not the stage-3 report'
TASKS = ('DoorKey-8x8', 'UnlockPickup', 'KeyCorridorS3R2')
METHODS = ('E', 'R', 'T', 'N', 'S')
NAMES = {'E':'E (CMA-ES)', 'R':'R (random search)', 'T':'T (task-only)',
         'N':'N (episodic novelty)', 'S':'fixed satiation',
         'tamper_on':'tamper on', 'tamper_off':'tamper off'}
COLORS = dict(zip((*METHODS, 'tamper_on', 'tamper_off'),
                  ('#0072b2','#009e73','#333333','#e69f00','#cc79a7','#d55e00','#0072b2')))
ALIASES = {
    'success': ('train/success',),
    'presses_episode': ('tamper/presses_per_episode', 'presses_per_episode', 'train/tampers'),
    'presses_step': ('tamper/presses_per_step', 'presses_per_step'),
    'intrinsic_return': ('intrinsic/episode_return', 'episode_intrinsic_return_mean', 'train/int_ret'),
    'entropy': ('policy/entropy', 'entropy', 'losses/entropy'),
}


def _finite(value):
    return isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _method(record):
    aliases={'fixed_satiation':'S','fixed satiation':'S','satiation':'S',
             'tamper-on':'tamper_on','tamper-off':'tamper_off'}
    return aliases.get(record['method'],record['method'])


def snapshot_payload(records, ledger, expected=20):
    """Build descriptive data from complete seeds only; reject duplicate observations.

    Record contract: method/task/seed, metrics=list[dict], area, solved, cpu_class.
    Incomplete or failed transport reads must never be passed as observations.
    Missing telemetry stays missing. Only omitted train/success is zero, matching the registered scorer.
    """
    records=list(records);seen=set();clean=[]
    for record in records:
        method=_method(record);task=record['task'].removeprefix('MiniGrid-').removesuffix('-v0')
        if method not in (*METHODS,'tamper_on','tamper_off') or task not in TASKS:
            raise ValueError('Unknown Stage3 tracking cell')
        identity=(method,task,record['seed'])
        if identity in seen:raise ValueError('Duplicate completed seed in tracking projection')
        seen.add(identity)
        if not _finite(record['area']) or not 0<=record['area']<=1:
            raise ValueError('Invalid complete-seed curve area')
        metrics=record.get('metrics',[])
        if not isinstance(metrics,list):raise ValueError('metrics must be logged rows')
        steps=[m.get('global_step') for m in metrics]
        if any(not _finite(s) or s<=0 for s in steps) or steps!=sorted(set(steps)):
            raise ValueError('Tracking metrics require unique increasing environment steps')
        clean.append(dict(record,method=method,task=task))
    cells={f'{method}/{task}':sum(r['method']==method and r['task']==task for r in clean)
           for method in METHODS for task in TASKS}
    cells.update({f'{m}/DoorKey-8x8':sum(r['method']==m for r in clean) for m in ('tamper_on','tamper_off')})
    calls=list(ledger.get('calls',{}).values())
    return {'label':LABEL,'records':clean,'completed_by_cell':cells,'expected_per_cell':expected,
            'guard_usd':float(ledger.get('guard_total_usd',0)),
            'cap_usd':float(ledger.get('limit_usd',15)),
            'candidate_failures':int(ledger.get('candidate_failures',sum(c.get('outcome')=='candidate_failure' or c.get('status')=='numerical_failure' for c in calls))),
            'infrastructure_failures':int(ledger.get('infrastructure_failures',len(ledger.get('infrastructure_history',[])))),
            'observed_restarts':ledger.get('observed_restarts'),
            'unresolved_inputs':sum(c.get('status')=='uncertain' for c in calls),
            'cpu_mix':dict(Counter(r.get('cpu_class','unknown') for r in clean)),
            'cpu_mix_scope':'completed seed attempts; physical-attempt mix when supplied by observer',
            'physical_cpu_mix':ledger.get('physical_cpu_mix'),
            'rounded_episode_start_payoff_reference':.64,
            'inferential_claims':False}


def curve_series(records, quantity):
    """One colored method mean and faint complete-seed lines, aligned on actual steps."""
    series=[];by_method=defaultdict(list)
    for r in records:
        points=[]
        for m in r.get('metrics',[]):
            value=next((m[k] for k in ALIASES[quantity] if k in m),0. if quantity=='success' else None)
            if _finite(value):points.append((m['global_step'],float(value)))
        if points:
            method=r['method'];by_method[method].append(points)
            series.append({'name':f"{NAMES[method]} seed {r['seed']}",'points':points,'color':COLORS[method],'opacity':.18,'mean':False})
    for method,seeds in by_method.items():
        values=defaultdict(list)
        for points in seeds:
            for step,value in points:values[step].append(value)
        series.append({'name':f'{NAMES[method]} mean ({len(seeds)} completed seeds)',
                       'points':[(s,math.fsum(values[s])/len(values[s])) for s in sorted(values)],
                       'color':COLORS[method],'opacity':1.,'mean':True})
    return series


def _document(title,body):
    return ('<!doctype html><html><head><meta charset="utf-8"><style>'
            'body{font:14px system-ui,sans-serif;margin:12px;color:#20252a}svg{width:100%;height:auto}'
            '.note{color:#60666d;font-size:12px}</style></head><body>'
            f'<h3>{html.escape(title)}</h3><p class="note">{LABEL}</p>'+body+'</body></html>')


def _chart(title,series,ylabel,reference=None):
    points=[p for s in series for p in s['points']]
    if not points:return _document(title,'<p>No completed-seed telemetry available yet.</p>')
    xmin=min(p[0] for p in points);xmax=max(p[0] for p in points)
    lo=min(0,min(p[1] for p in points));hi=max(p[1] for p in points)
    if reference is not None:hi=max(hi,reference)
    if hi==lo:hi=lo+1
    if xmax==xmin:xmin=0
    X=lambda x:70+(x-xmin)/max(1,xmax-xmin)*700
    Y=lambda y:310-(y-lo)/(hi-lo)*240
    body=['<svg viewBox="0 0 820 380" role="img" aria-label="'+html.escape(title)+'">',
          '<path d="M70 55V310H775" fill="none" stroke="#80868c"/>',
          f'<text x="425" y="365" text-anchor="middle">environment steps</text>',
          f'<text x="12" y="35">{html.escape(ylabel)}</text>']
    for i in range(5):
        frac=i/4;x=xmin+frac*(xmax-xmin);y=lo+frac*(hi-lo)
        body.extend([f'<text x="{X(x):.2f}" y="332" font-size="11" text-anchor="middle">{x:g}</text>',
                     f'<text x="64" y="{Y(y)+4:.2f}" font-size="11" text-anchor="end">{y:.3g}</text>'])
    if reference is not None:
        body.append(f'<path d="M70 {Y(reference):.2f}H775" stroke="#555" stroke-dasharray="5 4"/><text x="775" y="{Y(reference)-6:.2f}" text-anchor="end">P0 = {reference:.2f} (rounded episode reference)</text>')
    for s in series:
        path=' '.join(f'{X(x):.3f},{Y(y):.3f}' for x,y in s['points'])
        body.append(f'<polyline points="{path}" fill="none" stroke="{s["color"]}" opacity="{s["opacity"]}" stroke-width="{2.5 if s["mean"] else 1}"><title>{html.escape(s["name"])}</title></polyline>')
        if len(s['points'])==1:
            x,y=s['points'][0];body.append(f'<circle cx="{X(x):.2f}" cy="{Y(y):.2f}" r="3" fill="{s["color"]}"><title>{html.escape(s["name"])}</title></circle>')
    body.append('</svg>')
    body.append('<p class="note">'+' · '.join(html.escape(s['name']) for s in series if s['mean'])+'</p>')
    return _document(title,''.join(body))


def _bars(title,values,note=''):
    if not values:return _document(title,'<p>No observations available yet.</p>')
    maximum=max(1,max(v for _,v in values if v is not None));body=['<svg viewBox="0 0 860 '+str(35*len(values)+35)+'">']
    for i,(name,value) in enumerate(values):
        value=0 if value is None else value;y=i*35+12
        body.append(f'<text x="5" y="{y+16}" font-size="12">{html.escape(name)}</text><rect x="300" y="{y}" width="{value/maximum*430:.2f}" height="22" fill="#0072b2"/><text x="740" y="{y+16}">{value:g}</text>')
    body.extend(['</svg>',f'<p class="note">{html.escape(note)}</p>'])
    return _document(title,''.join(body))


def panel_html(payload):
    """Native Trackio media payloads, not a standalone dashboard fallback."""
    records=payload['records'];panels={}
    panels['progress/completed by method and task']=_bars('Completed runs per method × task',
        [(NAMES.get(name.split('/')[0],name.split('/')[0])+' / '+name.split('/',1)[1]+f" / {payload['expected_per_cell']}",n) for name,n in payload['completed_by_cell'].items()])
    panels['progress/spend and cap']=_bars('Spend guard against hard cap',
        [('Conservative guard USD',payload['guard_usd']),('Hard cap USD',payload['cap_usd'])],
        'Guard is not an invoice. These values overlap; do not add them.')
    panels['progress/failures and restarts']=_bars('Failures and restarts',
        [('Numerical training failures',payload['candidate_failures']),('Infrastructure failures',payload['infrastructure_failures']),
         ('Unresolved inputs (cause pending)',payload['unresolved_inputs'])]+
        ([('Observed restarts (cause may be unknown)',payload['observed_restarts'])] if payload['observed_restarts'] is not None else []),
        'An unavailable restart count is omitted; missing evidence is not zero.')
    for task in TASKS:
        rs=[r for r in records if r['task']==task and r['method'] in METHODS]
        panels[f'learning/{task}']=_chart(task+' · training success',curve_series(rs,'success'),'success')
        # Native table plus a bounded scatter shows every completed seed without binning choices.
        dots=['<svg viewBox="0 0 820 240">']
        for method in METHODS:
            i=METHODS.index(method);dots.append(f'<text x="10" y="{i*40+25}">{html.escape(NAMES[method])}</text>')
            for j,r in enumerate(sorted((r for r in rs if r['method']==method),key=lambda r:r['seed'])):
                x=210+r['area']*560;y=i*40+20+((j%5)-2)*3
                dots.append(f'<circle cx="{x:.2f}" cy="{y}" r="3" fill="{COLORS[method]}"><title>seed {r["seed"]}: area {r["area"]:.8g}</title></circle>')
        dots.append('<text x="210" y="230">0</text><text x="485" y="230">curve area</text><text x="770" y="230">1</text></svg>')
        panels[f'distributions/{task}']=_document(task+' · completed per-seed curve areas',''.join(dots))
        panels[f'solves/{task}']=_bars(task+' · solve counts',[(NAMES[m]+f" / {payload['expected_per_cell']}",sum(bool(r['solved']) for r in rs if r['method']==m)) for m in METHODS],
            'Counts include completed seeds only; incomplete seeds are not failures.')
    tamper=[r for r in records if r['method'] in ('tamper_on','tamper_off')]
    for quantity,title,unit in [('success','success','success'),('presses_episode','presses per episode','presses / completed episode'),
                                ('presses_step','presses per step','presses / environment step'),
                                ('intrinsic_return','intrinsic episode return','undiscounted intrinsic return / episode'),
                                ('entropy','policy entropy','policy entropy')]:
        panels['tamper/'+title]=_chart('DoorKey-8x8 tamper on/off · '+title,curve_series(tamper,quantity),unit,
                                      reference=.64 if quantity=='intrinsic_return' else None)
    mix=payload['physical_cpu_mix'] or payload['cpu_mix']
    panels['cpu/class mix']=_bars('CPU class mix',sorted(mix.items()),
        'All observed physical attempts.' if payload['physical_cpu_mix'] else payload['cpu_mix_scope'])
    return panels


@contextlib.contextmanager
def quiet_tracking_output():
    """Use around the whole observer lifetime to include asynchronous upload warnings."""
    previous=logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with open(os.devnull,'w') as discard:
            with contextlib.redirect_stdout(discard),contextlib.redirect_stderr(discard):
                yield
    finally:
        logging.disable(previous)


def remote_readback(remote, project, run, payload=None, step=None):
    """Read one server snapshot; receipts contain only safe counts, never media URLs."""
    summary=remote.predict(project=project,run=run,api_name='/get_run_summary')
    required={f'{LABEL}/{name}' for name in panel_html(snapshot_payload([],{}))}
    missing=sorted(required-set(summary.get('metrics',[])))
    result={'ok':summary.get('num_logs',0)>0 and not missing,
            'num_logs':summary.get('num_logs',0),'panel_count':len(required),
            'missing_panels':missing,'last_step':summary.get('last_step')}
    if payload is not None:
        snapshot=remote.predict(project=project,run=run,step=step,api_name='/get_snapshot')
        wanted={f'{LABEL}/progress/completed/{k}':v for k,v in payload['completed_by_cell'].items()}
        wanted[f'{LABEL}/progress/completed_total']=sum(payload['completed_by_cell'].values())
        checked={};mismatches=[]
        for name,expected in wanted.items():
            observed=snapshot.get(name,[])
            value=observed[-1].get('value') if observed else None
            checked[name]=value if _finite(value) else None
            if not _finite(value) or value!=expected:mismatches.append(name)
        missing_final=sorted(required-set(snapshot))
        invalid_media=[]
        for name in required-set(missing_final):
            values=snapshot.get(name,[])
            value=values[-1].get('value') if values else None
            if not isinstance(value,dict) or value.get('_type')!='trackio.html' or value.get('caption')!=LABEL:
                invalid_media.append(name)
        result.update(completed_scalar_checks=checked,completed_scalar_mismatches=mismatches,
                      final_snapshot_missing_panels=missing_final,final_snapshot_invalid_media=sorted(invalid_media),
                      completed_total=sum(payload['completed_by_cell'].values()),
                      verified_snapshot_step=step)
        result['ok']=result['ok'] and not mismatches and not missing_final and not invalid_media
    return result


class TrackioSink:
    """Only for a separate observer process: any tracking error is data, never fatal."""
    def __init__(self, *, environ=None, client=None):
        self.env=dict(os.environ if environ is None else environ);self.client=client;self.run=None

    def _init(self):
        production_client=self.client is None
        if production_client:
            import trackio
            self.client=trackio
        space=self.env.get('TRACKIO_SPACE_ID');server=self.env.get('TRACKIO_SERVER_URL')
        if not space and not server:raise ValueError('Remote tracking is not configured')
        if server:
            parsed=urlparse(server)
            if parsed.scheme!='https' or parsed.hostname in ('localhost','127.0.0.1','0.0.0.0'):
                raise ValueError('Tracking needs a remote HTTPS server')
            if parsed.query:raise ValueError('Use token environment variables, not credential URLs')
        if space and production_client:
            # Trackio private=True only applies on creation; reject any existing public Space.
            from huggingface_hub import HfApi
            from huggingface_hub.errors import RepositoryNotFoundError
            try:
                info=HfApi().space_info(space)
            except RepositoryNotFoundError:
                pass  # init below creates a private Space if permitted.
            else:
                if not info.private:raise ValueError('Configured tracking Space is public')
        project=self.env.get('TRACKIO_PROJECT');name=self.env.get('ORX_RUN_ID');group=self.env.get('ORX_EXPERIMENT')
        if not project or not name or not group:raise ValueError('Tracked run identity is incomplete')
        self.run=self.client.init(project=project,name=name,group=group,
            space_id=space,server_url=server,resume='allow',private=True,embed=False,
            auto_log_gpu=False,auto_log_cpu=False,
            config={'label':LABEL,'read_only_volume_observer':True,'training_dependency':False,'commit':self.env.get('ORX_COMMIT')})

    def publish(self,payload,step):
        with quiet_tracking_output():return self._publish(payload,step)

    def _publish(self,payload,step):
        try:
            if self.run is None:self._init()
            metrics={f'{LABEL}/progress/completed/{k}':v for k,v in payload['completed_by_cell'].items()}
            metrics.update({f'{LABEL}/budget/guard_usd':payload['guard_usd'],f'{LABEL}/budget/cap_usd':payload['cap_usd'],
                            f'{LABEL}/progress/completed_total':sum(payload['completed_by_cell'].values())})
            panels=panel_html(payload)
            metrics.update({f'{LABEL}/{name}':self.client.Html(doc,caption=LABEL) for name,doc in panels.items()})
            self.run.log(metrics,step=step)
            return {'ok':True,'panel_count':len(panels),'queued':True,'remote_verified':False,'step':step}
        except Exception as error:
            # Never echo request URLs/credential-bearing exception text.
            return {'ok':False,'error_type':type(error).__name__,'step':step}

    def verify_remote(self, payload=None, step=None):
        """Independent server readback, including exact completed counts when supplied."""
        with quiet_tracking_output():
            try:
                from trackio.remote_client import RemoteClient
                source=self.env.get('TRACKIO_SPACE_ID') or self.env.get('TRACKIO_SERVER_URL')
                if not source:raise ValueError('Remote tracking is not configured')
                remote=RemoteClient(source,hf_token=self.env.get('HF_TOKEN'),write_token=self.env.get('TRACKIO_WRITE_TOKEN'))
                return remote_readback(remote,self.env['TRACKIO_PROJECT'],self.env['ORX_RUN_ID'],payload,step)
            except Exception as error:return {'ok':False,'error_type':type(error).__name__}

    def finish(self):
        with quiet_tracking_output():return self._finish()

    def _finish(self):
        try:
            if self.run is not None:self.run.finish()
            return {'ok':True}
        except Exception as error:return {'ok':False,'error_type':type(error).__name__}
