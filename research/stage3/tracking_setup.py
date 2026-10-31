"""Private remote Trackio endpoint setup; creates no experiment observations.

Only call after authorized by the lead. Tokens stay in environment/app settings.
No Trackio run is initialized here; the tracked Modal smoke run owns the first log.
"""
from __future__ import annotations
import argparse,contextlib,io,json,os
from pathlib import Path
from urllib.request import Request,urlopen

DEFAULT_SPACE='cjm249/evolved-intrinsic-rewards-stage3-monitor'
PROJECT_ID='64555583-12e3-4641-82a0-b8841b663e70'


def save_env(key,value,base='http://127.0.0.1:4792'):
    request=Request(base+'/api/settings/env',data=json.dumps({'key':key,'value':value}).encode(),
                    headers={'Content-Type':'application/json'},method='POST')
    with urlopen(request,timeout=15) as response:response.read()  # never echo settings data.


def setup(space_id=DEFAULT_SPACE,*,save_settings=True):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    from trackio.deploy import create_space_if_not_exists
    api=HfApi();identity=api.whoami()
    if space_id.split('/')[0]!=identity['name']:raise ValueError('Tracking Space must belong to configured account')
    try:info=api.space_info(space_id)
    except RepositoryNotFoundError:info=None
    if info is not None and not info.private:raise ValueError('Refusing existing public tracking Space')
    # Explicit persistence ID avoids accidental ephemeral-only tracking.
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        create_space_if_not_exists(space_id,bucket_id=space_id,private=True)
    info=api.space_info(space_id)
    if not info.private:raise ValueError('Tracking Space did not remain private')
    runtime=api.get_space_runtime(space_id)
    if save_settings:
        save_env('TRACKIO_SPACE_ID',space_id)
        save_env('TRACKIO_BUCKET_ID',space_id)
        if os.environ.get('HF_TOKEN'):save_env('HF_TOKEN',os.environ['HF_TOKEN'])
    return {'space_id':space_id,'private':info.private,'space_commit':info.sha,
            'space_stage':str(runtime.stage),'hardware':str(runtime.hardware),
            'bucket_id':space_id,'app_configuration_saved':save_settings,'tracking_runs_created':0,
            'scientific_runs_created':0,'visualize_smoke_verification_pending':True}


def inspect_app():
    with urlopen(f'http://127.0.0.1:4792/api/trackio/projects/{PROJECT_ID}/runs',timeout=30) as response:
        result=json.load(response)
    # Never return signed URL credentials; dashboard existence plus listed IDs suffice.
    return {'dashboard_present':bool(result.get('dashboardUrl')),'project':result.get('project'),
            'run_count':len(result.get('runs',[])),'run_ids':result.get('runs',[]),
            'signed_dashboard_access_present':bool(result.get('dashboardSign'))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--space',default=DEFAULT_SPACE);p.add_argument('--setup',action='store_true');p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    try:result=setup(args.space) if args.setup else inspect_app();result['ok']=True
    except Exception as error:
        result={'ok':False,'error_type':type(error).__name__}
        response=getattr(error,'response',None)
        if response is not None:result['http_status']=response.status_code
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
