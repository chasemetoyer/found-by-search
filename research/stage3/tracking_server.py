"""Private Trackio ASGI service; deploy only from the tracked remote bootstrap.

The server and observer are separate from PPO. Tracking state uses its own Modal
volume. A secret capability prefix gates all read, write, media and UI routes;
stock Trackio's write token alone would not protect read access.
"""
from __future__ import annotations
import hmac
import asyncio
import inspect
import os

# This is the DEPLOYMENT name used by controller.deploy and guardian.lookup.
APP_NAME='evolved-intrinsic-rewards-stage3-monitor'
# Preserve the existing volume independently of the deployment name.
VOLUME_NAME='evolved-intrinsic-rewards-stage3-tracking'
PROJECT_ID='64555583-12e3-4641-82a0-b8841b663e70'


class CapabilityPrefix:
    def __init__(self, app, token, after_write=None):
        if not isinstance(token,str) or len(token)<32 or '/' in token:
            raise ValueError('Tracking access token missing or too short')
        self.app=app;self.token=token;self.after_write=after_write;self._commit_lock=asyncio.Lock()

    async def __call__(self,scope,receive,send):
        if scope['type']=='lifespan':
            async def lifecycle_send(message):
                if message['type']=='lifespan.shutdown.complete':await self._commit()
                await send(message)
            return await self.app(scope,receive,lifecycle_send)
        path=scope.get('path','');parts=path.split('/',3)
        valid=(len(parts)>=3 and parts[1]=='private' and hmac.compare_digest(parts[2],self.token))
        if not valid:
            if scope['type']=='websocket':return await send({'type':'websocket.close','code':1008})
            await send({'type':'http.response.start','status':404,'headers':[(b'content-type',b'text/plain'),(b'cache-control',b'no-store')]})
            return await send({'type':'http.response.body','body':b'Not found'})
        prefix='/private/'+self.token
        inner=dict(scope)
        inner['path']=path[len(prefix):] or '/'
        inner['raw_path']=inner['path'].encode()
        inner['root_path']=scope.get('root_path','')+prefix
        success=False
        async def wrapped_send(message):
            nonlocal success
            if message['type']=='http.response.start':success=200<=message['status']<300
            await send(message)
        await self.app(inner,receive,wrapped_send)
        if success and scope.get('method') in ('POST','PUT','PATCH','DELETE'):
            await self._commit()

    async def _commit(self):
        if self.after_write:
            async with self._commit_lock:
                result=self.after_write()
                if inspect.isawaitable(result):await result


def create_private_server(*,after_write=None):
    token=os.environ.get('TRACKIO_ACCESS_PATH_TOKEN','')
    if not os.environ.get('TRACKIO_WRITE_TOKEN'):raise ValueError('Tracking write token is required')
    # Must precede Trackio import; SQLite/media paths are established at import.
    os.environ['TRACKIO_DIR']='/tracking'
    from trackio.server import build_starlette_app_only
    app,_=build_starlette_app_only(mcp_server=False)
    return CapabilityPrefix(app,token,after_write=after_write)


def make_tracking_app(image):
    """Return a service definition; no deploy/compute happens on import."""
    import modal
    values={key:os.environ.get(key) for key in ('TRACKIO_ACCESS_PATH_TOKEN','TRACKIO_WRITE_TOKEN')}
    if not all(values.values()):raise ValueError('Tracking endpoint credentials missing')
    volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=True,version=2)
    app=modal.App(APP_NAME,tags={'project_id':PROJECT_ID,'stage':'3','role':'tracking'},include_source=False)
    @app.function(image=image,volumes={'/tracking':volume},secrets=[modal.Secret.from_dict(values)],serialized=True,
                  cpu=(.25,.25),memory=(512,512),max_containers=1,min_containers=1,
                  timeout=3600,scaledown_window=30)
    @modal.concurrent(max_inputs=64)
    @modal.asgi_app()
    def tracking_server():
        return create_private_server(after_write=volume.commit.aio)
    @app.function(image=image,volumes={'/tracking':volume},serialized=True,
                  secrets=[modal.Secret.from_dict({k:os.environ[k] for k in ('MODAL_TOKEN_ID','MODAL_TOKEN_SECRET')})] if all(os.environ.get(k) for k in ('MODAL_TOKEN_ID','MODAL_TOKEN_SECRET')) else [],
                  retries=modal.Retries(max_retries=3,initial_delay=1.,max_delay=4.),
                  cpu=(.125,.125),memory=(128,128),timeout=43320,
                  max_containers=5,min_containers=0,scaledown_window=2)
    def expire_tracking(lease_id: str,expires_unix: float):
        import time,json
        from pathlib import Path
        import modal
        time.sleep(max(0,expires_unix-time.time()))
        volume.reload()
        lease=json.loads(Path('/tracking/service-lease.json').read_text())
        from stage2.transport import retry_io
        return stop_if_current_lease(lease_id,lease,lambda:retry_io(lambda:modal.App.lookup(APP_NAME).stop(),deadline=expires_unix+90))
    return app,tracking_server,volume


def expiry_function(app):
    return next(fn for name,fn in app.registered_functions.items() if name.endswith('expire_tracking'))


def private_base_url(public_endpoint):
    token=os.environ.get('TRACKIO_ACCESS_PATH_TOKEN','')
    if len(token)<32:raise ValueError('Tracking access token missing')
    return public_endpoint.rstrip('/')+'/private/'+token


def stop_if_current_lease(lease_id,lease,stop):
    if lease['lease_id']!=lease_id:return {'status':'superseded','lease_id':lease_id}
    stop()
    return {'status':'stop_requested','lease_id':lease_id}
