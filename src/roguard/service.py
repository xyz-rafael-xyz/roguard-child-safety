"""Small ASGI service with explicit tenant/role credentials and bounded bodies."""
import argparse,asyncio,hashlib,hmac,json,os,time
from pathlib import Path
from .flow import identifier
from .cli import _strict_json_loads
from .conversation import analyze_conversation,ConversationScorer
from .long_context import analyze_long_conversation

class GatewayApp:
    def __init__(self,gateway,credentials,*,scorer=None,monitor=None,max_concurrency=4):
        self.gateway=gateway;self.scorer=scorer;self.monitor=monitor;self.credentials=[]
        for token,principal in credentials.items():
            if not isinstance(token,str) or len(token)<32 or set(principal)!={'tenant','role'} or principal['role'] not in ('reader','operator','admin'):raise ValueError('Invalid credentials')
            identifier(principal['tenant']);self.credentials.append((hashlib.sha256(token.encode()).digest(),dict(principal)))
        if not self.credentials:raise ValueError('Credentials required')
        self.limit=asyncio.Semaphore(max_concurrency)
    async def __call__(self,scope,receive,send):
        if scope['type']=='lifespan':
            while True:
                msg=await receive()
                await send({'type':msg['type']+'.complete'})
                if msg['type']=='lifespan.shutdown':return
        if scope['type']!='http':return
        headers={k.lower():v for k,v in scope.get('headers',[])}
        async def reply(code,data):
            await send({'type':'http.response.start','status':code,'headers':[(b'content-type',b'application/json'),(b'cache-control',b'no-store'),(b'x-content-type-options',b'nosniff')]})
            await send({'type':'http.response.body','body':json.dumps(data,allow_nan=False).encode()})
        auth=headers.get(b'authorization',b'');principal=None
        if auth.startswith(b'Bearer ') and len(auth)<=4096:
            hashed=hashlib.sha256(auth[7:]).digest()
            for expected,candidate in self.credentials:
                if hmac.compare_digest(hashed,expected):principal=candidate
        if principal is None:return await reply(401,{'error':'AUTHENTICATION_REQUIRED'})
        if b'origin' in headers:return await reply(403,{'error':'BROWSER_ORIGIN_NOT_ALLOWED'})
        if scope['method']!='POST':return await reply(405,{'error':'POST_REQUIRED'})
        if headers.get(b'content-type',b'').split(b';')[0]!=b'application/json':return await reply(415,{'error':'JSON_REQUIRED'})
        raw=bytearray()
        try:
            while True:
                msg=await asyncio.wait_for(receive(),timeout=10)
                if msg['type']=='http.disconnect':return
                raw.extend(msg.get('body',b''))
                if len(raw)>262144:return await reply(413,{'error':'BODY_TOO_LARGE'})
                if not msg.get('more_body'):break
            data=_strict_json_loads(raw.decode())
            if not isinstance(data,dict):raise ValueError()
            path=scope['path'];role=principal['role'];tenant=principal['tenant']
            if path in ('/v1/session/create','/v1/session/mutate','/v1/job/approve','/v1/session/erase','/v1/metrics/erase','/v1/job/execute') and role!='admin':return await reply(403,{'error':'ADMIN_REQUIRED'})
            if path in ('/v1/job/enqueue','/v1/job/cancel') and role=='reader':return await reply(403,{'error':'OPERATOR_REQUIRED'})
            if self.limit.locked():return await reply(429,{'error':'CAPACITY_REACHED'})
            async with self.limit:
                result=await asyncio.to_thread(self.route,path,tenant,data)
            await reply(200,result)
        except (ValueError,TypeError,KeyError,UnicodeError):await reply(400,{'error':'INVALID_REQUEST'})
        except asyncio.TimeoutError:await reply(408,{'error':'REQUEST_TIMEOUT'})
        except Exception:await reply(503,{'error':'SERVICE_UNAVAILABLE'})
    def route(self,path,tenant,data):
        g=self.gateway
        if path in ('/v1/analyze','/v1/analyze-long'):
            started=time.perf_counter();result=(analyze_conversation(data,self.scorer) if path.endswith('/analyze') else analyze_long_conversation(data,self.scorer))
            if self.monitor:
                decision=result['signal']['decision'] if 'signal' in result else result['decision']
                self.monitor.record(tenant=tenant,language=data['language'],decision=decision,latency_ms=(time.perf_counter()-started)*1000,model_id='v4-'+getattr(self.scorer,'backend','fast'),policy_version='v6')
            return result
        routes={'/v1/session/create':g.create,'/v1/session/mutate':g.mutate,'/v1/session/erase':g.erase,'/v1/repair':g.plan,'/v1/job/enqueue':g.enqueue,'/v1/job/approve':g.approve,'/v1/job/cancel':g.cancel,'/v1/job/status':g.status,'/v1/job/execute':g.execute}
        if path in routes:return routes[path](tenant=tenant,**data)
        if path=='/v1/metrics' and self.monitor:return self.monitor.report(tenant,**data)
        if path=='/v1/metrics/erase' and self.monitor and not data:self.monitor.erase(tenant);return {'erased':True}
        raise ValueError('Unknown route')


def main():
    p=argparse.ArgumentParser(description='Authenticated local gateway; reverse-proxy TLS required for network deployments')
    p.add_argument('--key-file',type=Path);p.add_argument('--state',type=Path,required=True);p.add_argument('--credentials',type=Path,required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8780);p.add_argument('--backend',choices=['fast','semantic','hybrid'],default='fast');p.add_argument('--model-path',type=Path)
    a=p.parse_args()
    from .durable import DurableGateway
    from .observability import Monitor
    import uvicorn
    key=a.key_file.read_text().strip() if a.key_file else os.environ.get('ROGUARD_STATE_KEY')
    if not key:p.error('Set ROGUARD_STATE_KEY to a Fernet key; keep it outside the database')
    credentials=_strict_json_loads(a.credentials.read_text());gateway=DurableGateway(a.state,key.encode());monitor=Monitor(str(a.state)+'.metrics')
    app=GatewayApp(gateway,credentials,scorer=ConversationScorer(a.backend,a.model_path),monitor=monitor)
    uvicorn.run(app,host=a.host,port=a.port,access_log=False,server_header=False)
if __name__=='__main__':main()
