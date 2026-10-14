"""Opt-in bounded metadata-only monitoring. No automatic exporter or telemetry."""
import json,math,sqlite3,time
from collections import Counter
from .flow import identifier

DECISIONS={'NO_REVIEW_SIGNAL','ALLOW','BLOCK','REVIEW','REVIEW_SUPPORT_SIGNAL','NO_MODEL_SIGNAL','INSUFFICIENT_EVIDENCE','EXECUTED','EXECUTION_UNCERTAIN','CANCELLED','FAILED'}
class Monitor:
    def __init__(self,path=':memory:',*,retention_seconds=86400,max_events=10000,tracer=None,clock=time.time):
        if type(retention_seconds) is not int or not 60<=retention_seconds<=2592000 or type(max_events) is not int or not 10<=max_events<=100000:raise ValueError('Invalid retention bounds')
        self.path=str(path);self.retention=retention_seconds;self.limit=max_events;self.tracer=tracer;self.clock=clock
        self.db=sqlite3.connect(self.path,check_same_thread=False);self.db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, at REAL NOT NULL, tenant TEXT NOT NULL, payload TEXT NOT NULL)');self.db.commit()
        from threading import RLock
        self.lock=RLock()
    def record(self,*,tenant,language,decision,latency_ms,model_id,policy_version,dataset_id='none'):
        identifier(tenant);identifier(model_id);identifier(policy_version);identifier(dataset_id)
        if language not in ('ro','uk') or decision not in DECISIONS or type(latency_ms) not in (float,int) or not math.isfinite(latency_ms) or not 0<=latency_ms<=3600000:raise ValueError('Invalid monitoring event')
        event=dict(language=language,decision=decision,latency_ms=latency_ms,model_id=model_id,policy_version=policy_version,dataset_id=dataset_id)
        with self.lock:
            self.db.execute('INSERT INTO events(at,tenant,payload) VALUES(?,?,?)',(self.clock(),tenant,json.dumps(event)));self._purge();self.db.commit()
        if self.tracer is not None:
            # Host-configured SDK exporter only; exception events never contain messages.
            with self.tracer.start_as_current_span('roguard.decision',record_exception=False,set_status_on_exception=False) as span:
                for k,v in event.items():span.set_attribute('roguard.'+k,v)
        return event
    def _purge(self):
        self.db.execute('DELETE FROM events WHERE at<?',(self.clock()-self.retention,))
        self.db.execute('DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT ?)',(self.limit,))
    def report(self,tenant,*,baseline=None):
        identifier(tenant)
        with self.lock:
            self._purge();self.db.commit();rows=[json.loads(r[0]) for r in self.db.execute('SELECT payload FROM events WHERE tenant=? ORDER BY id',(tenant,))]
        result={}
        for lang in ('ro','uk'):
            sample=[r for r in rows if r['language']==lang];counts=Counter(r['decision'] for r in sample);lat=sorted(r['latency_ms'] for r in sample);n=len(sample)
            result[lang]=dict(count=n,decisions=dict(counts),abstention_rate=counts['INSUFFICIENT_EVIDENCE']/n if n else None,p95_ms=lat[math.ceil(.95*n)-1] if n else None,models=sorted({r['model_id'] for r in sample}))
            if baseline:
                ref=baseline.get('languages',{}).get(lang,{}).get('decisions',{});total=sum(ref.values())
                if any(type(x) is not int or x<0 for x in ref.values()):raise ValueError('Invalid baseline')
                # Total variation: descriptive distribution movement, no accuracy inference.
                distance=.5*sum(abs(counts[k]/n-ref.get(k,0)/total) for k in set(counts)|set(ref)) if n and total else None
                result[lang]['distribution_distance']=distance
                result[lang]['distribution_alert']=n>=30 and total>=30 and distance>.2 if distance is not None else False
        return dict(languages=result,accuracy_established=False,raw_text_stored=False,retention_seconds=self.retention)
    def erase(self,tenant):
        identifier(tenant)
        with self.lock:self.db.execute('DELETE FROM events WHERE tenant=?',(tenant,));self.db.commit()
    def close(self):self.db.close()
