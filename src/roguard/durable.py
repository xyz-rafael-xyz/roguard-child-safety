"""Encrypted SQLite policy sessions and one-attempt jobs for trusted hosts.

SQLite serializes final authorization and synchronous local callbacks. Remote
receivers still require their own final check. A crashed STARTED job is uncertain
and is never automatically retried. Keys and callbacks belong to the host.
"""
from contextlib import contextmanager
import hashlib,json,os,secrets,sqlite3,time
from .flow import FlowLedger,ToolBinding,identifier,identifiers
from .repair import repair_proposal

class DurableGateway:
    def __init__(self,path,key,bindings=None,*,binding_version=None,clock=time.time):
        from cryptography.fernet import Fernet
        self.path=str(path);self.cipher=Fernet(key);self.clock=clock;self.bindings=bindings or {};self.binding_version=binding_version
        if self.bindings:identifier(binding_version,'binding version')
        if self.path==':memory:':raise ValueError('Durable gateway requires a file')
        for tenant,registry in self.bindings.items():
            identifier(tenant)
            for name,binding in registry.items():
                identifier(name)
                if not isinstance(binding,ToolBinding):raise ValueError('Invalid binding')
        # Private initial permissions; the parent directory and key are host responsibilities.
        fd=os.open(self.path,os.O_CREAT|os.O_RDWR,0o600);os.close(fd)
        with self._tx() as db:
            db.execute('CREATE TABLE IF NOT EXISTS clock_state (id INTEGER PRIMARY KEY, at INTEGER)')
            db.execute('INSERT OR IGNORE INTO clock_state VALUES(1,0)')
            db.execute('CREATE TABLE IF NOT EXISTS sessions (tenant TEXT, id TEXT, revision INTEGER, state BLOB, PRIMARY KEY(tenant,id))')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (tenant TEXT, id TEXT, session TEXT, revision INTEGER, status TEXT, created REAL, payload BLOB, PRIMARY KEY(tenant,id))')
    @contextmanager
    def _tx(self):
        db=sqlite3.connect(self.path,timeout=15,isolation_level=None)
        try:
            db.execute('BEGIN IMMEDIATE');yield db;db.commit()
        except BaseException:db.rollback();raise
        finally:db.close()
    def _pack(self,value):return self.cipher.encrypt(json.dumps(value,allow_nan=False).encode())
    def _unpack(self,value):return json.loads(self.cipher.decrypt(value))
    def _load(self,db,tenant,session):
        identifier(tenant);identifier(session)
        row=db.execute('SELECT revision,state FROM sessions WHERE tenant=? AND id=?',(tenant,session)).fetchone()
        if row is None:raise KeyError('Unknown session')
        envelope=self._unpack(row[1])
        if envelope['tenant']!=tenant or envelope['session']!=session:raise ValueError('Encrypted state context mismatch')
        state=envelope['events'];ledger=FlowLedger()
        for event in state:
            if event['kind']=='source':ledger.add_source(**event['args'])
            elif event['kind']=='derive':ledger.derive(**event['args'])
            else:ledger.permission(**event['args'])
        last=db.execute('SELECT at FROM clock_state WHERE id=1').fetchone()[0]
        now=max(last,int(self.clock()));db.execute('UPDATE clock_state SET at=? WHERE id=1',(now,));ledger.advance(now)
        return row[0],state,ledger
    def create(self,tenant,session):
        identifier(tenant);identifier(session)
        with self._tx() as db:
            if db.execute('SELECT COUNT(*) FROM sessions WHERE tenant=?',(tenant,)).fetchone()[0]>=100:raise ValueError('Tenant session limit')
            db.execute('INSERT INTO sessions VALUES(?,?,0,?)',(tenant,session,self._pack(dict(tenant=tenant,session=session,events=[]))))
        return {'session':session,'revision':0}
    def mutate(self,tenant,session,kind,args):
        if kind not in ('source','derive','permission') or not isinstance(args,dict):raise ValueError('Invalid mutation')
        with self._tx() as db:
            rev,state,ledger=self._load(db,tenant,session)
            try:
                if len(state)>=2048:raise ValueError('Session event limit')
                {'source':ledger.add_source,'derive':ledger.derive,'permission':ledger.permission}[kind](**args)
                state.append(dict(kind=kind,args=args));db.execute('UPDATE sessions SET revision=?,state=? WHERE tenant=? AND id=?',(rev+1,self._pack(dict(tenant=tenant,session=session,events=state)),tenant,session))
                # A changed policy/provenance invalidates every pending approval and proposal.
                db.execute("UPDATE jobs SET status='STALE' WHERE tenant=? AND session=? AND status IN ('PENDING','APPROVED')",(tenant,session))
            finally:ledger.close()
        return {'revision':rev+1}
    def plan(self,tenant,session,request):
        with self._tx() as db:
            rev,state,ledger=self._load(db,tenant,session)
            try:report=repair_proposal(ledger,**request);report['durable_revision']=rev;return report
            finally:ledger.close()
    def enqueue(self,tenant,session,tool,artifact_ids,*,request_id=None):
        identifier(tool);ids=identifiers(artifact_ids)
        if request_id is not None:identifier(request_id,'request ID')
        binding=self.bindings.get(tenant,{}).get(tool)
        if binding is None:raise ValueError('Unknown tenant tool')
        with self._tx() as db:
            rev,state,ledger=self._load(db,tenant,session)
            try:
                check=ledger.check(ids,binding.purpose,binding.recipient,binding.operation)
                if check['decision']!='ALLOW':raise ValueError('Ineligible job')
                if db.execute('SELECT COUNT(*) FROM jobs WHERE tenant=?',(tenant,)).fetchone()[0]>=1000:raise ValueError('Tenant job limit; delete completed sessions')
                job=request_id or secrets.token_hex(16)
                existing=db.execute('SELECT session,payload FROM jobs WHERE tenant=? AND id=?',(tenant,job)).fetchone()
                if existing:
                    prior=self._unpack(existing[1])
                    if existing[0]!=session or prior['tool']!=tool or prior['artifact_ids']!=list(ids):raise ValueError('Request ID conflict')
                    return {'job_id':job,'status':'EXISTING','revision':rev}
                db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)',(tenant,job,session,rev,'PENDING',ledger.clock,self._pack(dict(tool=tool,artifact_ids=ids,binding_version=self.binding_version,scope=[binding.purpose,binding.recipient,binding.operation]))))
            finally:ledger.close()
        return {'job_id':job,'status':'PENDING','revision':rev}
    def _job(self,db,tenant,job):
        identifier(tenant);identifier(job)
        row=db.execute('SELECT session,revision,status,created,payload FROM jobs WHERE tenant=? AND id=?',(tenant,job)).fetchone()
        if row is None:raise KeyError('Unknown job')
        return row
    def approve(self,tenant,job):
        with self._tx() as db:
            row=self._job(db,tenant,job)
            if row[2]!='PENDING':raise ValueError('Job is not awaiting approval')
            if not 0<=self.clock()-row[3]<=300:raise ValueError('Job expired')
            rev,state,ledger=self._load(db,tenant,row[0]);ledger.close()
            if rev!=row[1]:raise ValueError('Stale job')
            db.execute("UPDATE jobs SET status='APPROVED' WHERE tenant=? AND id=?",(tenant,job))
        return {'job_id':job,'status':'APPROVED'}
    def cancel(self,tenant,job):
        with self._tx() as db:
            row=self._job(db,tenant,job)
            if row[2] not in ('PENDING','APPROVED'):raise ValueError('Cannot cancel after execution starts')
            db.execute("UPDATE jobs SET status='CANCELLED' WHERE tenant=? AND id=?",(tenant,job))
        return {'job_id':job,'status':'CANCELLED'}
    def status(self,tenant,job):
        with self._tx() as db:row=self._job(db,tenant,job)
        return {'job_id':job,'status':'EXECUTION_UNCERTAIN' if row[2]=='STARTED' else row[2],
                'automatic_retry_allowed':False,'text_in_report':False}
    def execute(self,tenant,job):
        # Commit intent before any possible side effect, so a crash cannot enable replay.
        with self._tx() as db:
            row=self._job(db,tenant,job)
            if row[2]!='APPROVED':return self._result(job,row[2])
            db.execute("UPDATE jobs SET status='STARTED' WHERE tenant=? AND id=?",(tenant,job))
        with self._tx() as db:
            row=self._job(db,tenant,job);rev,state,ledger=self._load(db,tenant,row[0]);payload=self._unpack(row[4])
            try:
                binding=self.bindings.get(tenant,{}).get(payload['tool'])
                status='BLOCK'
                if rev==row[1] and 0<=ledger.clock-row[3]<=300 and binding and payload['binding_version']==self.binding_version and payload['scope']==[binding.purpose,binding.recipient,binding.operation]:
                    check=ledger.check(payload['artifact_ids'],binding.purpose,binding.recipient,binding.operation)
                    if check['decision']=='ALLOW':
                        try:
                            binding.callback({i:ledger._artifacts[i].value for i in payload['artifact_ids']});status='EXECUTED'
                        except Exception:status='EXECUTION_UNCERTAIN'
                db.execute('UPDATE jobs SET status=? WHERE tenant=? AND id=?',(status,tenant,job))
            finally:ledger.close()
        return self._result(job,status)
    @staticmethod
    def _result(job,status):
        return {'job_id':job,'status':'EXECUTION_UNCERTAIN' if status=='STARTED' else status,'automatic_retry_allowed':False,'external_action_taken':None if status in ('STARTED','EXECUTED','EXECUTION_UNCERTAIN') else False}
    def erase(self,tenant,session):
        identifier(tenant);identifier(session)
        with self._tx() as db:
            db.execute('DELETE FROM jobs WHERE tenant=? AND session=?',(tenant,session));db.execute('DELETE FROM sessions WHERE tenant=? AND id=?',(tenant,session))
        return {'erased':True,'cryptographic_erasure_guaranteed':False}
