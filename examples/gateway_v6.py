"""Runnable fictional local callback; never sends a message to anyone."""
import tempfile,time
from pathlib import Path
from cryptography.fernet import Fernet
from roguard import DurableGateway,ToolBinding

def main():
    captured=[]
    with tempfile.TemporaryDirectory() as directory:
        gateway=DurableGateway(Path(directory)/'state.db',Fernet.generate_key(),
            {'demo':{'capture':ToolBinding('lesson','teacher','export',lambda values:captured.append(values))}},binding_version='capture_v1')
        gateway.create('demo','session')
        gateway.mutate('demo','session','source',{'artifact_id':'lesson','value':'Fictional learning summary'})
        gateway.mutate('demo','session','permission',{'root':'lesson','purpose':'lesson','recipient':'teacher','operation':'export','action':'grant','expires_at':int(time.time())+60})
        job=gateway.enqueue('demo','session','capture',['lesson'],request_id='demo_request')['job_id']
        gateway.approve('demo',job)
        assert gateway.execute('demo',job)['status']=='EXECUTED'
        gateway.execute('demo',job)
        assert len(captured)==1
        print('Local fictional callback executed once; no external delivery.')
if __name__=='__main__':main()
