"""Local commands for v6 reliability workflows."""
import argparse,json,os,secrets
from pathlib import Path
from .cli import _strict_json_loads

def repair_demo(language='ro'):
    from .flow import FlowLedger
    from .repair import repair_proposal
    if language not in ('ro','uk'):raise ValueError('Unsupported language')
    with FlowLedger() as ledger:
        ledger.add_source('lesson','Exercițiu de adunare' if language=='ro' else 'Вправа на додавання')
        ledger.add_source('private_note','Notă personală fictivă' if language=='ro' else 'Вигадана особиста нотатка')
        ledger.permission('lesson','tutoring','teacher','export','grant')
        return repair_proposal(ledger,['lesson','private_note'],[dict(artifact_id='lesson',covers=['learning'],cost=1),dict(artifact_id='private_note',covers=[],cost=9)],['learning'],purpose='tutoring',recipient='teacher',operation='export')

def main():
    p=argparse.ArgumentParser(description='RoGuard reliability workflows; local by default');sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('repair-demo');d.add_argument('--language',choices=['ro','uk'],default='ro')
    l=sub.add_parser('long');l.add_argument('input',type=Path);l.add_argument('--backend',choices=['fast','semantic','hybrid'],default='fast');l.add_argument('--model-path',type=Path)
    c=sub.add_parser('calibrate');c.add_argument('input',type=Path);c.add_argument('--output',type=Path,required=True)
    m=sub.add_parser('monitor');m.add_argument('database',type=Path);m.add_argument('--tenant',required=True);m.add_argument('--baseline',type=Path)
    i=sub.add_parser('init');i.add_argument('--directory',type=Path,required=True)
    a=p.parse_args()
    def read(path):
        if path.stat().st_size>2000000:raise ValueError('Input too large')
        return _strict_json_loads(path.read_text())
    try:
        if a.command=='repair-demo':result=repair_demo(a.language)
        elif a.command=='long':
            from .conversation import ConversationScorer
            from .long_context import analyze_long_conversation
            result=analyze_long_conversation(read(a.input),ConversationScorer(a.backend,a.model_path))
        elif a.command=='calibrate':
            from .uncertainty import CalibratedDecisions
            result=CalibratedDecisions.fit(**read(a.input)).artifact
            with a.output.open('x') as f:json.dump(result,f,indent=2)
        elif a.command=='monitor':
            from .observability import Monitor
            monitor=Monitor(a.database)
            try:result=monitor.report(a.tenant,baseline=read(a.baseline) if a.baseline else None)
            finally:monitor.close()
        else:
            from cryptography.fernet import Fernet
            a.directory.mkdir(mode=0o700,parents=False,exist_ok=False)
            for name,value in [('state.key',Fernet.generate_key().decode()),('credentials.json',json.dumps({secrets.token_urlsafe(32):dict(tenant='local',role=role) for role in ('admin','operator','reader')},indent=2))]:
                fd=os.open(str(a.directory/name),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'w') as stream:stream.write(value)
            result={'initialized':str(a.directory),'credentials_printed':False}
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,KeyError,TypeError,OSError,ImportError):p.error('Invalid input, missing optional dependencies or existing output; see docs/RELIABILITY.md')
if __name__=='__main__':main()
