"""Exercise installed bilingual CLIs under a legacy Windows text encoding."""
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

from roguard.context import ContextModel, analyze_context
from roguard.evidence import EvidenceModel, analyze_evidence
import roguard


def main():
    root=Path(os.environ['GITHUB_WORKSPACE']).resolve()
    package=Path(roguard.__file__).resolve()
    if root in package.parents:
        raise AssertionError('Expected installed package outside the checkout')
    executable_dir=Path(sysconfig.get_path('scripts'))
    suffix='.exe' if os.name=='nt' else ''
    environment=dict(os.environ,PYTHONUTF8='0',PYTHONIOENCODING='cp1252',PYTHONCOERCECLOCALE='0')
    with tempfile.TemporaryDirectory() as directory:
        for language,text in [('ro','Îmi este teamă și aș vrea să vorbesc cu cineva.'),
                              ('uk','Мій друг почувається самотнім і хоче поговорити.')]:
            payload={'language':language,'turns':[{'role':'minor','text':text}]}
            path=Path(directory)/(language+'.json')
            path.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
            for command,reference in [('roguard-evidence',analyze_evidence(payload,EvidenceModel())),
                                      ('roguard-context',analyze_context(payload,ContextModel('character')))]:
                args=[str(executable_dir/(command+suffix)),str(path)]
                if command=='roguard-context':args+=['--backend','character']
                result=subprocess.run(args,env=environment,capture_output=True,check=True)
                actual=json.loads(result.stdout)
                if actual!=reference:
                    raise AssertionError('Installed CLI differs from Unicode API input: '+command+'/'+language)
            assurance=subprocess.run([str(executable_dir/('roguard-assure'+suffix)),str(path),'--conversation'],
                                     env=environment,capture_output=True,check=True)
            report=json.loads(assurance.stdout)
            if not report['entries'] or report['permits_external_action']:
                raise AssertionError('Invalid installed assurance response')
            from roguard.assurance import verify
            for entry in report['entries']:
                verify(entry['certificate'])
            certificate_path=Path(directory)/(language+'-certificate.json')
            certificate_path.write_text(json.dumps(report['entries'][0]['certificate']),encoding='utf-8')
            subprocess.run([str(executable_dir/('roguard-assure'+suffix)),str(certificate_path),'--verify'],
                           env=environment,capture_output=True,check=True)
        expected=[];predictions=[]
        for language in ('ro','uk'):
            for kind in ('formal','conversational'):
                for gold in (0,1):
                    case_id='ș-ї-'+language+'-'+kind+'-'+str(gold)
                    expected.append(dict(case_id=case_id,language=language,kind=kind,gold=gold))
                    predictions.append(dict(case_id=case_id,language=language,predicted=gold))
        expected_path=Path(directory)/'expected.jsonl'
        prediction_path=Path(directory)/'predictions.jsonl'
        report_path=Path(directory)/'report.json'
        for path,rows in ((expected_path,expected),(prediction_path,predictions)):
            path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
        command=[str(executable_dir/('roguard-evaluate-review'+suffix)),
                 '--expected',str(expected_path),'--predictions',str(prediction_path),
                 '--output',str(report_path),'--require-targets']
        subprocess.run(command,env=environment,capture_output=True,check=True)
        predictions[0]['predicted']=None
        prediction_path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in predictions),encoding='utf-8')
        failed=subprocess.run(command,env=environment,capture_output=True)
        if failed.returncode!=1:
            raise AssertionError('Installed evaluator did not reject an unresolved benign case')
    print('Installed Romanian and Ukrainian CLIs preserve UTF-8 inputs under legacy output encoding')

if __name__=='__main__':main()
