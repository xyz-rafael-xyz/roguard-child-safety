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
    print('Installed Romanian and Ukrainian CLIs preserve UTF-8 inputs under legacy output encoding')

if __name__=='__main__':main()
