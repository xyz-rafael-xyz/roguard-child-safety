"""Require complete decision and field-label parity before batched evaluation."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from roguard.evidence import ONTOLOGY


def compare(reference, candidate):
    if not reference or [r['case_id'] for r in reference]!=[r['case_id'] for r in candidate]:
        raise ValueError('Decoder case coverage/order differs')
    if len({r['case_id'] for r in reference})!=len(reference): raise ValueError('Duplicate cases')
    differences=dict(decisions=0,interpretations=0,validity=0);maximum=0.;abstentions=0
    for old,new in zip(reference,candidate):
        if old['language']!=new['language'] or old['gold']!=new['gold']: raise ValueError('Decoder labels differ')
        differences['decisions']+=old['predicted']!=new['predicted']
        differences['interpretations']+=old['interpretation']!=new['interpretation']
        differences['validity']+=old['valid_label_tokens']!=new['valid_label_tokens']
        abstentions+=(old['valid_label_tokens'] is not True or new['valid_label_tokens'] is not True
                      or type(old['predicted']) is not int or old['predicted'] not in (0,1)
                      or type(new['predicted']) is not int or new['predicted'] not in (0,1))
        for row in (old,new):
            if set(row['conditional_field_probabilities'])!=set(ONTOLOGY): raise ValueError('Missing field probabilities')
            for field,labels in ONTOLOGY.items():
                values=row['conditional_field_probabilities'][field]
                if (len(values)!=len(labels) or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values)
                        or abs(sum(values)-1)>1e-4): raise ValueError('Invalid conditional probabilities')
        maximum=max(maximum,max(abs(x-y) for field in ONTOLOGY for x,y in zip(
            old['conditional_field_probabilities'][field],new['conditional_field_probabilities'][field])))
    return dict(rows=len(reference),differences=differences,abstentions=abstentions,maximum_probability_error=maximum,
                parity_pass=not any(differences.values()) and not abstentions and maximum<=.001)


def choose_decoder(selected, base, deadline):
    root=Path('research/v9-stabilization');study=root/'interpreter-batch-64'
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def read(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
    for name,expected in json.loads((study/'source-freeze.json').read_text()).items():
        if sha(name)!=expected:raise ValueError('Frozen batch comparison source changed')
    report=dict(selected_backend='single',reserved_probes_used=False,stable_release_ready=False,
        cached_single_reference=True,controlled_latency_benchmark=False,
        selected_adapter_sha256=sha(selected/'selected-adapter/adapters.safetensors'))
    if deadline-time.time()<360:
        report['skipped']='Insufficient time for bounded decoder comparison'
    else:
        data=Path('research/v8-stabilization/instruction-sft-01/screen.jsonl')
        command=[sys.executable,'-m','training.evaluate_interpreter_batch_v9','--base',base,
            '--adapter',str(selected/'selected-adapter'),'--data',str(data),'--output',str(study/'screen')]
        log=Path(tempfile.gettempdir())/'roguard-v9-window/interpreter-batch-64.log'
        print('START batched decoder parity screen',flush=True)
        with log.open('w') as stream:
            try:report['exit_code']=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=360).returncode
            except subprocess.TimeoutExpired:report.update(exit_code=-1,timed_out=True)
        if report['exit_code']==0:
            baseline=selected/'adapted';batch=study/'screen'
            a=json.loads((baseline/'metrics.json').read_text());b=json.loads((batch/'metrics.json').read_text())
            if (a['data_sha256']!=sha(data) or b['data_sha256']!=sha(data)
                    or b['adapter_sha256']!=report['selected_adapter_sha256'] or a['base_revision']!=b['base_revision']):
                raise ValueError('Decoder comparison identities differ')
            report.update(compare(read(baseline/'predictions.jsonl'),read(batch/'predictions.jsonl')))
            if report['rows']!=400:raise ValueError('Expected complete400-case parity screen')
            for metric in (a,b):
                if not isinstance(metric['seconds'],(int,float)) or not math.isfinite(metric['seconds']) or metric['seconds']<=0:
                    raise ValueError('Invalid decoder timing')
            report.update(reference_seconds=a['seconds'],batch_seconds=b['seconds'],
                reference_predictions_sha256=sha(baseline/'predictions.jsonl'),batch_predictions_sha256=sha(batch/'predictions.jsonl'),
                reference_metrics_sha256=sha(baseline/'metrics.json'),batch_metrics_sha256=sha(batch/'metrics.json'))
            if report['parity_pass'] and b['seconds']<a['seconds']:report['selected_backend']='batch4'
    (study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Decoder comparison',json.dumps(report),flush=True)
    return 'training.evaluate_interpreter_batch_v9' if report['selected_backend']=='batch4' else 'training.evaluate_interpreter_v9'
