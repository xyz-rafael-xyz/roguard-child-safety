"""Export a completed attention response study as an unqualified local artifact."""
import argparse
import json
from pathlib import Path

from roguard.gemma4_contract import BASE_MODEL, BASE_REVISION
from roguard.gemma4_response import LABELS, PROMPT, QUERY_SHA256
from roguard.gemma4_response_attention import FORMAT
from roguard.trajectory_data import file_sha


def export(study, weights, output):
    study,weights,output=map(Path,(study,weights,output))
    if output.exists():raise ValueError('Refusing to overwrite response attention artifact')
    report=json.loads((study/'report.json').read_text())
    selection=json.loads((study/'selection.json').read_text())
    model=json.loads((study/'model.json').read_text())
    protocol=json.loads((study/'protocol.json').read_text())
    if (report.get('complete') is not True or selection.get('complete') is not True
            or report['selection_sha256']!=file_sha(study/'selection.json')
            or report['model_spec_sha256']!=file_sha(study/'model.json')
            or model['selection_sha256']!=report['selection_sha256']
            or model['protocol_sha256']!=file_sha(study/'protocol.json')
            or report['selected']!=selection['selected']):raise ValueError('Incomplete or changed response attention selection')
    sha=file_sha(weights)
    if (model.get('format')!='roguard-response-attention-research-1'
            or model['base_model']!=BASE_MODEL or model['base_revision']!=BASE_REVISION
            or model['prompt']!=PROMPT or protocol['prompt']!=PROMPT
            or model['labels']!=list(LABELS) or model['boundary']!=24
            or type(model['head'].get('hidden')) is not int or model['head']['hidden']!=0
            or model['qualified'] is not False or model['adapter_sha256']!=sha
            or report['selected']['adapter_sha256']!=sha):raise ValueError('Unsupported response attention model')
    if not 1<=weights.stat().st_size<=16_000_000:raise ValueError('Response attention weights exceed envelope')
    spec={'format':FORMAT,'base_model':BASE_MODEL,'base_revision':BASE_REVISION,'query_sha256':QUERY_SHA256,
          'adapter_sha256':sha,'boundary':24,'hidden':0,'labels':list(LABELS),'qualified':False}
    output.mkdir(parents=True);(output/'adapter.safetensors').write_bytes(weights.read_bytes())
    (output/'response.json').write_text(json.dumps(spec,indent=2)+'\n')
    manifest={'format':'roguard-response-attention-artifact-research-1','source_sha256':file_sha(__file__),
              'selection_sha256':file_sha(study/'selection.json'),'report_sha256':file_sha(study/'report.json'),
              'model_sha256':file_sha(study/'model.json'),
              'artifacts':{name:file_sha(output/name) for name in ('response.json','adapter.safetensors')},
              'numerical_runtime_verification_required':True,'research_only':True,'stable_release_ready':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','weights','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(export(args.study,args.weights,args.output),indent=2))


if __name__=='__main__':main()
