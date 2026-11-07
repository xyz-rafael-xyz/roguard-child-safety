"""Export a completed frozen readout as an explicitly unqualified local artifact."""
import argparse
import json
from pathlib import Path

from roguard.gemma4_contract import BASE_MODEL, BASE_REVISION
from roguard.gemma4_response import FORMAT, LABELS, QUERY_SHA256
from roguard.trajectory_data import file_sha


def export(study, weights, output):
    study,weights,output=map(Path,(study,weights,output))
    if output.exists():raise ValueError('Refusing to overwrite response artifact')
    report=json.loads((study/'report.json').read_text())
    selection=json.loads((study/'selection.json').read_text())
    if (report.get('complete') is not True or selection.get('complete') is not True
            or report['selection_sha256']!=file_sha(study/'selection.json')
            or report['selected']!=selection['selected']):raise ValueError('Incomplete or changed response selection')
    selected=report['selected'];configuration=selected['configuration'];head_sha=file_sha(weights)
    if selected.get('head_sha256')!=head_sha:raise ValueError('Selected response weights differ')
    if type(configuration.get('hidden')) is not int or configuration['hidden'] not in (0,128):raise ValueError('Unsupported response head')
    if not 1<=weights.stat().st_size<=2_000_000:raise ValueError('Response head exceeds artifact envelope')
    spec={'format':FORMAT,'base_model':BASE_MODEL,'base_revision':BASE_REVISION,'query_sha256':QUERY_SHA256,
          'head_sha256':head_sha,'hidden':configuration['hidden'],'labels':list(LABELS),'qualified':False}
    output.mkdir(parents=True);(output/'head.safetensors').write_bytes(weights.read_bytes())
    (output/'response.json').write_text(json.dumps(spec,indent=2)+'\n')
    manifest={'format':'roguard-response-artifact-research-1','source_sha256':file_sha(__file__),
              'selection_sha256':file_sha(study/'selection.json'),'report_sha256':file_sha(study/'report.json'),
              'artifacts':{name:file_sha(output/name) for name in ('response.json','head.safetensors')},
              'numerical_runtime_verification_required':True,'research_only':True,'stable_release_ready':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','weights','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.study,a.weights,a.output),indent=2))


if __name__=='__main__':main()
