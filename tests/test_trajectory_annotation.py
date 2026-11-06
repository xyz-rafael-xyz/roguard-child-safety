from copy import deepcopy
import json
from importlib.resources import files
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from roguard.evidence import ONTOLOGY
from roguard.trajectory_annotation import render, validate_packet
from roguard.trajectory_data import FORMAT
from roguard.trajectory_review import annotation_packet, validate_annotations


def fixture():
    case = {'format': FORMAT, 'case_id': 'case', 'family': 'family', 'author': 'practice',
            'generator': 'practice', 'synthetic': True, 'split': 'dev', 'language': 'ro',
            'turns': [{'role': 'minor', 'text': 'A😀é __ONTOLOGY_JSON__ </script><script>bad()</script>'}],
            'checkpoints': [{'focus': 0, 'review': 0, 'world': None, 'evidence': []}]}
    return annotation_packet([case], annotator_id='reviewer')


def test_html_preserves_unicode_and_template_tokens_without_script_injection():
    packet, _ = fixture(); html = render(packet)
    embedded = re.search(r'<script id="packet-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(embedded) == packet
    assert '<script>bad()' not in html
    assert 'connect-src \'none\'' in html
    assert '<script src=' not in html


@pytest.mark.parametrize('mutation', ['model', 'key', 'label', 'digest', 'duplicate', 'empty'])
def test_form_rejects_unblinded_or_changed_packet(mutation):
    packet, _ = fixture()
    if mutation == 'model': packet['items'][0]['model_output'] = 1
    elif mutation == 'key': packet['key'] = {}
    elif mutation == 'label': packet['items'][0]['review'] = 0
    elif mutation == 'digest': packet['items'][0]['conversation']['turns'][0]['text'] += 'changed'
    elif mutation == 'duplicate': packet['items'].append(deepcopy(packet['items'][0]))
    elif mutation == 'empty': packet['items'] = []
    with pytest.raises(ValueError): validate_packet(packet)


def run_js(code, data):
    node = shutil.which('node')
    if node is None: pytest.skip('Node is required for annotation JavaScript checks')
    script = files('roguard').joinpath('trajectory-annotation.js')
    program = "const core=require(process.argv[1]); const assert=require('node:assert/strict'); const data=JSON.parse(require('node:fs').readFileSync(0,'utf8'));\n"+code
    return subprocess.run([node, '-e', program, str(script)], input=json.dumps(data), text=True, encoding='utf-8', capture_output=True, check=True).stdout


def test_javascript_completed_export_validates_with_python_coordinator():
    packet, key = fixture()
    completed = json.loads(run_js("""
const p=structuredClone(data.packet), row=p.items[0];
row.review=0;row.uncertain=false;row.outside_ontology=false;
row.world=Object.fromEntries(Object.entries(data.ontology).map(([k,v])=>[k,v[0]]));
row.evidence=[core.spanFromOffsets(row.conversation.turns[0].text,0,1,3)];
assert.deepEqual(row.evidence,[{turn:0,start:1,end:2}]);
assert.deepEqual(core.annotationIssues(row,data.ontology,true),[]);
console.log(JSON.stringify(core.resumePacket(data.packet,p,data.ontology)));
""", {'packet': packet, 'ontology': ONTOLOGY}))
    labels = validate_annotations(packet, key, completed)
    assert next(iter(labels.values()))['evidence'] == [{'turn': 0, 'start': 1, 'end': 2}]


def test_javascript_rejects_changed_inputs_and_incomplete_final_but_resumes_partial_draft():
    packet, _ = fixture()
    run_js("""
const p=structuredClone(data.packet), row=p.items[0];
assert.ok(core.annotationIssues(row,data.ontology,true).length);
row.world=Object.fromEntries(Object.keys(data.ontology).map(k=>[k,'']));row.world.subject='SELF';
assert.deepEqual(core.resumePacket(data.packet,p,data.ontology),p);
assert.ok(core.annotationIssues(row,data.ontology,true).length);
const changed=structuredClone(p);changed.items[0].conversation.turns[0].text+=' changed';
assert.throws(()=>core.resumePacket(data.packet,changed,data.ontology));
assert.throws(()=>core.spanFromOffsets(row.conversation.turns[0].text,0,1,2));
row.world=null;row.review=null;row.uncertain=true;row.outside_ontology=false;
assert.deepEqual(core.annotationIssues(row,data.ontology,true),[]);
row.evidence=[{turn:1,start:0,end:1}];assert.ok(core.annotationIssues(row,data.ontology,true).length);
""", {'packet': packet, 'ontology': ONTOLOGY})
