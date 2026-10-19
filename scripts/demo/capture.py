"""Capture real v9 outputs for a designed walkthrough of fictional examples."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from roguard.entailment_bundle import load_bundle
from roguard.flow import FlowLedger
from roguard.model_install import cache_directory, install, profile

parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=Path, default=cache_directory())
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('facts.json'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
assert version('roguard-child-safety') == '9.0.0'
raw, manifest_sha = profile('v9')
bundle, bundle_sha = install(raw, manifest_sha, args.cache, download=False)
model = load_bundle(bundle, expected_sha256=bundle_sha)
facts = dict(version='9.0.0', fictional_examples=True, visualization_not_screen_recording=True,
             captured_utc=datetime.now(timezone.utc).isoformat(),
             manifest_sha256=manifest_sha, bundle_sha256=bundle_sha)
messages = {
    'ro': ('Mă simt singur și aș vrea să vorbesc cu cineva.',
           'Vrei să îmi spui mai mult?', 'Mă ajuți să înțeleg exercițiul?'),
    'uk': ('Мені самотньо і хочеться з кимось поговорити.',
           'Хочеш розповісти більше?', 'Допоможеш розібратися із завданням?'),
}
for language, (support, reply, task) in messages.items():
    signal = dict(language=language, turns=[dict(role='minor', text=support)])
    shifted = dict(language=language, turns=[dict(role='minor', text=support),
        dict(role='assistant', text=reply), dict(role='minor', text=task)])
    outputs = model.predict_many([signal, shifted], focuses=[0, 2])
    assert len(outputs) == 2 and [r['predicted'] for r in outputs] == [1, 0]
    facts[language] = dict(signal=dict(input=signal, output=outputs[0]),
                          topic_shift=dict(input=shifted, output=outputs[1]))
with FlowLedger() as ledger:
    ledger.add_source('lesson', 'Exercițiu fictiv de adunare')
    ledger.derive('summary', 'Rezumat fictiv', ['lesson'])
    ledger.permission('lesson', 'tutoring', 'teacher', 'export', 'grant')
    before = ledger.check(['summary'], 'tutoring', 'teacher', 'export')
    ledger.permission('lesson', 'tutoring', 'teacher', 'export', 'revoke')
    after = ledger.check(['summary'], 'tutoring', 'teacher', 'export')
facts['revocation'] = dict(before=before, after=after)
study = root/'research/v9-stabilization'
facts['evidence'] = {}
for name, path in {
    'reserved': study/'reported-tail-reserved-199/actual/report.json',
    'installed': study/'reported-tail-installed-198/actual/report.json',
    'publication': study/'stable-release-200/publication.json',
}.items():
    data = path.read_bytes()
    facts['evidence'][name] = dict(path=str(path.relative_to(root)),
        sha256=hashlib.sha256(data).hexdigest(), report=json.loads(data))
assert facts['evidence']['reserved']['report']['counts_pass']
assert facts['evidence']['publication']['report']['stable_release_ready']
args.output.write_text(json.dumps(facts, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({language: [facts[language][kind]['output']['predicted']
    for kind in ('signal', 'topic_shift')] for language in messages}))
