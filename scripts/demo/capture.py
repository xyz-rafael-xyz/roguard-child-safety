"""Capture fictional, reproducible v6 outputs used by the visual walkthrough."""
import argparse
import json
from pathlib import Path
from roguard.conversation import ConversationScorer, analyze_conversation
from roguard.flow import FlowLedger
from roguard.ops import repair_demo

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=Path, required=True)
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
scorer = ConversationScorer('adapted', args.model_path, args.device)
result = {'version': '6.0.0', 'fictional_examples': True, 'backend': 'adapted'}
for language in ('ro', 'uk'):
    payload = json.loads((root / 'examples' / ('conversation_' + language + '.json')).read_text())
    result[language] = {'input': payload, 'output': analyze_conversation(payload, scorer)}
result['repair'] = repair_demo('ro')
with FlowLedger() as ledger:
    ledger.add_source('lesson', 'Exercițiu fictiv de adunare')
    ledger.derive('summary', 'Rezumat fictiv', ['lesson'])
    ledger.permission('lesson', 'tutoring', 'teacher', 'export', 'grant')
    before = ledger.check(['summary'], 'tutoring', 'teacher', 'export')
    ledger.permission('lesson', 'tutoring', 'teacher', 'export', 'revoke')
    after = ledger.check(['summary'], 'tutoring', 'teacher', 'export')
result['revocation'] = {'before': before, 'after': after}
(root / 'scripts/demo/facts.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({language: result[language]['output']['signal']['decision'] for language in ('ro', 'uk')}))
