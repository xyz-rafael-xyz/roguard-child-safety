"""Training-free NLI interpretation control on the fixed development screen."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from roguard.evidence import ONTOLOGY, disposition
from roguard.nli_v21 import verify_base, REVISION
from roguard.context import render
from roguard.review_evaluation import evaluate
from training.prepare_instruction_sft_v8 import target

HYPOTHESES = {
    'concern': ['The focal message is ordinary content without emotional distress or a request for emotional support.',
                'The focal message concerns emotional distress, safety, loneliness, or a need for emotional support, even if denied or fictional.'],
    'subject': ['The focal message is about the speaker.', 'The focal message is about another real person.', 'The focal message is about a fictional character.'],
    'source': ['The focal message gives a personal direct account.', 'The focal message reports another real person\'s account.', 'The focal message is a fictional example.'],
    'assertion': ['The focal message affirms that the described situation is true.', 'The focal message explicitly denies that the described situation is true.',
                  'The focal message expresses uncertainty whether the described situation is true.', 'The focal message explicitly withdraws or corrects an earlier statement.'],
    'relation': ['The focal message is a new statement, continuation or clarification.', 'The focal message explicitly retracts an earlier statement.'],
    'time': ['The focal message describes the present.', 'The focal message describes the past.', 'The focal message describes a hypothetical situation.'],
}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    verify_base(Path(a.base)); torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, local_files_only=True, trust_remote_code=False,
                                                              use_safetensors=True).float().eval()
    if model.config.id2label.get(0) != 'entailment': raise ValueError('Unexpected NLI class order')
    rows = [json.loads(l) for l in a.data.read_text().splitlines()]
    a.output.mkdir(exist_ok=False); start = time.monotonic(); predictions = []; expected = []
    fields = tuple(HYPOTHESES)
    field_correct = dict.fromkeys(fields, 0)
    with torch.inference_mode(), (a.output/'predictions.jsonl').open('w') as out:
        for i, row in enumerate(rows):
            focus = row['focus']
            premise = 'Previous messages:\n'+render(row['turns'][:focus])+'\nFocal message:\n'+row['turns'][focus]['text']
            hypotheses = [text for f in fields for text in HYPOTHESES[f]]
            values = []
            for begin in range(0, len(hypotheses), 4):
                batch = hypotheses[begin:begin+4]
                tokens = tok([premise]*len(batch), batch, padding=True, truncation=False, return_tensors='pt')
                if tokens['input_ids'].shape[1] > 512: raise ValueError('NLI token limit exceeded')
                values.extend(model(**tokens).logits[:, 0].tolist())
            interpretation = {}; probabilities = {}; cursor = 0
            for f in fields:
                n = len(HYPOTHESES[f]); logits = torch.tensor(values[cursor:cursor+n]); cursor += n
                probs = torch.softmax(logits, dim=0).tolist(); probabilities[f] = probs
                interpretation[f] = ONTOLOGY[f][int(torch.argmax(logits))]
                field_correct[f] += interpretation[f] == row['attributes'][f]
            prediction = int(disposition({k: {'label': v} for k, v in interpretation.items()})[0] != 'CONTINUE')
            record = dict(case_id=row['case_id'], language=row['language'], gold=target(row), predicted=prediction,
                          interpretation=interpretation, relative_entailment_scores=probabilities)
            predictions.append(record); out.write(json.dumps(record)+'\n'); out.flush()
            expected.append(dict(case_id=row['case_id'], language=row['language'], gold=target(row),
                                 kind='conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal'))
            if (i+1) % 25 == 0: print(i+1, 'correct', sum(p['gold'] == p['predicted'] for p in predictions), flush=True)
    metrics = dict(base_revision=REVISION, seconds=time.monotonic()-start, synthetic_only=True,
                   stable_release_ready=False, reserved_probes_used=False, scores_are_calibrated=False,
                   field_accuracy={k:v/len(rows) for k,v in field_correct.items()},
                   strict_evaluation=evaluate(expected, predictions),
                   data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest())
    (a.output/'expected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected))
    (a.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n'); print('FINISHED', json.dumps(metrics), flush=True)


if __name__ == '__main__': main()
