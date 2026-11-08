"""Audit continuation97 with the frozen independent screen82 recount helpers."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl
from eval.verify_response_diversity_screen_v10 import (SCOPE, require, equal, ordered_rows,
    replay_tokens, recount_raw, check_sequence, verify_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private', 'base', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--queue-plan-sha256', required=True)
    parser.add_argument('--queue-pid', type=int, required=True)
    a = parser.parse_args()
    require(not a.output.exists(), 'Refusing to overwrite verification')
    evidence = a.root/'research/v10-evidence'
    plan_path = evidence/'protocol-amendment-78.json'
    queue_path = a.private/'response-diversity-screen-sequence-97-plan.json'
    require(file_sha(plan_path) == a.plan_sha256 and file_sha(queue_path) == a.queue_plan_sha256, 'Frozen plan changed')
    plan = json.loads(plan_path.read_text()); queue = json.loads(queue_path.read_text())
    sequence_path = evidence/'response-diversity-screen-sequence-97/report.json'
    study = evidence/'response-diversity-screen-97'
    raw_path = a.private/'response-diversity-screen-97/judgments.jsonl'
    output_paths = [sequence_path, study/'protocol.json', study/'report.json',
                    study/'progress.json', study/'source.py', raw_path]
    # Capture before parsing; reject replacement even when its new contents are valid.
    sequence_sha = file_sha(sequence_path)
    sequence = json.loads(sequence_path.read_text())
    check_sequence(sequence, queue, a.queue_plan_sha256)
    require(sequence['source_sha256'] == file_sha(a.root/'scripts/run_report_sequence_v10.py')
            and sequence['dependency_report_sha256'] == file_sha(a.root/plan['dependency_report'])
            and sequence['stable_release_ready'] is False and sequence['promotion_performed'] is False,
            'Sequence provenance or scope differs')
    pid = int((a.private/'response-diversity-screen-sequence-97.pid').read_text())
    require(type(a.queue_pid) is int and a.queue_pid > 0 and pid == a.queue_pid, 'Registered queue PID differs')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Screen queue has not exited')
    output_hashes = {path: file_sha(path) for path in output_paths}
    output_hashes[sequence_path] = sequence_sha
    for root, bindings in ((a.root, plan['public']), (a.private, plan['private']),
                           (a.root, queue['source_hashes'])):
        for name, expected in bindings.items():
            require(file_sha(root/name) == expected, 'Bound input changed: '+name)
    protocol = json.loads((study/'protocol.json').read_text())
    report = json.loads((study/'report.json').read_text())
    progress = json.loads((study/'progress.json').read_text())
    source_sha = plan['public']['training/screen_response_diversity_v10.py']
    require(file_sha(study/'source.py') == source_sha and protocol['source_sha256'] == source_sha
            and protocol['plan_sha256'] == a.plan_sha256, 'Protocol source binding')
    for key in ('judge', 'model_files', 'software', 'public', 'private', 'task_prompt_sha256'):
        equal(protocol[key], plan[key], 'Protocol differs: '+key)
    require(protocol['format'] == 'roguard-v10-single-response-screen-1'
            and digest(protocol['screen_prompt']) == plan['screen_prompt_sha256'], 'Screen prompt differs')
    # Tokenizer-only replay: never import/load a model or regenerate judgments.
    require(a.base.name == plan['judge']['revision'], 'Tokenizer revision differs')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json'):
        require(file_sha(a.base/name) == plan['model_files'][name], 'Tokenizer file changed: '+name)
    for name in ('transformers', 'tokenizers'):
        require(version(name) == plan['software'][name], 'Tokenizer software differs: '+name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.base), local_files_only=True, trust_remote_code=False)
    rows_path = a.private/'response-diversity-pilot-69/train-draft.jsonl'
    rows = ordered_rows(read_jsonl(rows_path))
    tokens = replay_tokens(rows, protocol['screen_prompt'], tokenizer)
    for key, value in {'rows': 128, 'ordered_inputs_sha256': digest(rows), 'query_tokens_sha256': digest(tokens),
                       'minimum_query_tokens': min(map(len, tokens.values())),
                       'maximum_query_tokens': max(map(len, tokens.values()))}.items():
        equal(protocol[key], value, 'Replayed inputs differ: '+key)
    require(report['judgments_sha256'] == file_sha(raw_path)
            and report['protocol_sha256'] == file_sha(study/'protocol.json'), 'Output file binding differs')
    counts = recount_raw(rows, read_jsonl(raw_path), tokens)
    verify_report(report, protocol, progress, counts)
    proof = {'verification': 'PASS', 'source_sha256': file_sha(__file__), 'plan_sha256': a.plan_sha256,
             'queue_plan_sha256': a.queue_plan_sha256, 'sequence_sha256': file_sha(sequence_path),
             'report_sha256': file_sha(study/'report.json'), 'protocol_sha256': file_sha(study/'protocol.json'),
             'progress_sha256': file_sha(study/'progress.json'), 'judgments_sha256': file_sha(raw_path),
             'drafts_sha256': file_sha(rows_path), 'query_tokens_sha256': digest(tokens), 'recount': counts,
             'model_loaded': False, 'generations_repeated': False, 'qualified': False, **SCOPE,
             'limitations': 'Independent software recount and tokenizer replay only. Raw generations are retained '
                            'records, not repeated inference. Machine agreement does not verify language or labels. '
                            'A rejected screen can have a passing integrity audit. No fitting admission.'}
    for root, bindings in ((a.root, plan['public']), (a.private, plan['private']), (a.root, queue['source_hashes'])):
        for name, expected in bindings.items():
            require(file_sha(root/name) == expected, 'Bound input changed during audit: '+name)
    require(file_sha(plan_path) == a.plan_sha256 and file_sha(queue_path) == a.queue_plan_sha256,
            'Plan changed during audit')
    for path, expected in output_hashes.items():
        require(file_sha(path) == expected, 'Output changed during audit: '+path.name)
    with a.output.open('x') as output:
        output.write(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
