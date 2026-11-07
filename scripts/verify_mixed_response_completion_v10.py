"""Verify study65's complete counters and terminal log before another encoder job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy


def split_completed_log(raw, report):
    """Keep every update event; separately verify the trainer's pretty-printed report."""
    marker = b'\n{\n'
    if raw.count(marker) != 1:
        raise ValueError('Expected one terminal multiline training report')
    offset = raw.index(marker)+1
    prefix, terminal = raw[:offset], raw[offset:]
    if not terminal.endswith(b'\n') or json.loads(terminal) != report or report.get('complete') is not True:
        raise ValueError('Terminal log report differs from completed study')
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--encoder-pid', type=int, required=True)
    parser.add_argument('--paired-verification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite verification')
    try:
        os.kill(a.encoder_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Preceding full-encoder process still exists')
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = a.root/'research/v10-evidence'
    helper = evidence/'mixed-response-epoch1-observation-65/progress-verifier.py'
    if sha(helper) != '1204dd694dbb52ad114be2dab67e49ec35934dd9ac6cde414a121154908efdd8':
        raise ValueError('Registered counter verifier changed')
    functions = runpy.run_path(str(helper))
    study = evidence/'mixed-large-response-attention-65'
    report = json.loads((study/'report.json').read_text())
    protocol = json.loads((study/'protocol.json').read_text())
    progress = json.loads((study/'progress.json').read_text())
    paired = json.loads(a.paired_verification.read_text())
    sequence = json.loads((evidence/'mixed-response-attention-sequence-65/report.json').read_text())
    if (paired.get('verification') != 'PASS' or paired['report_sha256'] != sha(study/'report.json')
            or paired['protocol_sha256'] != functions['PROTOCOL_SHA']
            or sha(study/'protocol.json') != functions['PROTOCOL_SHA']
            or sequence.get('execution_complete') is not True or sequence.get('phase') != 'finished'
            or len(sequence['completed_steps']) != 1 or sequence['completed_steps'][0]['exit_code'] != 0):
        raise ValueError('Completed dependency or paired verification differs')
    selected = functions['audit_history'](progress, protocol)
    if (len(progress['epochs']) != 7 or progress['epochs'][-1]['completed_training_updates'] != 18432
            or progress['epochs'][-1]['completed_training_presentations'] != 147456
            or report['selected'] != selected):
        raise ValueError('Incomplete registered training or changed selection')
    raw = (a.private/'mixed-response-attention-sequence-65.log').read_bytes()
    prefix = split_completed_log(raw, report)
    counters = functions['audit_log'](prefix, progress)
    if counters['latest_logged_update'] != 18425 or counters['logged_updates'] != 737:
        raise ValueError('Incomplete training log')
    result = {'verification': 'PASS', 'source_sha256': sha(Path(__file__)),
              'report_sha256': sha(study/'report.json'), 'progress_sha256': sha(study/'progress.json'),
              'paired_verification_sha256': sha(a.paired_verification),
              'sequence_sha256': sha(evidence/'mixed-response-attention-sequence-65/report.json'),
              'complete_log_sha256': hashlib.sha256(raw).hexdigest(), 'terminal_report_matches': True,
              'encoder_pid_absent': True, 'completed_updates': 18432, 'completed_presentations': 147456,
              'selected_epoch': selected['epoch'], **counters, 'model_qualified': False}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
