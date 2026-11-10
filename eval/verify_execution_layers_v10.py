"""Recount retained layer probes without loading an encoder."""
import argparse
import json
from pathlib import Path

from roguard.trajectory_data import file_sha
from eval.diagnose_review_execution_layers_v10 import selected_cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'private', 'prior_study', 'prior_private', 'output'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite verification')
    report = json.loads((a.study/'report.json').read_text())
    protocol = json.loads((a.study/'protocol.json').read_text())
    if (report.get('complete') is not True or report.get('optimizer_updates') != 0
            or report.get('qualified') is not False
            or report.get('decision') != 'DIAGNOSTIC_ONLY_NO_MODEL_QUALIFICATION'
            or set(report['languages']) != {'ro', 'uk'}
            or file_sha(a.study/'source.py') != protocol['source_sha256']
            or file_sha(a.prior_study/'report.json') != protocol['prior_report_sha256']
            or file_sha(a.prior_study/'protocol.json') != protocol['prior_protocol_sha256']):
        raise ValueError('Incomplete or changed diagnostic')
    prior = json.loads((a.prior_study/'report.json').read_text())
    import numpy as np

    def error(left, right):
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('Invalid probe arrays')
        return float(np.max(np.abs(left-right)))

    languages = {}; artifacts = {}
    for lang, records in report['languages'].items():
        for name, expected in prior['languages'][lang]['artifacts'].items():
            if Path(name).name != name or file_sha(a.prior_private/name) != expected:
                raise ValueError('Prior observation artifact differs')
        observations = [json.loads(line) for line in
                        (a.prior_private/(lang+'-observations.jsonl')).read_text().splitlines()]
        selected = selected_cases(observations)
        if ([r['feature_index'] for r in records] != protocol['cases'][lang]
                or [r['prior_observation'] for r in records] != selected):
            raise ValueError('Selected case inventory differs')
        max_replay = max_instrumentation = max_boundary = max_native = 0.
        for r in records:
            path = a.private/(lang+'-'+str(r['feature_index'])+'-probes.npz')
            if file_sha(path) != r['artifact_sha256']:
                raise ValueError('Probe artifact differs')
            artifacts[path.name] = file_sha(path)
            with np.load(path, allow_pickle=False) as arrays:
                native = error(arrays['prefix'], arrays['full'])
                replay = {name: error(arrays[name], arrays[name+'_replay']) for name in ('prefix', 'full')}
                instrumentation = {name: error(arrays[name], arrays[name+'_traced'])
                    for name in ('prefix', 'full', 'prefix_replay', 'full_replay')}
                layers = {key: error(arrays['full_'+key], arrays['prefix_'+key])
                          for key in r['layer_errors']}
                tail = {name: {key: error(arrays[name+'_'+key], arrays[name+'_replay_'+key])
                    for key in r['native_replay_layer_errors'][name]} for name in ('prefix', 'full')}
                if (native != r['native_cached_vs_uncached'] or replay != r['replay_errors']
                        or instrumentation != r['instrumentation_errors'] or layers != r['layer_errors']
                        or tail != r['native_replay_layer_errors']):
                    raise ValueError('Reported probe metrics differ')
                for name in ('prefix', 'full'):
                    expected = {str(i)+'_'+suffix for i in range(42)
                        for suffix in ('hidden', 'key_last', 'value_last', 'key_prefix_last', 'value_prefix_last')}
                    if set(layers) != expected:
                        raise ValueError('Native layer coverage differs')
                    if set(tail[name]) != {k for k in expected if int(k.split('_')[0]) >= 24}:
                        raise ValueError('Tail layer coverage differs')
            max_replay = max(max_replay, *replay.values())
            max_native = max(max_native, native)
            max_instrumentation = max(max_instrumentation, *instrumentation.values())
            # Capture-state arrays were not retained; this is a reported observation.
            max_boundary = max(max_boundary, *r['capture_boundary_errors'].values())
        languages[lang] = dict(cases=len(records), maximum_replay_error=max_replay,
            maximum_native_cached_uncached_error=max_native,
            maximum_instrumentation_error=max_instrumentation,
            reported_maximum_capture_boundary_error=max_boundary)
    proof = dict(verification='PASS', source_sha256=file_sha(__file__),
        report_sha256=file_sha(a.study/'report.json'), protocol_sha256=file_sha(a.study/'protocol.json'),
        languages=languages, artifacts=artifacts, encoder_loaded=False, training_performed=False,
        qualified=False, scope='Recomputed native/replay, cache-path and instrumentation differences from all retained probes; exact case selection and artifact hashes. Capture boundary and historical-reference comparisons are reported observations, not independently replayed here.')
    a.output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
