"""Standard-library research response readout; not a safety certificate."""
import math

from roguard.trajectory_model import DIMENSION, features, head_predict


def response_features(draft, language, context, representation):
    if representation == 'draft-only':
        return features('\n[DRAFT] '+draft, language)
    if representation != 'separate-context-draft':
        raise ValueError('Unknown response representation')
    result = {}
    for offset, text in ((0, draft), (DIMENSION//2, context)):
        for key, value in features(text, language).items():
            index = key % (DIMENSION//2)+offset
            result[index] = result.get(index, 0)+value
    norm = math.sqrt(sum(v*v for v in result.values()))
    return {k: v/norm for k, v in result.items()} if norm else result


def predict(spec, row):
    if spec['format'] != 'roguard-response-readout-research-1':
        raise ValueError('Wrong response artifact')
    probabilities = head_predict(spec['head'], response_features(
        row['draft'], row['language'], row['context'], spec['representation']))
    return {'label': max(probabilities, key=probabilities.get), 'probabilities': probabilities,
            'qualified': False, 'safe_response_established': False,
            'contextual_factual_support_established': False, 'permits_external_action': False}
