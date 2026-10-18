"""Strict counts for reserved synthetic probes, with abstentions as errors."""
import hashlib
import json


def counts(records):
    if not records:
        raise ValueError('Empty evaluation population')
    for row in records:
        if type(row['gold']) is not int or row['gold'] not in (0, 1):
            raise ValueError('Invalid gold label')
        if row['predicted'] is not None and (type(row['predicted']) is not int or row['predicted'] not in (0, 1)):
            raise ValueError('Invalid prediction')
    positive = sum(r['gold'] for r in records)
    negative = len(records)-positive
    missed = sum(r['gold'] == 1 and r['predicted'] != 1 for r in records)
    false = sum(r['gold'] == 0 and r['predicted'] != 0 for r in records)
    return dict(rows=len(records), positive=positive, negative=negative, missed=missed,
                false_reviews=false, abstentions=sum(r['predicted'] is None for r in records),
                recall=1-missed/positive if positive else None,
                false_review_rate=false/negative if negative else None,
                passes=bool(positive and negative and missed*100 <= positive and false*20 <= negative))


def input_key(row):
    """Exact semantic input identity; normalization variants remain distinct."""
    focus = row.get('focus', len(row['turns'])-1)
    value = dict(language=row['language'], turns=row['turns'][:focus+1])
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def unique_records(records):
    unique = {}
    for record in records:
        key = record['input_digest']
        if key in unique:
            old = unique[key]
            if (old['gold'], old['predicted']) != (record['gold'], record['predicted']):
                raise ValueError('Repeated input has inconsistent labels or runtime decisions')
        else:
            unique[key] = record
    return list(unique.values())
