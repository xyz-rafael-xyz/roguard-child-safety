"""Partition checks using the exact normalized inputs seen by the review encoder."""
import json
from roguard.context import render
from roguard.evidence import disposition
from roguard.semantic import normalize_semantic_text


def input_key(row):
    return json.dumps([row['language'],
                      normalize_semantic_text(render(row['turns'][:row['focus']+1])),
                      normalize_semantic_text(row['turns'][row['focus']]['text'])],
                     ensure_ascii=False)


def isolate_partitions(partitions, prior_training=()):
    """Keep earlier-partition ownership; reject contradictory binary targets."""
    history={}
    for row in prior_training:
        key=input_key(row)
        target=disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE'
        if key in history and history[key]!=target:
            raise ValueError('Conflicting historical review labels')
        history[key]=target
    seen = {}; cleaned = {}; removed = {}
    for split in ('train', 'dev', 'cal'):
        cleaned[split] = []; removed[split] = []
        for row in partitions[split]:
            key = input_key(row)
            target = disposition({k: {'label': v} for k, v in row['attributes'].items()})[0] != 'CONTINUE'
            if key in history and history[key]!=target:
                raise ValueError('Conflicting review label against training history')
            if key in seen or (split!='train' and key in history):
                if key in seen and seen[key] != target:
                    raise ValueError('Conflicting review labels for identical normalized input')
                removed[split].append(row['case_id'])
                continue
            seen[key] = target
            cleaned[split].append(row)
    return cleaned, removed
