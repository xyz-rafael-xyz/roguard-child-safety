"""Portable feature and probability computations for a semantic-only probe."""
from collections import Counter
import hashlib
import math
import re
import unicodedata

from roguard.trajectory_data import validate_conversation

DIMENSION = 32768


def vector(payload):
    validate_conversation(payload)
    counts = Counter()
    focus = payload['turns'][-1]
    history = '\n'.join('['+t['role']+'] '+t['text'] for t in payload['turns'][:-1])
    for channel, raw in (('focus:'+focus['role'], focus['text']), ('history', history)):
        value = re.sub(r'\s+', ' ', unicodedata.normalize('NFC',raw).casefold())
        words = re.findall(r'\w+',value)
        tokens = ['language:'+payload['language']]
        tokens.extend('w:'+word for word in words)
        tokens.extend('b:'+a+' '+b for a,b in zip(words,words[1:]))
        tokens.extend('c:'+value[i:i+n] for n in (3,4,5) for i in range(len(value)-n+1))
        for token in tokens:
            key = (channel+'|'+token).encode('utf-8')
            index = int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'big')%DIMENSION
            counts[index] += 1
    values = {k:1+math.log(v) for k,v in counts.items()}
    norm = math.sqrt(sum(v*v for v in values.values()))
    return {k:v/norm for k,v in values.items()}


def probabilities(head, features):
    logits = [bias+sum(weights[i]*value for i,value in features.items())
              for weights,bias in zip(head['weights'],head['bias'])]
    if not logits or any(not math.isfinite(v) for v in logits):
        raise ValueError('Invalid semantic classifier computation')
    if len(logits)==1 and len(head['classes'])==2:
        p = 1/(1+math.exp(-max(-700,min(700,logits[0]))))
        return dict(zip(head['classes'],(1-p,p)))
    exp = [math.exp(v-max(logits)) for v in logits]; total = sum(exp)
    return dict(zip(head['classes'],[v/total for v in exp]))
