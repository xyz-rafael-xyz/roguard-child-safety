"""Trainable context/focus pooling, exactly mean pooling at zero queries."""
import math


def attention_pool(hidden, mask, queries):
    import torch
    if hidden.ndim!=3 or mask.shape!=hidden.shape[:2] or queries.shape!=(2,hidden.shape[-1]) or hidden.shape[0]%2:
        raise ValueError('Expected concatenated context and focus batches')
    if not torch.all(mask.sum(1)>0):
        raise ValueError('Empty sequence')
    query=queries.repeat_interleave(hidden.shape[0]//2,dim=0)
    logits=(hidden*query[:,None,:]).sum(-1)/math.sqrt(hidden.shape[-1])
    weights=torch.softmax(logits.masked_fill(mask==0,float('-inf')),dim=1)
    pooled=(hidden*weights[:,:,None]).sum(1)
    return torch.nn.functional.normalize(pooled,dim=1)*10
