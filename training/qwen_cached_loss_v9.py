"""Reuse only the immutable frozen prefix below the trainable final block."""
from copy import deepcopy


def make_loss(model, sequences):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask,create_ssm_mask
    from mlx_lm.models.cache import make_prompt_cache
    if not sequences or any(len(s)<2 for s in sequences):
        raise ValueError('Expected nonempty aligned training sequences')
    common=0
    for values in zip(*sequences):
        if len(set(values))!=1:break
        common+=1
    common=min(common,min(map(len,sequences))-2)
    if common<1:raise ValueError('No common frozen prefix')
    if any(s[:common]!=sequences[0][:common] for s in sequences):
        raise ValueError('Frozen prefix differs')
    text=model.language_model.model
    if model.layers[-1].is_linear:raise ValueError('Final layer must use full attention')
    model.eval()
    cache=make_prompt_cache(model)
    hidden=text.embed_tokens(mx.array([sequences[0][:common]]))
    fa=create_attention_mask(hidden,cache[text.fa_idx])
    ssm=create_ssm_mask(hidden,cache[text.ssm_idx])
    for layer,state in zip(model.layers[:-1],cache[:-1]):
        hidden=layer(hidden,mask=ssm if layer.is_linear else fa,cache=state)
    prefix_hidden=mx.stop_gradient(hidden);mx.eval(prefix_hidden)

    def loss(model,batch,lengths):
        if batch.shape[0]!=1:raise ValueError('Frozen-prefix training requires batch1')
        for layer in model.layers[:-1]:layer.eval()
        state=deepcopy(cache)
        hidden=text.embed_tokens(batch[:,common:-1])
        fa=create_attention_mask(hidden,state[text.fa_idx])
        ssm=create_ssm_mask(hidden,state[text.ssm_idx])
        for layer,c in zip(model.layers[:-1],state[:-1]):
            hidden=layer(hidden,mask=ssm if layer.is_linear else fa,cache=c)
        hidden=mx.stop_gradient(mx.concatenate([prefix_hidden,hidden],axis=1))
        hidden=model.layers[-1](hidden,mask=create_attention_mask(hidden))
        hidden=text.norm(hidden)
        positions=lengths[:,0]-1;indices=mx.arange(batch.shape[0])
        logits=model.language_model.lm_head(hidden[indices,positions,:][:,None,:])[:,0,:].astype(mx.float32)
        labels=batch[indices,positions+1]
        return nn.losses.cross_entropy(logits,labels,reduction='mean'),mx.array(batch.shape[0])
    return loss,common
