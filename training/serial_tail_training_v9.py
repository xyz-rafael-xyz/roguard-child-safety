"""Keep attention backward calls single-example while preserving joint losses.

The local Metal backward path produced nondeterministic gradients for equal-
length multi-example groups. Development inference retains the original grouped
forward path. Minibatch weighting, paired losses and history KL are unchanged.
"""
from collections import defaultdict
from training.shared_tail_training_v9 import make_head


def make_classifier(base, tensors, *, boundary, lora):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from training.gemma4_shared_tail_v9 import _backbone, replay
    from training.shared_tail_cache_v9 import batch_states
    _backbone(base, boundary)
    base.freeze()
    linear_to_lora_layers(base, len(base.layers)-boundary, lora)
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.heads = {language: make_head(head) for language, head in tensors.items()}
        def __call__(self, states, languages):
            if len(states) != len(languages) or not states:
                raise ValueError('Mismatched cached classifier inputs')
            grouped = defaultdict(list)
            for i, (state, language) in enumerate(zip(states, languages)):
                if language not in self.heads or state['start'] != boundary:
                    raise ValueError('Unexpected language or tail boundary')
                grouped[(language, state['length'], i if self.training else None)].append(i)
            output = [None]*len(states)
            for (language, _, _), indices in grouped.items():
                hidden = replay(self.base, batch_states([states[i] for i in indices]))[:, 0, :]
                values = self.heads[language](hidden)
                for position, index in enumerate(indices):
                    output[index] = values[position]
            return mx.stack(output)
    return Classifier()

