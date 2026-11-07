"""Training-only hidden dropout with the unchanged deployed semantic-head layout."""
import math


def make_head(tensors, probability):
    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability < 1:
        raise ValueError('Invalid hidden-dropout probability')
    import mlx.core as mx
    import mlx.nn as nn
    from training.native_fields_v9 import SIZES

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            hidden, dimensions = tensors['0.weight'].shape
            self.first = nn.Linear(dimensions, hidden)
            self.last = nn.Linear(hidden, sum(SIZES)+2)
            self.dropout = nn.Dropout(p=probability)
            self.first.weight, self.first.bias = mx.array(tensors['0.weight']), mx.array(tensors['0.bias'])
            self.last.weight, self.last.bias = mx.array(tensors['3.weight']), mx.array(tensors['3.bias'])

        def __call__(self, features):
            normalized = features/mx.sqrt(mx.sum(mx.square(features), axis=1, keepdims=True))*features.shape[1]**.5
            return self.last(self.dropout(nn.gelu(self.first(normalized))))
    return Head()
