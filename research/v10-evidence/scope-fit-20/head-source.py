"""Research classifier over independently normalized frozen representation views."""


def make_head(configuration):
    import mlx.core as mx
    import mlx.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            dimensions = 2560 if configuration['view'] == 'base' else 5120
            hidden = configuration['hidden']
            if hidden:
                self.first = nn.Linear(dimensions, hidden)
                self.last = nn.Linear(hidden, 19)
            else:
                self.last = nn.Linear(dimensions, 19)

        def __call__(self, features):
            if features.ndim != 2 or features.shape[1] != 5120:
                raise ValueError('Expected paired 2560-dimensional representations')
            views = [features[:, 2560:]] if configuration['view'] == 'base' else mx.split(features, 2, axis=1)
            normalized = [v/mx.sqrt(mx.maximum(mx.mean(mx.square(v), axis=1, keepdims=True), 1e-12)) for v in views]
            value = mx.concatenate(normalized, axis=1)
            if configuration['hidden']:
                value = nn.gelu(self.first(value))
            return self.last(value)

    if configuration['view'] not in ('base', 'dual') or configuration['hidden'] not in (0, 128):
        raise ValueError('Unregistered reference-head configuration')
    return Head()
