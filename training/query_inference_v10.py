"""Fresh fixed-query hidden features for research regression inference."""
from roguard.gemma4_attention import capture, capture_with_prefix, replay
from training.extract_base_reference_v10 import without_adapters
from training.probe_assertion_query_v10 import tokens_for


class QueryView:
    def __init__(self, engine, prompt):
        self.engine = engine
        self.prompt = prompt
        probes = [tokens_for(engine.tokenizer, prompt, {'turns': turns})[0] for turns in (
            [{'role': 'minor', 'text': 'x'}],
            [{'role': 'minor', 'text': 'y'}, {'role': 'minor', 'text': 'z'}])]
        common = 0
        for values in zip(*probes):
            if len(set(values)) != 1:
                break
            common += 1
        if not common or common >= min(map(len, probes)):
            raise ValueError('Invalid fixed-query prefix')
        self.prefix_tokens = probes[0][:common]
        self.prefix = engine.make_cache(engine.model.base)
        with without_adapters(engine.model.base):
            engine.mx.eval(engine.model.base.model(engine.mx.array([self.prefix_tokens]), cache=self.prefix))
        self.maximum_full_parity_error = 0.

    def __call__(self, payload, *, verify_full=False):
        import numpy as np
        engine = self.engine; mx = engine.mx
        tokens, _ = tokens_for(engine.tokenizer, self.prompt, payload)
        if not tokens or len(tokens) > 1024:
            raise ValueError('Query input exceeds its verified envelope')
        common = len(self.prefix_tokens)
        with without_adapters(engine.model.base):
            if len(tokens) > common and tokens[:common] == self.prefix_tokens:
                state = capture_with_prefix(engine.model.base, mx.array([tokens[common:]]), self.prefix, start=24)
            else:
                state = capture(engine.model.base, mx.array([tokens]), start=24)
            hidden = replay(engine.model.base, state)[:, 0, :]; mx.eval(hidden)
            values = np.asarray(hidden)[0].copy()
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite query representation')
            if verify_full:
                full = engine.model.base.model(mx.array([tokens]))[:, -1, :]; mx.eval(full)
                error = float(np.max(np.abs(np.asarray(full)[0]-values)))
                self.maximum_full_parity_error = max(error, self.maximum_full_parity_error)
                if error > 1e-3:
                    raise ValueError('Query regression replay differs from full inference')
        return values
