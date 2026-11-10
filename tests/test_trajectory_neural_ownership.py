"""Caller-owned configuration cannot change a bound neural pipeline."""
from types import SimpleNamespace

from roguard.trajectory_neural import NeuralTrajectoryModel, FieldEngine


class SelectorFixture:
    identity = '0'*64

    def __init__(self):
        self.spec = {'min_relevance': .5}

    def relevance(self, *args):
        return .75


def make_pipeline():
    selector = SelectorFixture()
    components = {}
    for lang in ('ro', 'uk'):
        engine = object.__new__(FieldEngine)
        engine.last_fields = [[1/432]*432]
        components[lang] = SimpleNamespace(identity=lang*32, language=lang, _engine=engine,
            predict=lambda payload: dict(predicted=0, score=.1, reason='fixture'))
    return NeuralTrajectoryModel(components, selector, selection_mode='uniform'), selector, components


def test_external_selector_mutation_cannot_change_bound_memory():
    model, selector, _ = make_pipeline()
    payload = {'language':'ro', 'turns':[
        {'role':'minor', 'text':'Mi-e teamă să mă întorc acasă.'},
        {'role':'minor', 'text':'Acum desenez un copac.'}]}
    before = model.predict(payload)
    assert {span['turn'] for span in before['evidence']} == {0, 1}
    selector.spec['min_relevance'] = 1.0
    after = model.predict(payload)
    assert model.selector is not selector
    assert after == before


def test_external_component_mapping_replacement_cannot_change_bound_models():
    model, _, components = make_pipeline()
    original = model.components['ro']
    components['ro'] = SimpleNamespace(identity='replacement')
    assert model.components is not components
    assert model.components['ro'] is original
