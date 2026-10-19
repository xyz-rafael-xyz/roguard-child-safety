from collections import Counter,defaultdict
from training.morphology_scope_v9 import generate,realization
from training.policy_entailment_v9 import targets
from training.train_paired_nli_v9 import premise


def test_balanced_gender_and_source_have_no_label_shortcut():
    for language in ('ro','uk'):
        rows=generate(language)
        assert len(rows)==1728
        assert len({premise(r) for r in rows})==len(rows)
        groups=defaultdict(list)
        counts=Counter()
        for r in rows:
            assert targets(r)==(r['review'],0)
            counts[r['grammatical_gender'],r['review']]+=1
            if r['pair_group']:groups[r['pair_group']].append(r)
        assert counts['feminine',0]==counts['masculine',0]
        assert counts['feminine',1]==counts['masculine',1]
        assert all(sorted(r['review'] for r in pair)==[0,1] for pair in groups.values())


def test_feminine_quoted_speech_is_inflected_without_changing_other_words():
    assert realization('Mă simt singur și speriat.','ro','feminine')=='Mă simt singură și speriată.'
    assert realization('Я залишився. Загубив зошит.','uk','feminine')=='Я залишилася. Загубила зошит.'
    assert realization('Mă simt copleșit.','ro','masculine')=='Mă simt copleșit.'


def test_protected_scope_examples_are_not_copied():
    from training.paired_scope_data_v9 import generate as previous
    for language in ('ro','uk'):
        protected={premise(r) for split in ('dev','test') for r in previous(language,split)}
        assert not protected & {premise(r) for r in generate(language)}
