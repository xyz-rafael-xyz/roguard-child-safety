"""Build TRAIN-only authored contrasts without claiming independent semantic gold."""
from collections import Counter
from copy import deepcopy
import unicodedata

from roguard.assurance import digest
from roguard.evidence import ONTOLOGY
from training.prepare_instruction_sft_v8 import target


OPERATORS = {
    'ro': {
        'AFFIRMED': ('Descrierea citată este adevărată în situația despre care vorbim.',
                     'Confirm ceea ce spune citatul.'),
        'NEGATED': ('Afirmația din citat este falsă în această situație.',
                    'Ceea ce spune citatul nu este adevărat.'),
        'UNCERTAIN': ('Descrierea din citat ar putea fi adevărată, dar încă am îndoieli.',
                      'Nu știu încă dacă afirmația citată corespunde situației.'),
        'RETRACTED': ('Îmi retrag afirmația de mai devreme; citatul o reproduce.',
                      'Citatul redă ce am spus înainte, dar acum retrag acea afirmație.')},
    'uk': {
        'AFFIRMED': ('Наведений у цитаті опис правдивий у ситуації, про яку ми говоримо.',
                     'Підтверджую те, що сказано в цитаті.'),
        'NEGATED': ('Твердження в цитаті хибне в цій ситуації.',
                    'Те, що сказано в цитаті, не відповідає дійсності.'),
        'UNCERTAIN': ('Опис у цитаті може бути правдивим, але я ще маю сумніви.',
                      'Я ще не знаю, чи відповідає наведене твердження дійсності.'),
        'RETRACTED': ('Відкликаю своє попереднє твердження, яке відтворене в цитаті.',
                      'Цитата передає мої попередні слова, але зараз я відкликаю це твердження.')}}

CONTEXTS = {
    'ro': {'direct': 'Vorbim despre situația ta reală de acum.',
           'fiction': 'Repetăm replicile unui personaj inventat, în scena care se petrece acum.',
           'start': 'Aș vrea să discutăm o descriere.'},
    'uk': {'direct': 'Говоримо про твою реальну ситуацію зараз.',
           'fiction': 'Репетируємо репліки вигаданого персонажа в сцені, що відбувається зараз.',
           'start': 'Хочу обговорити один опис.'}}


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def validate_manifest(manifest):
    if (manifest.get('format') != 'roguard-authored-semantic-diversity-1'
            or manifest.get('partition') != 'train'
            or manifest.get('synthetic_only') is not True
            or manifest.get('independently_annotated') is not False):
        raise ValueError('Expected explicitly synthetic TRAIN-only manifest')
    families = set(); cores = set()
    pairs = manifest.get('pairs')
    if not isinstance(pairs, list) or not pairs:
        raise ValueError('Missing contrast pairs')
    for pair in pairs:
        family = pair.get('family')
        if not isinstance(family, str) or not family or family in families:
            raise ValueError('Missing or duplicate family')
        families.add(family)
        for lang in ('ro', 'uk'):
            texts = pair.get(lang)
            if not isinstance(texts, list) or len(texts) != 2:
                raise ValueError('Each language requires benign and concern cores')
            for core in texts:
                if (not isinstance(core, str) or not core.strip() or '\n' in core
                        or '«' in core or '»' in core):
                    raise ValueError('Expected one unquoted core statement')
                key = (lang, normalized(core))
                if key in cores:
                    raise ValueError('Duplicate core')
                cores.add(key)
    return cores


def build(manifest, protected_rows):
    """Use existing inputs only to reject overlap, never as generation parents."""
    cores = validate_manifest(manifest)
    protected_inputs = set(); protected_focus = set()
    for row in protected_rows:
        payload = row['payload']
        if row['partition'] not in ('train', 'dev'):
            raise ValueError('Only the already exposed TRAIN/DEV corpus is permitted')
        lang = payload['language']
        focus = normalized(payload['turns'][-1]['text'])
        protected_inputs.add(digest(payload)); protected_focus.add((lang, focus))
        if any(language == lang and core in focus for language, core in cores):
            raise ValueError('Authored core occurs in existing TRAIN/DEV focus')
    rows = []; seen = set()

    def add(lang, pair, concern, scope, assertion, operator_index, placement, bare=False):
        core = pair[lang][concern]
        attributes = dict(subject='SELF' if scope == 'direct' else 'FICTIONAL',
                          time='CURRENT', assertion=assertion,
                          source='DIRECT' if scope == 'direct' else 'FICTION',
                          relation='RETRACTS' if assertion == 'RETRACTED' else 'NEW',
                          concern=str(concern))
        if bare:
            turns = [dict(role='minor', text=core)]
        else:
            operator = OPERATORS[lang][assertion][operator_index]
            quote = '«'+core+'»'
            focus = operator+' '+quote if placement == 'before' else quote+' '+operator
            prior = core if assertion == 'RETRACTED' else CONTEXTS[lang]['start']
            turns = [dict(role='minor', text=prior),
                     dict(role='assistant', text=CONTEXTS[lang][scope]),
                     dict(role='minor', text=focus)]
        payload = dict(language=lang, turns=turns)
        key = digest(payload)
        if key in seen or key in protected_inputs:
            raise ValueError('Duplicate or protected model input')
        if (lang, normalized(turns[-1]['text'])) in protected_focus:
            raise ValueError('Focal text overlaps existing TRAIN/DEV')
        seen.add(key)
        identity = dict(language=lang, family=pair['family'], concern=concern,
                        scope=scope, assertion=assertion, operator=operator_index,
                        placement=placement, bare=bare)
        rows.append(dict(case_id='semantic99:'+digest(identity), language=lang,
                         partition='train', group='authored-semantic-diversity',
                         family=pair['family'], attributes=attributes,
                         fields=[ONTOLOGY[k].index(attributes[k]) for k in ONTOLOGY],
                         gold=target(dict(attributes=attributes)), payload=payload,
                         source_focus=len(turns)-1, core=core, factors=identity,
                         synthetic_only=True, independently_annotated=False))

    for pair in manifest['pairs']:
        for lang in ('ro', 'uk'):
            for concern in (0, 1):
                add(lang, pair, concern, 'direct', 'AFFIRMED', 0, 'bare', bare=True)
                for scope in ('direct', 'fiction'):
                    for assertion in ONTOLOGY['assertion']:
                        for op in (0, 1):
                            for placement in ('before', 'after'):
                                add(lang, pair, concern, scope, assertion, op, placement)
    expected = len(manifest['pairs'])*2*2*(1+2*4*2*2)
    if len(rows) != expected:
        raise ValueError('Incomplete factorial population')
    return deepcopy(rows)


def summarize(rows):
    counts = Counter((r['language'], r['factors']['scope'], r['attributes']['assertion'],
                      r['attributes']['concern'], r['gold'], r['factors']['bare']) for r in rows)
    return dict(rows=len(rows), families=len({r['family'] for r in rows}),
                distinct_cores={lang:len({r['core'] for r in rows if r['language']==lang})
                                for lang in ('ro','uk')},
                counts=[dict(language=k[0], scope=k[1], assertion=k[2], concern=k[3],
                             gold=k[4], bare=k[5], rows=n) for k,n in sorted(counts.items())])
