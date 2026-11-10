"""Continue draft99 with explicit fictional framing in the minor's own turns."""
from copy import deepcopy

from roguard.assurance import digest


FRAMES = {
    'ro': {'prior':'Repet o replică a unui personaj fictiv.',
           'prior_quote':'În rolul personajului fictiv, spun: «{core}»',
           'focus':'Aceasta este replica unui personaj fictiv. '},
    'uk': {'prior':'Репетирую репліку вигаданого персонажа.',
           'prior_quote':'У ролі вигаданого персонажа кажу: «{core}»',
           'focus':'Це репліка вигаданого персонажа. '}}


def continue_row(parent, tokenize):
    if (parent['partition']!='train' or not parent['case_id'].startswith('semantic99:')
            or parent['synthetic_only'] is not True or parent['independently_annotated'] is not False):
        raise ValueError('Expected retained synthetic TRAIN parent99')
    scope=parent['factors']['scope']
    if scope not in ('direct','fiction'):raise ValueError('Unsupported parent scope')
    row=deepcopy(parent)
    row.update(case_id=parent['case_id'].replace('semantic99:','semantic100:',1),
               group='role-framed-authored-semantic-diversity',parent_case_id=parent['case_id'],
               parent_row_sha256=digest(parent),framing_changed=scope=='fiction')
    if scope=='fiction':
        turns=row['payload']['turns']; frame=FRAMES[row['language']]
        if len(turns)!=3 or [t['role'] for t in turns]!=['minor','assistant','minor']:
            raise ValueError('Unexpected fictional parent roles')
        if row['attributes']['assertion']=='RETRACTED':
            if turns[0]['text']!=row['core']:raise ValueError('Prior statement differs')
            turns[0]['text']=frame['prior_quote'].format(core=row['core'])
        else:
            turns[0]['text']=frame['prior']
        turns[-1]['text']=frame['focus']+turns[-1]['text']
    values=tokenize(row['payload'])
    row.update(tokens=values,token_sha256=digest(values),token_count=len(values))
    if scope=='direct' and any(row[k]!=parent[k] for k in ('payload','tokens','token_sha256','token_count')):
        raise ValueError('Direct input changed')
    return row
