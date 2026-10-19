from collections import Counter
import json

from training.reported_support_v9 import BANKS, generate
from training.prepare_instruction_sft_v8 import target


def test_scope_support_contrasts_follow_existing_policy_and_have_distinct_inputs():
    seen = set()
    for partition, count in [('train', 2304), ('dev', 288)]:
        rows = generate('uk', partition)
        assert len(rows) == count
        assert len({r['case_id'] for r in rows}) == count
        assert set(Counter(r['pair_group'] for r in rows).values()) == {2}
        assert {r['attributes']['source'] for r in rows} == {'DIRECT', 'REPORTED', 'FICTION'}
        for row in rows:
            key = json.dumps(row['turns'], ensure_ascii=False)
            assert key not in seen
            seen.add(key)
            a = row['attributes']
            assert row['review'] == target(row)
            if a['concern'] == '0':
                assert row['review'] == 0
            elif a['source'] == 'REPORTED' and a['assertion'] in ('AFFIRMED', 'UNCERTAIN', 'RETRACTED'):
                assert row['review'] == 1
            if a['assertion'] == 'RETRACTED':
                assert len(row['turns']) >= 3
    for key in ('concern', 'ordinary', 'scope', 'time', 'assertion', 'history'):
        assert not set(BANKS['train'][key]) & set(BANKS['dev'][key])
