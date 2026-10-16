import unittest
from collections import defaultdict
from training.context_families_v8 import generate
from training.prepare_instruction_sft_v8 import target, sample, messages


class ContextFamiliesTests(unittest.TestCase):
    def test_whole_families_are_disjoint(self):
        splits=generate()
        seen=set()
        for split,rows in splits.items():
            families={r['family'] for r in rows}
            self.assertFalse(seen & families)
            seen.update(families)
            self.assertTrue(all(r['partition']==split for r in rows))
            self.assertEqual({r['language'] for r in rows},{'ro','uk'})
        ids=[r['case_id'] for rows in splits.values() for r in rows]
        self.assertEqual(len(ids),len(set(ids)))

    def test_scope_pairs_and_invariance(self):
        groups=defaultdict(list)
        for rows in generate().values():
            for row in rows:
                expected=0 if row['scope'] in ('fiction','denial','topic_reset') else 1
                self.assertEqual(target(row),expected,row['case_id'])
                groups[row['invariant_group']].append(row)
        for rows in groups.values():
            self.assertEqual({r['variant'] for r in rows},{'original','nfd','spaces'})
            self.assertEqual(len({target(r) for r in rows}),1)

    def test_chat_payload_excludes_labels_and_family(self):
        row=generate()['train'][0]
        chat=messages(row)
        self.assertEqual(len(chat),2)
        self.assertNotIn('attributes',chat[1]['content'])
        self.assertNotIn('family',chat[1]['content'])
        self.assertEqual(messages(row,True)[-1]['content'],str(target(row)))

    def test_missing_sampling_group_fails(self):
        with self.assertRaises(ValueError):sample(generate()['train'],1,1)
