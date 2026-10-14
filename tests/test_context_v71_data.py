"""Guard the synthetic experiment's split and topic-transition contracts."""
import hashlib
import json
from pathlib import Path
import unittest
from training.context_v71 import additions, TEXT
from roguard.context import render


class ContextDataTests(unittest.TestCase):
    def test_partition_inputs_do_not_overlap(self):
        seen=set()
        for split in ('train','dev','cal','test'):
            rows=additions(split)
            current={(r['language'],render(r['turns'])) for r in rows}
            self.assertFalse(seen & current)
            seen.update(current)
            self.assertEqual({r['language'] for r in rows},{'ro','uk'})
            self.assertEqual({r['signal'] for r in rows},{0,1})
        self.assertEqual(len(TEXT['ro']),len(TEXT['uk']))

    def test_frozen_test_matches_generator_and_hash(self):
        root=Path('research/v7-1-context')
        data=(root/'fresh-test.jsonl').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),json.loads((root/'test-freeze.json').read_text())['sha256'])
        self.assertEqual([json.loads(line) for line in data.decode().splitlines()],additions('test'))

    def test_concern_after_schoolwork_has_explicit_personal_transition(self):
        for split in ('train','dev','cal','test'):
            rows=additions(split)
            for row in rows:
                if row['signal']:
                    self.assertEqual(row['frame'],'SELF_CURRENT')
                    self.assertEqual(len(row['turns']),2)
                    text=row['turns'][-1]['text']
                    self.assertIn(({'train':'mine','dev':'propria mea','cal':'mie','test':'mine'}[split]) if row['language']=='ro' else ('я почуваюся' if split=='dev' else 'мною' if split=='cal' else 'мене' if split=='test' else 'себе'),text)

if __name__=='__main__': unittest.main()

class TransferPartitionTests(unittest.TestCase):
    def test_repair_support_and_transfer_partitions_stay_separate(self):
        from training.context_repair_v71 import repairs
        from training.context_support_v71 import support
        from training.context_transfer_v71 import transfer
        seen=set()
        for split in ('train','dev','cal'):
            rows=repairs(split)+support(split)+transfer(split)
            current={(r['language'],render(r['turns'])) for r in rows}
            self.assertFalse(seen & current)
            seen.update(current)
        final_path=Path('research/v7-1-context/final2-test.jsonl')
        data=final_path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),json.loads(final_path.with_name('final2-freeze.json').read_text())['sha256'])
        final_rows=[json.loads(line) for line in data.decode().splitlines()]
        self.assertFalse(seen & {(r['language'],render(r['turns'])) for r in final_rows})
