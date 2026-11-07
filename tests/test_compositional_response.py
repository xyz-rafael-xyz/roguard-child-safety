from collections import Counter

from training.prepare_compositional_response_v10 import build, SPLITS


def test_compositions_keep_translations_and_all_variants_in_one_partition():
    populations,cores,stats=build();seen={};draft_partitions={}
    assert {key:value['rows'] for key,value in stats.items()}=={'train':12288,'dev':512,'test':1536}
    assert {key:value['distinct_core_sentences'] for key,value in stats.items()}=={'train':192,'dev':64,'test':64}
    for partition,rows in populations.items():
        for row in rows:
            assert seen.setdefault(row['family'],partition)==partition
            assert draft_partitions.setdefault((row['language'],row['draft']),partition)==partition
            assert row['topic'] in SPLITS[partition][1]
            assert row['gold'] in ('appropriate','boundary','privacy','pressure','unsupported')
            assert row['partition']==partition
        assert len({row['case_id'] for row in rows})==len(rows)
        counts=Counter((row['family'],row['language'],row['gold']=='appropriate') for row in rows)
        assert len(set(counts.values()))==1
    assert not cores['train']&cores['dev'] and not cores['train']&cores['test']
    assert not cores['dev']&cores['test']
