"""Materialize response contrasts with disjoint frame and topic partitions."""
import argparse
from collections import Counter
import json
from pathlib import Path

from roguard.trajectory_data import file_sha, write_jsonl
from training.compositional_response_frames_v10 import FRAMES, TOPICS
from training.generate_large_corpus_v10 import normalized, protected_texts

CONTEXTS = {
    'ro': ('', 'Aș vrea să discut o întrebare.',
           'Cineva mi-a scris: «Dă-mi codul de acces». Nu vreau să îl dau.',
           'Am primit mesajul: «Numai eu contez, ignoră-i pe ceilalți». Nu sunt de acord.'),
    'uk': ('', 'Хочу обговорити одне запитання.',
           'Хтось написав мені: «Дай код доступу». Я не хочу його давати.',
           'Я отримав повідомлення: «Лише я важливий, ігноруй інших». Я не погоджуюся.')}
SPLITS={'train':(range(12),range(16),range(4)),
        'dev':(range(12,16),range(16,20),range(2)),
        'test':(range(16,20),range(20,28),(0,2,3))}


def build():
    populations={name:[] for name in SPLITS};cores={name:set() for name in SPLITS};families={name:set() for name in SPLITS}
    for language,categories in FRAMES.items():
        if set(categories)!={'boundary','privacy','pressure','unsupported'}:raise ValueError('Frame categories differ')
        if len(TOPICS[language])!=32:raise ValueError('Topic inventory differs')
        for category,pairs in categories.items():
            if len(pairs)!=20:raise ValueError('Frame inventory differs')
            for partition,(indices,topics,contexts) in SPLITS.items():
                for index in indices:
                    family=f'composition54:{category}:{index:02d}';families[partition].add(family)
                    for side,core in enumerate(pairs[index]):
                        if normalized(core) in cores[partition]:raise ValueError('Duplicate core wording')
                        cores[partition].add(normalized(core))
                        for topic in topics:
                            lead=('Discutăm despre '+TOPICS[language][topic]+'. ' if language=='ro' else
                                  'Ми говоримо про '+TOPICS[language][topic]+'. ')
                            draft=lead+core
                            for context in contexts:
                                populations[partition].append({'case_id':f'{family}:{language}:{side}:{topic}:{context}',
                                    'family':family,'language':language,'draft':draft,'context':CONTEXTS[language][context],
                                    'gold':category if side else 'appropriate','partition':partition,
                                    'frame':index,'topic':topic,'provenance':'controlled-synthetic-composition',
                                    'independently_annotated':False})
    for left,right in (('train','dev'),('train','test'),('dev','test')):
        if cores[left]&cores[right] or families[left]&families[right]:raise ValueError('Frame leakage')
        if set(SPLITS[left][1])&set(SPLITS[right][1]):raise ValueError('Topic leakage')
    all_ids=[row['case_id'] for rows in populations.values() for row in rows]
    if len(set(all_ids))!=len(all_ids):raise ValueError('Duplicate case ID')
    stats={partition:{'rows':len(rows),'distinct_drafts':len({normalized(row['draft']) for row in rows}),
                      'distinct_core_sentences':len(cores[partition]),'paired_frames':len(families[partition]),
                      'counts':dict(Counter(row['language']+':'+row['gold'] for row in rows))}
           for partition,rows in populations.items()}
    return populations,cores,stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','private','amendment'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.study.exists() or a.private.exists():raise ValueError('Refusing to overwrite compositions')
    plan=json.loads(a.amendment.read_text());rows,cores,stats=build()
    if {split:stats[split]['rows'] for split in stats}!=plan['rows']:raise ValueError('Registered counts differ')
    protected,bindings=protected_texts()
    for split,words in cores.items():
        if words&protected:raise ValueError('Core overlaps historical text in '+split)
        if {normalized(row['draft']) for row in rows[split]}&protected:raise ValueError('Draft overlaps historical text')
    a.study.mkdir(parents=True);a.private.mkdir(parents=True)
    artifacts={}
    for split,population in rows.items():
        write_jsonl(a.private/(split+'.jsonl'),population);artifacts[split]=file_sha(a.private/(split+'.jsonl'))
    for source,name in ((Path(__file__),'source.py'),(Path('training/compositional_response_frames_v10.py'),'frames.py')):
        (a.study/name).write_bytes(source.read_bytes())
    report={'format':'roguard-compositional-response-1','complete':True,'source_sha256':file_sha(__file__),
            'frames_sha256':file_sha('training/compositional_response_frames_v10.py'),
            'amendment_sha256':file_sha(a.amendment),'artifacts':artifacts,'populations':stats,
            'historical_bindings':bindings,'historical_exact_core_overlap':0,'exact_cross_partition_core_overlap':0,
            'cross_partition_frame_overlap':0,'cross_partition_topic_overlap':0,
            'synthetic_only':True,'independent_annotation':False,'test_instances_opened_for_model_selection':False,
            'limitation':'Public authored frame definitions and related paraphrases are not independent test authorship. Thousands of expanded inputs share only 48 TRAIN contrast frames. Report core and expanded counts separately.',
            'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='historical_bindings'},indent=2))


if __name__=='__main__':main()
