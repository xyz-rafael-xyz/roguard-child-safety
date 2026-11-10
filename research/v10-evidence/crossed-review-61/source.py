"""Cross assertion, attribution and time while retaining composition56 frame splits."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

from roguard.assurance import digest
from roguard.trajectory_data import file_sha,write_jsonl
from training.compositional_review_frames_v10 import PAIRS
from training.compositional_response_frames_v10 import TOPICS
from training.prepare_compositional_review_v10 import CONTEXTS
from training.prepare_instruction_sft_v8 import target
from training.generate_large_corpus_v10 import normalized,protected_texts

SPLITS={'train':(range(16),(0,1)),'dev':(range(16,24),(16,)),'test':(range(24,32),(20,21))}
TIMES={
 'ro':{'CURRENT':'Vorbesc despre situația de acum.','PAST':'Vorbesc doar despre situația de anul trecut.',
       'HYPOTHETICAL':'Vorbesc despre o situație posibilă, care ar putea să apară.'},
 'uk':{'CURRENT':'Я говорю про теперішню ситуацію.','PAST':'Я говорю лише про торішню ситуацію.',
       'HYPOTHETICAL':'Я говорю про можливу ситуацію, яка могла б виникнути.'}}
CLAIMS={
 'ro':{
  'SELF':{'AFFIRMED':'În situația discutată, următoarea descriere este adevărată despre mine:',
          'NEGATED':'În situația discutată, următoarea descriere NU este adevărată despre mine:',
          'UNCERTAIN':'Nu știu dacă, în situația discutată, următoarea descriere este adevărată despre mine:'},
  'OTHER':{'AFFIRMED':'Un prieten spune că, în situația discutată, următoarea descriere este adevărată despre el:',
           'NEGATED':'Un prieten spune că, în situația discutată, următoarea descriere NU este adevărată despre el:',
           'UNCERTAIN':'Un prieten nu știe dacă, în situația discutată, următoarea descriere este adevărată despre el:'}},
 'uk':{
  'SELF':{'AFFIRMED':'У ситуації, про яку йдеться, наступний опис відповідає дійсності щодо мене:',
          'NEGATED':'У ситуації, про яку йдеться, наступний опис НЕ відповідає дійсності щодо мене:',
          'UNCERTAIN':'Я не знаю, чи в ситуації, про яку йдеться, наступний опис відповідає дійсності щодо мене:'},
  'OTHER':{'AFFIRMED':'Друг каже, що в ситуації, про яку йдеться, наступний опис відповідає дійсності щодо нього:',
           'NEGATED':'Друг каже, що в ситуації, про яку йдеться, наступний опис НЕ відповідає дійсності щодо нього:',
           'UNCERTAIN':'Друг не знає, чи в ситуації, про яку йдеться, наступний опис відповідає дійсності щодо нього:'}}}


def compose(core,language,assertion,subject,time,concern,order):
    if not concern and assertion=='NEGATED':raise ValueError('Negating ordinary functioning has no automatic benign label')
    if order not in (0,1):raise ValueError('Unknown clause order')
    claim=CLAIMS[language][subject][assertion]+' «'+core+'»'
    temporal=TIMES[language][time]
    text=temporal+' '+claim if order==0 else claim+' '+temporal
    attrs={'subject':subject,'time':time,'assertion':assertion,'source':'DIRECT' if subject=='SELF' else 'REPORTED',
           'relation':'NEW','concern':str(int(concern))}
    return text,attrs


def build():
    rows={split:[] for split in SPLITS};seen={};families={};cores={split:set() for split in SPLITS}
    for language,pairs in PAIRS.items():
        if len(pairs)!=32:raise ValueError('Frame inventory changed')
        for split,(frames,topics) in SPLITS.items():
            for frame in frames:
                # Preserve the parent's family ID across experiments and translations.
                family=f'composition56:review:{frame:02d}'
                if families.setdefault(family,split)!=split:raise ValueError('Family leakage')
                for side,core in enumerate(pairs[frame]):
                    concern=side==0;cores[split].add(normalized(core))
                    assertions=('AFFIRMED','NEGATED','UNCERTAIN') if concern else ('AFFIRMED','UNCERTAIN')
                    for assertion,subject,time,order,topic,context in itertools.product(
                            assertions,('SELF','OTHER'),('CURRENT','PAST','HYPOTHETICAL'),(0,1),topics,(0,1)):
                        text,attrs=compose(core,language,assertion,subject,time,concern,order)
                        lead=('Vorbeam despre '+TOPICS[language][topic]+'.' if language=='ro' else
                              'Ми говорили про '+TOPICS[language][topic]+'.')
                        instruction=CONTEXTS[language][split][context]
                        turns=[{'role':'assistant','text':lead+(' '+instruction if instruction else '')},
                               {'role':'minor','text':text}]
                        key=digest({'language':language,'turns':turns})
                        if key in seen:raise ValueError('Duplicate crossed input')
                        seen[key]=split
                        row={'case_id':f'composition61:review:{frame:02d}:{language}:{side}:{assertion}:{subject}:{time}:{order}:{topic}:{context}',
                             'family':family,'language':language,'turns':turns,'focus':1,'attributes':attrs,
                             'partition':split,'source':'controlled-crossed-scope-composition','scope':assertion+':'+subject+':'+time,
                             'topic':topic,'synthetic_only':True,'independently_annotated':False}
                        row['review']=target(row);rows[split].append(row)
    for left,right in (('train','dev'),('train','test'),('dev','test')):
        if cores[left]&cores[right] or set(SPLITS[left][1])&set(SPLITS[right][1]):raise ValueError('Core/topic leakage')
    stats={split:{'rows':len(population),'paired_frames':len({row['family'] for row in population}),
                 'distinct_core_sentences':len(cores[split]),
                 'counts':dict(Counter(row['language']+':'+str(row['review']) for row in population)),
                 'factor_counts':dict(Counter(row['scope']+':'+row['attributes']['concern'] for row in population))}
           for split,population in rows.items()}
    return rows,cores,stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','private','amendment','parent_study'):p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    a=p.parse_args()
    if a.study.exists() or a.private.exists():raise ValueError('Refusing to overwrite crossed compositions')
    parent=json.loads((a.parent_study/'report.json').read_text())
    if (parent.get('complete') is not True or parent['format']!='roguard-compositional-review-1'
            or parent['frame_source_sha256']!=file_sha('training/compositional_review_frames_v10.py')
            or parent['topic_source_sha256']!=file_sha('training/compositional_response_frames_v10.py')):
        raise ValueError('Parent frame or topic binding differs')
    rows,cores,stats=build();plan=json.loads(a.amendment.read_text());protected,bindings=protected_texts()
    if {split:stats[split]['rows'] for split in stats}!=plan['rows']:raise ValueError('Registered counts differ')
    for split,words in cores.items():
        if words&protected:raise ValueError('Core overlaps protected historical wording')
        if any(normalized(row['turns'][-1]['text']) in protected for row in rows[split]):raise ValueError('Focal wording overlaps protected historical input')
    a.study.mkdir(parents=True);a.private.mkdir(parents=True);artifacts={}
    for split,population in rows.items():
        write_jsonl(a.private/(split+'.jsonl'),population);artifacts[split]=file_sha(a.private/(split+'.jsonl'))
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    report={'format':'roguard-compositional-review-1','complete':True,'source_sha256':file_sha(__file__),
            'parent_report_sha256':file_sha(a.parent_study/'report.json'),
            'frame_source_sha256':file_sha('training/compositional_review_frames_v10.py'),
            'topic_source_sha256':file_sha('training/compositional_response_frames_v10.py'),
            'context_source_sha256':file_sha('training/prepare_compositional_review_v10.py'),
            'ontology_source_sha256':file_sha('src/roguard/evidence.py'),'amendment_sha256':file_sha(a.amendment),
            'artifacts':artifacts,'populations':stats,'historical_bindings':bindings,'historical_exact_core_overlap':0,
            'parent_family_partition_preserved':True,'exact_cross_partition_core_overlap':0,'cross_partition_frame_overlap':0,
            'synthetic_only':True,'independent_annotation':False,'test_instances_opened_for_model_selection':False,
            'limitation':'Controlled combinations reuse composition56 cores. Counts represent correlated expansions, not new independent meanings or human judgments. Gold labels are conditional on the intended scopes of quoted content.',
            'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({split:{k:v for k,v in stats[split].items() if k!='factor_counts'} for split in stats},indent=2))


if __name__=='__main__':main()
