"""Prepare or independently audit the registered role-framed continuation100."""
import argparse
from collections import Counter
import json
from pathlib import Path

from roguard.gemma4_contract import PREFIX_TOKENS, classifier_tokens
from roguard.trajectory_data import file_sha, write_jsonl
from training.role_framed_diversity_v10 import continue_row
from eval.verify_role_framed_diversity_v10 import verify_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','private_root','base','plan','study','output'):
        parser.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--mode',choices=('prepare','audit'),required=True)
    a=parser.parse_args()
    if file_sha(a.plan)!=a.plan_sha256:raise ValueError('Plan differs')
    plan=json.loads(a.plan.read_text())
    if (plan['format']!='roguard-role-framed-diversity-plan-1'
            or plan['training_admitted'] is not False or plan['maximum_tokens']!=600):
        raise ValueError('Invalid preparation-only contract')
    def check():
        for base,mapping in ((a.root,plan['public']),(a.private_root,plan['private']),
                             (a.base,plan['tokenizer_files'])):
            for name,sha in mapping.items():
                if file_sha(base/name)!=sha:raise ValueError('Binding differs: '+name)
        if file_sha(a.plan)!=a.plan_sha256:raise ValueError('Plan changed')
    check()
    if a.base.name!=plan['tokenizer_revision']:raise ValueError('Tokenizer revision differs')
    parent_proof=json.loads((a.root/plan['parent_verification']).read_text())
    review=json.loads((a.root/plan['parent_author_review']).read_text())
    if (parent_proof['verification']!='PASS' or parent_proof['training_admitted'] is not False
            or parent_proof['rows_sha256']!=file_sha(a.private_root/plan['parents'])
            or review['rows_sha256']!=parent_proof['rows_sha256']
            or review['decision']!='DO_NOT_ADMIT_REVISE_FICTION_FRAMING'):
        raise ValueError('Parent evidence differs')
    parents=[json.loads(line) for line in (a.private_root/plan['parents']).open()]
    protected=[json.loads(line) for line in (a.private_root/plan['protected_inputs']).open()]
    if len(parents)!=plan['rows']:raise ValueError('Parent count differs')
    if any(r['partition'] not in ('train','dev') for r in protected):
        raise ValueError('Reserved partition encountered')
    destination=a.study/('report.json' if a.mode=='prepare' else 'verification.json')
    if destination.exists():raise ValueError('Refusing to overwrite evidence')
    if a.mode=='prepare' and (a.study.exists() or a.output.exists()):
        raise ValueError('Refusing to overwrite generated dataset')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    tokenize=lambda payload:classifier_tokens(tokenizer,payload)
    if a.mode=='prepare':
        rows=[continue_row(parent,tokenize) for parent in parents]
        old={r['token_sha256'] for r in protected};seen=set()
        for row in rows:
            values=row['tokens'];sha=row['token_sha256']
            if (sha in seen or sha in old or values[:len(PREFIX_TOKENS)]!=PREFIX_TOKENS
                    or not len(PREFIX_TOKENS)<len(values)<=600):
                raise ValueError('Duplicate/protected tokens or token envelope violation')
            seen.add(sha)
        counts=Counter((r['language'],r['framing_changed']) for r in rows)
        summary=dict(rows=len(rows),fiction_rows_changed=sum(r['framing_changed'] for r in rows),
                     direct_rows_unchanged=sum(not r['framing_changed'] for r in rows),
                     minimum_tokens=min(r['token_count'] for r in rows),
                     maximum_tokens=max(r['token_count'] for r in rows),
                     counts=[dict(language=k[0],fiction=k[1],rows=n) for k,n in sorted(counts.items())])
        if summary['fiction_rows_changed']!=plan['fiction_rows']:
            raise ValueError('Fiction count differs')
        check();a.output.mkdir(parents=True);a.study.mkdir(parents=True)
        write_jsonl(a.output/'rows.jsonl',rows)
        result=dict(format='roguard-role-framed-diversity-1',complete=True,**summary,
                    rows_sha256=file_sha(a.output/'rows.jsonl'),source_sha256=file_sha(__file__),
                    plan_sha256=a.plan_sha256,encoder_loaded=False,training_performed=False,
                    training_admitted=False,independent_language_or_label_validation=False,
                    held_out_test_opened=False,qualified=False)
    else:
        rows_path=a.output/'rows.jsonl';before=file_sha(rows_path)
        rows=[json.loads(line) for line in rows_path.open()]
        report_path=a.study/'report.json';report_sha=file_sha(report_path)
        report=json.loads(report_path.read_text())
        summary=verify_rows(parents,rows,tokenize,protected)
        if (not report['complete'] or report['plan_sha256']!=a.plan_sha256
                or report['source_sha256']!=file_sha(__file__) or report['rows_sha256']!=before
                or any(report[k]!=summary[k] for k in ('rows','fiction_rows_changed',
                       'direct_rows_unchanged','minimum_tokens','maximum_tokens','counts'))):
            raise ValueError('Report differs from independent audit')
        for k in ('encoder_loaded','training_performed','training_admitted',
                  'independent_language_or_label_validation','held_out_test_opened','qualified'):
            if report[k] is not False:raise ValueError('Unsupported preparation claim')
        check()
        if file_sha(rows_path)!=before or file_sha(report_path)!=report_sha:
            raise ValueError('Outputs changed during verification')
        result=dict(verification='PASS',**summary,rows_sha256=before,report_sha256=report_sha,
                    plan_sha256=a.plan_sha256,audit_source_sha256=file_sha(a.root/'eval/verify_role_framed_diversity_v10.py'),
                    encoder_loaded=False,training_admitted=False,independent_language_or_label_validation=False,
                    held_out_test_opened=False,qualified=False,
                    scope='Independent allowed-change audit and actual token replay of every retained continuation row. No generator transform is called by audit mode. Explicit framing and preserved labels are structural/author evidence, not independent natural-language gold.')
    with destination.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('verification','complete','rows',
        'fiction_rows_changed','direct_rows_unchanged','minimum_tokens','maximum_tokens','training_admitted')}))


if __name__=='__main__':main()
