"""Exercise complete audit orchestration with synthetic arrays and CPU-only heads."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from roguard.trajectory_data import file_sha,write_jsonl
from eval.verify_role_diversity_runtime_v10 import main


@unittest.skipUnless(importlib.util.find_spec('mlx') and importlib.util.find_spec('numpy'), 'MLX CPU fixture')
class RoleAuditCliTests(unittest.TestCase):
    def fixture(self,base):
        import numpy as np
        from safetensors.numpy import save_file
        root=base/'root';private=base/'private';study=root/'study';data=private/'data'
        study.mkdir(parents=True);data.mkdir(parents=True)
        source=study/'source.py';source.write_text('# fixed fixture extractor\n')
        ep=root/'extraction.json';ep.write_text('{}\n')
        ep_sha=file_sha(ep);source_sha=file_sha(source)
        extraction=dict(public={'training/extract_role_diversity_runtime_v10.py':source_sha})
        protocol=dict(format='roguard-role-diversity-runtime-1',plan_sha256=ep_sha,source_sha256=source_sha,
            modes=['v9','selected95'],new_train_rows=3168,new_forward_inputs=6336,selected_development_rows=8837,
            native_replay_tolerance=1e-3,reference_logit_tolerance=1e-5,runtime_score_tolerance=1e-10,
            training_performed=False,held_out_test_opened=False,qualified=False)
        (study/'protocol.json').write_text(json.dumps(protocol))
        report=dict(complete=True,decision='REFERENCE_EXTRACTION_ONLY_REQUIRES_INDEPENDENT_AUDIT',
            training_performed=False,held_out_test_opened=False,qualified=False,languages={},seconds=200.)
        groups=list(itertools.product(('direct','fiction'),('AFFIRMED','NEGATED','UNCERTAIN','RETRACTED'),(0,1),(False,)))
        groups.extend([('direct','AFFIRMED',c,True) for c in (0,1)])
        languages={};events=[]
        for lang,n_dev in [('ro',4202),('uk',4635)]:
            rows=[dict(case_id=f'{lang}-{i}',partition='train',token_sha256=f'{i:064x}',
                       factors=dict(zip(('scope','assertion','concern','bare'),groups[i%18]))) for i in range(1584)]
            write_jsonl(data/(lang+'-rows.jsonl'),rows)
            tensors={f'heads.{lang}.first.weight':np.zeros((128,2560),dtype=np.float32),
                     f'heads.{lang}.first.bias':np.zeros(128,dtype=np.float32),
                     f'heads.{lang}.last.weight':np.zeros((19,128),dtype=np.float32),
                     f'heads.{lang}.last.bias':np.zeros(19,dtype=np.float32)}
            weight=private/(lang+'-weights.safetensors');save_file(tensors,str(weight))
            spec=dict(mode='binary',threshold=.5)
            reference=np.zeros((n_dev,19),dtype=np.float32)
            languages[lang]=dict(train=rows,dev=[{} for _ in range(n_dev)],reference=reference,
                weights={m:weight for m in ('v9','selected95')},specs={m:spec for m in ('v9','selected95')})
            record=dict(complete=True,modes={},rows_sha256=file_sha(data/(lang+'-rows.jsonl')))
            for mode in ('v9','selected95'):
                names=[]
                for suffix,values in [('features.npy',np.ones((1584,2560),dtype=np.float32)),('logits.npy',np.zeros((1584,19),dtype=np.float32))]:
                    path=data/f'{lang}-{mode}-{suffix}';np.save(path,values,allow_pickle=False);names.append(path)
                obs=[dict(index=i,case_id=r['case_id'],token_sha256=r['token_sha256'],native_replay_error=0.,
                    runtime_score_error=0. if i<18 else None) for i,r in enumerate(rows)]
                path=data/f'{lang}-{mode}-observations.jsonl';write_jsonl(path,obs);names.append(path)
                m=dict(complete=True,train_checked=1584,dev_checked=0,maximum_native_replay_error=0.,
                       maximum_runtime_score_error=0.,runtime_groups_checked=18)
                stages=[('train',1584)]
                if mode=='selected95':
                    path=data/(lang+'-selected95-dev-logits.npy');np.save(path,reference,allow_pickle=False);names.append(path)
                    m.update(dev_checked=n_dev,development_parity=dict(maximum_logit_error=0.,argmax_rows_differing=0,fixed_threshold_decision_differences=0))
                    stages.append(('dev',n_dev))
                m['artifacts']={p.name:file_sha(p) for p in names};record['modes'][mode]=m
                for stage,n in stages:
                    for checked in [*range(100,n+1,100),n]:
                        events.append(dict(language=lang,mode=mode,stage=stage,checked=checked,rows=n,seconds=float(len(events)+1)))
            report['languages'][lang]=record
        (study/'report.json').write_text(json.dumps(report))
        events.append(dict(complete=True,decision=report['decision'],seconds=200.))
        (private/'log.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
        queue=dict(source_hashes={},steps=[dict(name='extract')]);(private/'queue.json').write_text(json.dumps(queue))
        sequence=dict(plan_sha256=file_sha(private/'queue.json'),source_hashes={},execution_complete=True,phase='finished',completed_steps=[dict(name='extract',exit_code=0)])
        (root/'sequence.json').write_text(json.dumps(sequence))
        audit=dict(public={},private={'queue.json':file_sha(private/'queue.json')},cpu_logit_tolerance=1e-3,
                   study='study',data='data',sequence='sequence.json',queue_plan='queue.json',producer_queue_pid=123,
                   extraction_plan='extraction.json',extraction_plan_sha256=ep_sha,log='log.jsonl')
        ap=root/'audit.json';ap.write_text(json.dumps(audit))
        return root,private,ap,extraction,languages

    def invoke(self,root,private,ap,extraction,languages,output):
        args=['audit','--root',str(root),'--private-root',str(private),'--audit-plan',str(ap),
              '--audit-plan-sha256',file_sha(ap),'--output',str(output)]
        with patch.object(sys,'argv',args),patch('scripts.wait_completed_sequence_exit_v10.require_exit'),patch(
                'training.role_diversity_runtime_v10.load_registered',return_value=(extraction,languages)):
            main()

    def test_complete_cpu_fixture_and_late_report_corruption(self):
        import mlx.core as mx
        device=mx.default_device()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root,private,ap,extraction,languages=self.fixture(Path(temporary))
                output=private/'verification.json'
                self.invoke(root,private,ap,extraction,languages,output)
                proof=json.loads(output.read_text())
                self.assertEqual((proof['artifact_count'],proof['cpu_head_rows'],proof['development_rows_compared'],proof['ordered_progress_events']),(16,6336,8837,154))
                self.assertFalse(proof['encoder_loaded']);self.assertFalse(proof['qualified'])
                path=root/'study/report.json';report=json.loads(path.read_text())
                report['languages']['ro']['modes']['selected95']['development_parity']['argmax_rows_differing']=1
                path.write_text(json.dumps(report));bad=private/'corrupt-verification.json'
                with self.assertRaisesRegex(ValueError,'development parity'):
                    self.invoke(root,private,ap,extraction,languages,bad)
                self.assertFalse(bad.exists())
        finally:mx.set_default_device(device)


if __name__=='__main__':unittest.main()
