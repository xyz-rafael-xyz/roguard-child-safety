"""Prospective expanded disagreement masking ONLY on explicitly tagged new FIT occurrences."""
from roguard.native_fields import FIELDS,SIZES
MODES=('mask_wrong_assertion','mask_wrong_new_fit_semantics')
TERMS=len(SIZES)+1
ASSERTION_INDEX=FIELDS.index('assertion')

def training_teacher_mask(rows,teacher_logits,mode):
 import numpy as np
 if mode not in MODES:raise ValueError('Unknown semantic mask mode')
 if not isinstance(rows,list) or not rows:raise ValueError('No admitted fitting rows')
 if not isinstance(teacher_logits,np.ndarray) or teacher_logits.shape!=(len(rows),sum(SIZES)+2) or teacher_logits.dtype.kind!='f' or not np.isfinite(teacher_logits).all():raise ValueError('Invalid teacher logits')
 for row in rows:
  fields=row.get('fields')
  if row.get('partition')!='train' or row.get('fit_origin') not in ('original_train','new_fit'):raise ValueError('Only explicitly tagged fitting origins allowed')
  if not isinstance(fields,list) or len(fields)!=len(SIZES) or any(type(v) is not int or not 0<=v<n for v,n in zip(fields,SIZES)):raise ValueError('Invalid TRAIN fields')
 keep=np.ones((len(rows),TERMS),dtype=np.bool_);offset=0
 for j,size in enumerate(SIZES):
  agree=teacher_logits[:,offset:offset+size].argmax(1)==np.array([r['fields'][j] for r in rows])
  if j==ASSERTION_INDEX:keep[:,j]=agree
  elif mode=='mask_wrong_new_fit_semantics':keep[:,j]=agree|np.array([r['fit_origin']=='original_train' for r in rows])
  offset+=size
 return keep

def teacher_kl_per_row(mx,student_logits,teacher_logits,keep):
 if len(student_logits.shape)!=2 or student_logits.shape[1]!=sum(SIZES)+2 or teacher_logits.shape!=student_logits.shape or keep.shape!=(student_logits.shape[0],TERMS) or keep.dtype!=mx.bool_:raise ValueError('Semantic KL tensor/mask contract differs')
 divergences=[];offset=0
 for j,size in enumerate((*SIZES,2)):
  old=teacher_logits[:,offset:offset+size];new=student_logits[:,offset:offset+size]
  old_logp=old-mx.logsumexp(old,axis=-1,keepdims=True);new_logp=new-mx.logsumexp(new,axis=-1,keepdims=True)
  term=(mx.exp(old_logp)*(old_logp-new_logp)).sum(axis=-1)
  # Binary distillation is structural: even an invalid caller column cannot suppress it.
  divergences.append(term if j==TERMS-1 else term*keep[:,j]);offset+=size
 return mx.stack(divergences,axis=1).sum(axis=1)/TERMS
