"""Portable learned draft-text + frozen query response readout. No encoder imports.

Vocabulary/IDF are learned only from supplied TRAIN drafts by the registered
trainer. No labels, source IDs, partition names, family names or context text
are accepted by the lexical feature extractor.
"""
from collections import Counter
import hashlib,json,math,re,unicodedata
from pathlib import Path
LABELS=('appropriate','boundary','privacy','pressure','unsupported')
MODES=('hybrid','q_only','text_only')
SETTINGS={'normalization':'NFC_casefold_whitespace_collapse','word_orders':[1,2,3],'char_orders':[3,4,5],'minimum_document_frequency':2,'word_maximum_features':32768,'char_maximum_features':65536,'tf':'1+log(count)','idf':'log((1+TRAIN_rows)/(1+TRAIN_document_frequency))+1','block_normalization':'L2; concatenate active blocks divided by sqrt(active_blocks)','q_width':2560}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text(value):
 require(isinstance(value,str) and 0<len(value.strip())<=6000 and len(value)<=6000,'Invalid assistant draft');value.encode('utf-8')
 return re.sub(r'\s+',' ',unicodedata.normalize('NFC',value).casefold()).strip()
def counts(draft,kind):
 value=text(draft);result=Counter()
 if kind=='word':
  words=re.findall(r'\w+',value,flags=re.UNICODE)
  for n in SETTINGS['word_orders']:
   for i in range(len(words)-n+1):result[str(n)+':'+ ' '.join(words[i:i+n])]+=1
 elif kind=='char':
  for n in SETTINGS['char_orders']:
   for i in range(len(value)-n+1):result[str(n)+':'+value[i:i+n]]+=1
 else:raise ValueError('Unknown text channel')
 return result

def fit_lexicon(train_drafts):
 require(isinstance(train_drafts,list) and len(train_drafts)>0,'Empty TRAIN vocabulary population');result={'format':'roguard-learned-response-lexicon-1','settings':json.loads(canonical(SETTINGS)),'train_rows':len(train_drafts),'channels':{}}
 for kind in ('word','char'):
  df=Counter();tf=Counter()
  for draft in train_drafts:
   c=counts(draft,kind);df.update(c.keys());tf.update(c)
  eligible=[w for w in df if df[w]>=SETTINGS['minimum_document_frequency']]
  chosen=sorted(eligible,key=lambda w:(-tf[w],w))[:SETTINGS[kind+'_maximum_features']];chosen=sorted(chosen)
  result['channels'][kind]={'vocabulary':chosen,'idf':[math.log((1+len(train_drafts))/(1+df[w]))+1 for w in chosen]}
 return result

def validate_lexicon(lex):
 require(lex['format']=='roguard-learned-response-lexicon-1' and canonical(lex['settings'])==canonical(SETTINGS) and type(lex['train_rows']) is int and lex['train_rows']>0 and set(lex['channels'])=={'word','char'},'Wrong lexical specification')
 for kind in ('word','char'):
  c=lex['channels'][kind];v=c['vocabulary'];idf=c['idf'];require(isinstance(v,list) and v==sorted(set(v)) and len(v)<=SETTINGS[kind+'_maximum_features'] and len(idf)==len(v),'Invalid vocabulary inventory')
  require(all(isinstance(w,str) and w for w in v) and all(type(x) in (int,float) and math.isfinite(x) and x>=1 for x in idf),'Invalid lexical values')
 return lex

def entries(draft,kind,channel):
 lookup={w:i for i,w in enumerate(channel['vocabulary'])};values=[]
 for word,n in counts(draft,kind).items():
  if word in lookup:
   i=lookup[word];values.append((i,(1+math.log(n))*channel['idf'][i]))
 values.sort();norm=math.sqrt(sum(v*v for _,v in values))
 return [(i,v/norm) for i,v in values] if norm else []

def text_matrix(drafts,kind,channel):
 import numpy as np
 from scipy.sparse import csr_matrix
 # Build lookup once per batch rather than once per row.
 lookup={w:i for i,w in enumerate(channel['vocabulary'])};indices=[];data=[];indptr=[0]
 for draft in drafts:
  vv=sorted((lookup[w],(1+math.log(n))*channel['idf'][lookup[w]]) for w,n in counts(draft,kind).items() if w in lookup);norm=math.sqrt(sum(v*v for _,v in vv))
  if norm:
   indices.extend(i for i,v in vv);data.extend(v/norm for i,v in vv)
  indptr.append(len(data))
 return csr_matrix((np.asarray(data,np.float64),np.asarray(indices,np.int32),np.asarray(indptr,np.int32)),shape=(len(drafts),len(lookup)))

def normalized_q(q,np):
 require(isinstance(q,np.ndarray) and q.ndim==2 and q.shape[1]==2560 and q.dtype==np.float32 and np.isfinite(q).all(),'Invalid frozen response query features')
 x=q.astype(np.float64);norm=np.sqrt(np.sum(x*x,axis=1,keepdims=True));require(np.isfinite(norm).all() and (norm>0).all(),'Invalid query feature norm');return x/norm

def matrix(q,drafts,lex,mode):
 import numpy as np
 from scipy.sparse import csr_matrix,hstack
 require(mode in MODES and len(q)==len(drafts),'Wrong arm or feature rows');validate_lexicon(lex);blocks=[]
 if mode in ('hybrid','q_only'):blocks.append(csr_matrix(normalized_q(q,np)))
 if mode in ('hybrid','text_only'):
  blocks.extend(text_matrix(drafts,kind,lex['channels'][kind]) for kind in ('word','char'))
 return hstack(blocks,format='csr')/math.sqrt(len(blocks))

def width(lex,mode):return (2560 if mode!='text_only' else 0)+(sum(len(lex['channels'][k]['vocabulary']) for k in ('word','char')) if mode!='q_only' else 0)

def portable_logits(q,draft,lex,mode,weights,bias):
 """One-row scorer uses only NumPy/stdlib; no scipy/sklearn requirement."""
 import numpy as np
 require(mode in MODES and isinstance(q,np.ndarray) and q.shape==(1,2560),'Wrong response arm or single query shape');validate_lexicon(lex);require(weights.dtype==bias.dtype==np.float64 and weights.shape==(5,width(lex,mode)) and bias.shape==(5,) and np.isfinite(weights).all() and np.isfinite(bias).all(),'Invalid exported five-way readout')
 z=bias.copy();offset=0;nblocks=3 if mode=='hybrid' else (1 if mode=='q_only' else 2);scale=1/math.sqrt(nblocks)
 if mode!='text_only':
  x=normalized_q(q,np)[0];z+=weights[:,:2560]@x*scale;offset=2560
 if mode!='q_only':
  for kind in ('word','char'):
   for i,v in entries(draft,kind,lex['channels'][kind]):z+=weights[:,offset+i]*(v*scale)
   offset+=len(lex['channels'][kind]['vocabulary'])
 require(np.isfinite(z).all(),'Nonfinite portable logits');return z

def probabilities(z):
 import numpy as np
 require(z.shape==(5,) and np.isfinite(z).all(),'Invalid five-way logits');p=np.exp(z-z.max());return p/p.sum()
