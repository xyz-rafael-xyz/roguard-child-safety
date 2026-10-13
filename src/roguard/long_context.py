"""Bounded overlapping context windows; no silent truncation or summary memory."""
from .conversation import default_scorer,language_uncertainty,render
from .uncertainty import probability


def analyze_long_conversation(payload, scorer=None, *, window_chars=900, overlap=1):
    if not isinstance(payload,dict) or set(payload)-{'language','turns'}:raise ValueError('Expected language and turns')
    lang=payload.get('language');turns=payload.get('turns')
    if lang not in ('ro','uk') or not isinstance(turns,list) or not 1<=len(turns)<=128:raise ValueError('Invalid conversation')
    if type(window_chars) is not int or not 128<=window_chars<=1800 or overlap!=1:raise ValueError('Invalid window configuration')
    for t in turns:
        if not isinstance(t,dict) or set(t)!={'role','text'} or t['role'] not in ('minor','assistant') or not isinstance(t['text'],str) or not 1<=len(t['text'])<=8000:
            raise ValueError('Unsupported role or turn')
    if sum(len(t['text']) for t in turns)>64000:raise ValueError('Conversation exceeds 64000 characters')
    reasons=language_uncertainty(turns,lang)
    if not any(t['role']=='minor' for t in turns):reasons.append('NO_DECLARED_MINOR_TURN')
    scorer=scorer or default_scorer();pieces=[]
    for index,t in enumerate(turns):
        for start in range(0,len(t['text']),window_chars//2):
            end=min(len(t['text']),start+window_chars//2)
            pieces.append(dict(role=t['role'],text=t['text'][start:end],turn_index=index,start=start,end=end))
    windows=[];at=0
    while at<len(pieces):
        end=at;size=0
        while end<len(pieces) and size+len(pieces[end]['text'])+12<=window_chars:
            size+=len(pieces[end]['text'])+12;end+=1
        end=max(at+1,end);windows.append(pieces[at:end]);at=end if end==len(pieces) else max(at+1,end-1)
    if len(windows)>192:raise ValueError('Too many windows')
    results=[]
    for index,parts in enumerate(windows):
        row={'window':index,'references':[{k:p[k] for k in ('turn_index','start','end')} for p in parts]}
        if reasons:row.update(decision='INSUFFICIENT_EVIDENCE',reason_codes=reasons,score=None)
        else:
            try:
                score=probability(scorer.scores([render(parts)],lang)[0]);setting=scorer.thresholds(lang)
                decision='INSUFFICIENT_EVIDENCE' if abs(score-setting['cutoff'])<=setting['abstain_margin'] else 'REVIEW_SUPPORT_SIGNAL' if score>=setting['cutoff'] else 'NO_MODEL_SIGNAL'
                row.update(decision=decision,score=score,reason_codes=[])
            except ValueError:row.update(decision='INSUFFICIENT_EVIDENCE',score=None,reason_codes=['WINDOW_INPUT_UNSUPPORTED'])
        results.append(row)
    unresolved=sum(r['decision']=='INSUFFICIENT_EVIDENCE' for r in results)
    positive=any(r['decision']=='REVIEW_SUPPORT_SIGNAL' for r in results)
    # Window negatives cannot exclude a concern dependent on distant context.
    decision='REVIEW_SUPPORT_SIGNAL' if positive else 'INSUFFICIENT_EVIDENCE'
    return dict(schema_version='6.0',language=lang,decision=decision,windows=results,
                windows_scored=sum(r['score'] is not None for r in results),unresolved_windows=unresolved,
                text_coverage_complete=True,semantic_coverage_complete=all(r['score'] is not None for r in results),
                aggregate_calibrated=False,negative_windows_do_not_close_concern=True,
                cross_window_reasoning_verified=False,text_in_report=False,authority_from_model=False)
