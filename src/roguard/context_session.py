"""Host-integrated review sessions with exact extractive summary proposals."""
from copy import deepcopy
from .context import analyze_context, compare_context, validate
from .flow import FlowLedger, identifier
from .repair import repair_proposal

class ContextSession:
    """Ephemeral transcript owned by the host; no implicit grants or callbacks.

    The host is responsible for identity, retention and coverage declarations.
    Each complete turn is a source: selecting an excerpt never declassifies it.
    """
    def __init__(self,language,model=None):
        if language not in ('ro','uk'):raise ValueError('Unsupported language')
        self.language=language;self.model=model;self.ledger=FlowLedger()
        self._turns=[];self._report=None;self._closed=False

    def append(self,role,text):
        if self._closed:raise ValueError('Closed review session')
        proposed=self._turns+[{'role':role,'text':text}]
        payload={'language':self.language,'turns':proposed}
        validate(payload)
        report=analyze_context(payload,self.model)
        artifact='turn_'+str(len(self._turns))
        self.ledger.add_source(artifact,text)
        delta=compare_context(self._report,report) if self._report else {'added_turn_indices':[e['turn_index'] for e in report['evidence']], 'changed_turn_indices':[], 'new_review_turn_indices':report['review_turn_indices'],'external_action_taken':False}
        self._turns=proposed;self._report=report
        return {'source_id':artifact,'review':deepcopy(report),'change':delta}

    def propose_summary(self,turn_indices,reviewed_coverage,requirements,*,purpose,recipient,operation='export',draft_id=None):
        """Host-reviewed coverage is a declaration, never inferred from model labels.

        Returns metadata. A draft, if requested, is registered in the same ledger
        with all selected parents. No external delivery or permission change.
        """
        if self._closed:raise ValueError('Closed review session')
        if not isinstance(turn_indices,list) or not turn_indices or len(turn_indices)>12 or len(set(turn_indices))!=len(turn_indices):raise ValueError('Invalid turn selection')
        if any(type(i) is not int or not 0<=i<len(self._turns) for i in turn_indices):raise ValueError('Unknown turn')
        if not isinstance(reviewed_coverage,dict) or set(reviewed_coverage)!={str(i) for i in turn_indices}:raise ValueError('Host-reviewed coverage required for each selected turn')
        candidates=[{'artifact_id':'turn_'+str(i),'covers':reviewed_coverage[str(i)],'cost':len(self._turns[i]['text'])} for i in turn_indices]
        plan=repair_proposal(self.ledger,[c['artifact_id'] for c in candidates],candidates,requirements,purpose=purpose,recipient=recipient,operation=operation)
        artifact=None
        if draft_id is not None:
            identifier(draft_id,'draft')
            if plan['status']=='REPAIR_PROPOSED':
                text='\n'.join(self._turns[int(source.split('_')[1])]['text'] for source in plan['selected'])
                self.ledger.derive(draft_id,text,plan['selected'],kind='summary')
                artifact=draft_id
        return {'proposal':plan,'draft_artifact_id':artifact,'summary_method':'exact_selected_turns',
                'model_granted_permission':False,'execution_authorized':False,'external_action_taken':False,
                'coverage_source':'host_reviewed_declaration','text_in_report':False}

    def close(self):
        self.ledger.close();self._turns.clear();self._report=None;self._closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
