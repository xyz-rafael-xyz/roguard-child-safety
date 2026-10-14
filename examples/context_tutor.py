"""Fictional tutoring workflow. No network, real child records or external delivery."""
import json
from roguard.context_session import ContextSession
from roguard.context import ContextModel

with ContextSession('ro', ContextModel('character')) as session:
    first=session.append('minor','Am rezolvat exercițiul: 4 + 3 = 7.')
    second=session.append('minor','Mă simt singur și aș vrea să vorbesc cu cineva.')
    # Explicit host policy: only the learning source is permitted for this scope.
    session.ledger.permission('turn_0','tutoring','teacher','export','grant')
    proposal=session.propose_summary([0,1],{'0':['learning'],'1':[]},['learning'],
        purpose='tutoring',recipient='teacher',draft_id='learning_summary')
    before=session.ledger.check(['learning_summary'],'tutoring','teacher','export')
    session.ledger.permission('turn_0','tutoring','teacher','export','revoke')
    after=session.ledger.check(['learning_summary'],'tutoring','teacher','export')
    print(json.dumps({'review':second['review'],'change':second['change'],'proposal':proposal,
                     'before_revocation':before,'after_revocation':after,'external_action_taken':False},ensure_ascii=False,indent=2))
