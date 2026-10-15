"""Fictional local session: evidence, permission inheritance, response proposal."""
import json
from roguard.evidence import EvidenceSession

with EvidenceSession('ro') as session:
    update = session.append('minor', 'Mă simt singur. Mă ajuți și cu tema?')
    source = update['source_id']
    # The host obtains permission; model predictions cannot grant it.
    session.ledger.permission(source, 'tutoring', 'minor', 'use', 'grant')
    report = update['state']['report']
    session.derive_from_evidence(
        'reply', 'Putem lucra la temă și putem vorbi despre cum te simți.',
        [item['id'] for item in report['evidence']],
    )
    candidates = [{'id': 'continue_with_support', 'artifacts': ['reply'],
                   'covers': ['task', 'support', 'review'], 'disruption': 1}]
    proposal = session.propose_response(candidates, purpose='tutoring', recipient='minor')
    print(json.dumps({'review_ids': report['review_ids'], 'response_proposal': proposal}, indent=2))
    # Revocation propagates to the derived reply. A prior proposal is not authority.
    session.ledger.permission(source, 'tutoring', 'minor', 'use', 'revoke')
    revoked = session.propose_response(candidates, purpose='tutoring', recipient='minor')
    print(json.dumps({'after_revocation': revoked}, indent=2))
