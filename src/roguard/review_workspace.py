"""Stateless local review display; recomputation is not reviewer authentication."""
from .assurance import assure, answer, verify
from .review_labels import question_view


def review_workspace(payload):
    if not isinstance(payload,dict):raise ValueError('Expected a review request')
    operation=payload.get('operation')
    if operation=='plan' and set(payload)=={'operation','problem'}:
        certificate=assure(payload['problem']);result={'status':'PLAN_CREATED'}
    elif operation=='verify' and set(payload)=={'operation','certificate'}:
        certificate=payload['certificate'];verify(certificate);result={'status':'PLAN_RECOMPUTED'}
    elif operation=='answer' and set(payload)=={'operation','certificate','answers','binding'}:
        result=answer(payload['certificate'],payload['answers'],expected_certificate_digest=payload['binding'])
        certificate=result.get('certificate')
    else:
        raise ValueError('Invalid review operation or fields')
    return dict(format='roguard-local-review-workspace-v1',result=result,certificate=certificate,
        question=question_view(certificate) if certificate is not None else None,
        reviewer_authenticated=False,permits_external_action=False,permits_notification=False,
        interpretation_truth_verified=False)
