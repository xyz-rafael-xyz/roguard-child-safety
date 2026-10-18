"""Review mass under independent interpretation marginals and the fixed policy.

This is a differentiable policy calculation, not a calibrated confidence or
proof that the estimated marginals match the text. Arithmetic also accepts
Torch tensors; no Torch dependency is required for reference validation.
"""


def review_mass(p):
    subject_fiction = p['subject']['FICTIONAL']
    source_fiction = p['source']['FICTION']
    negated = p['assertion']['NEGATED']
    retracted = p['assertion']['RETRACTED']
    retracts = p['relation']['RETRACTS']
    mismatch = subject_fiction*(1-source_fiction)+(1-subject_fiction)*source_fiction
    fiction_review = retracted+retracts-retracted*retracts
    nonfiction_review = 1-negated*(1-retracts)
    return p['concern']['1']*(mismatch+subject_fiction*source_fiction*fiction_review+
                              (1-subject_fiction)*(1-source_fiction)*nonfiction_review)
