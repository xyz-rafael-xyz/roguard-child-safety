"""Differentiable form of the existing exact factorized review policy."""
from roguard.evidence import ONTOLOGY
from training.native_fields_v9 import FIELDS, SIZES


def policy_mass(torch, logits, couplings=None):
    if logits.ndim != 2 or logits.shape[1] != sum(SIZES):
        raise ValueError('Expected six semantic logit groups')
    values = [torch.softmax(part, dim=-1) for part in torch.split(logits, SIZES, dim=-1)]
    def p(field, label):
        return values[FIELDS.index(field)][:, ONTOLOGY[field].index(label)]
    if couplings is not None:
        pairs = {}
        for name, left, right in (('scope', 'subject', 'source'), ('assertion_relation', 'assertion', 'relation')):
            weights = torch.as_tensor(couplings[name], device=logits.device, dtype=logits.dtype)
            if weights.shape != (len(ONTOLOGY[left]), len(ONTOLOGY[right])):
                raise ValueError('Invalid coupling dimensions')
            joint = values[FIELDS.index(left)][:, :, None]*values[FIELDS.index(right)][:, None, :]*weights[None, :, :]
            pairs[name] = joint/joint.sum(dim=(1, 2), keepdim=True)
        scope, ar = pairs['scope'], pairs['assertion_relation']
        sf, fi = ONTOLOGY['subject'].index('FICTIONAL'), ONTOLOGY['source'].index('FICTION')
        neg, ret, new = ONTOLOGY['assertion'].index('NEGATED'), ONTOLOGY['assertion'].index('RETRACTED'), ONTOLOGY['relation'].index('NEW')
        mismatch = sum(scope[:, i, j] for i in range(scope.shape[1]) for j in range(scope.shape[2]) if (i == sf) != (j == fi))
        real = sum(scope[:, i, j] for i in range(scope.shape[1]) for j in range(scope.shape[2]) if i != sf and j != fi)
        fiction_review = 1-sum(ar[:, i, new] for i in range(ar.shape[1]) if i != ret)
        return (p('concern', '1')*(mismatch+scope[:, sf, fi]*fiction_review+real*(1-ar[:, neg, new]))).clamp(0, 1)
    fictional, fiction = p('subject', 'FICTIONAL'), p('source', 'FICTION')
    negated, retracted, retracts = p('assertion', 'NEGATED'), p('assertion', 'RETRACTED'), p('relation', 'RETRACTS')
    mismatch = fictional*(1-fiction)+(1-fictional)*fiction
    fiction_review = retracted+retracts-retracted*retracts
    return (p('concern', '1')*(mismatch+fictional*fiction*fiction_review+(1-fictional)*(1-fiction)*(1-(1-retracts)*negated))).clamp(0, 1)
