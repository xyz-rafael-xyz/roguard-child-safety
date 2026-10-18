# V9 alpha: decision assurance

Version **9.0.0a1** adds a local assurance workflow to the existing toolkit. It is
an alpha, not stable v9.0. The language-model promotion gates remain open. The
existing models are retained; this release does not claim better text accuracy.
[Acceptance plan](V9_PLAN.md) · [Previous model experiments](CONTEXTUAL_MODEL_STUDY.md).

The current source checkout additionally provides an experimental local
six-field interpreter and a report-to-planner bridge. The bridge verifies the
conversation, focal turn, configured model identities and interpretation
contract before constructing a joint-hypothesis plan. Invalid output preserves
all possible interpretations. These additions are undergoing installed-runtime
validation and are not part of the published alpha. See the
[interpreter setup and integration guide](INTERPRETER_RUNTIME.md).

## What changes

- **Joint hypotheses:** preserve combinations proposed by a model or reviewer.
  For example, “fictional + fiction” and “self + direct” need not create the
  unsupported combinations “fictional + direct” and “self + fiction”.
- **Model disagreement:** `combine()` unions complete hypotheses from matching
  contexts. It never silently intersects disagreeing predictions. Model digests
  remain in the resulting certificate; a review floor cannot be removed by fusion.
- **Adaptive clarification:** choose the complete question tree with minimum
  worst-case total declared cost, rather than only the next greedy question.
  Stop as soon as every remaining interpretation has the same policy action.
- **Recomputation:** bind results to the structured input, source-context digest,
  model digests, and exhaustive policy-table digest. Verification recomputes the
  entire tree and cost. Changed answers must cite the exact parent certificate.
- **Unexpected answers:** an answer outside the proposed set requests
  reinterpretation. It cannot clear a concern or authorize an external action.

The solver is exact within a finite supplied problem, with at most 512 hypotheses
(432 distinct worlds in the current ontology), six question fields and positive
integer costs from 1 to 1000. Omitted question fields are unavailable. If no
available question sequence resolves the action, the result is `UNRESOLVED`.

## Try it

Install the available preview with `pip install --upgrade --pre roguard-child-safety`.
From a repository checkout, run the included fictional examples:

```sh
roguard-assure examples/assurance_ro.json > certificate.json
roguard-assure certificate.json --verify
roguard-assure examples/assurance_uk.json
roguard-assure examples/evidence_ro.json --conversation
roguard-assure examples/evidence_uk.json --conversation
```

The conversation mode invokes the actual existing evidence API. Each entry has
its original source reference and an assurance certificate. The existing API
supplies Cartesian attribute sets; this adapter does not pretend they are learned
joint distributions. Supply complete joint hypotheses directly when available.

The result's `tree.question` is the next field to clarify; `tree.label` is a
Romanian or Ukrainian reviewer prompt and `tree.branches` lists its proposed
values. These are project-authored interface labels, not a new independent
linguistic review. `optimal_worst_case_cost` is declared reviewer effort, not a
probability or measured human time. `ACTION_DETERMINED` means policy agreement
within the supplied set; it does not establish safety.

## Integrate review

```python
from roguard.assurance import assure_conversation, answer, verify

payload = {"language": "ro", "turns": [
    {"role": "minor", "text": "Am nevoie de ajutor la matematică."}
]}
report = assure_conversation(payload)
certificate = report["entries"][0]["certificate"]
verify(certificate, expected_input_digest=certificate["input_digest"])
if certificate["status"] == "CLARIFICATION_REQUIRED":
    field = certificate["tree"]["question"]
    # An authenticated reviewer supplies a value; do not infer it from this loop.
    # result = answer(certificate, {field: reviewed_value},
    #                 expected_certificate_digest=certificate["certificate_digest"])
```

In a host application, retain the expected digests separately and compare them
against the current source/model state. A certificate supplied by an untrusted
caller must not supply its own trusted binding. The snippet above demonstrates
local recomputation only. CLI review accepts an answers JSON object through
`--answers answers.json --binding PARENT_CERTIFICATE_DIGEST`; `--combine` accepts
a JSON array of certificates for the same context, language and question costs.

Certificates are **not signed attestations**, proof of reviewer identity, legal
authority, or calibrated coverage. An attacker who replaces both inputs and a
certificate can create a different valid computation. Hashes do not conceal
guessable source text. Keep reports under the host's access and retention policy.
No API here sends messages, grants permission, or executes callbacks.

## Evaluation and release gates

`eval/run_assurance_v9.py` generates reproducible structural problems and checks
every supplied world against its tree leaf. Tests independently enumerate the
432-state policy and compare optimal cost against a separate brute-force solver
on small problems. Additional tests cover input permutation, duplicate worlds,
tampering, context changes, source-model bindings, out-of-set answers, unavailable
questions, malformed JSON, resource limits, and the real bilingual evidence API.

Two seeds cover **20,000 structural cases** and **178,586
world paths**, with zero leaf or cost mismatches. Optimal plans improve over the
greedy baseline in **3,338 cases**, are worse in none,
and reduce aggregate declared worst-case cost by **7.71%**.
The full software suite passes 479 tests, with 14 private-artifact tests skipped.
Results and source hashes live in `research/v9-assurance/`. Generated structural
worlds are not child messages. Cost comparisons use an explicitly defined
cost-aware, one-step action-ambiguity greedy baseline. They are neither a reviewer
usability study nor evidence that the hypothesis set includes the truth.

Stable language-model promotion still requires >=99% concern recall and <=5%
benign review per language and formal/conversational group, then reserved fresh
evaluation and integrated runtime checks. V9 does not remove those requirements.

## Research position

The contribution is the implemented connection between preserved interpretation
correlations, adaptive policy clarification, reviewer bindings and local replay.
Optimal decision trees, provenance and structured uncertainty are established
techniques. See [structured prediction sets](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1868a3c73d0d2a44c42458575fa8514c-Abstract-Conference.html)
and [OPA decision logs](https://www.openpolicyagent.org/docs/management-decision-logs).
No first-of-kind, conformal coverage, real-world accuracy or Eastern European
market-leadership claim is made.

## Localized answer choices in the source checkout

The current checkout includes a display helper for Romanian and Ukrainian
reviewers. It supplies localized questions and answer labels while retaining
the certificate's exact branch codes and parent binding:

```python
from roguard.review_labels import question_view

view = question_view(certificate)
if view is not None:
    print(view["prompt"])
    print(view["note"])
    for option in view["options"]:
        print(option["value"], option["label"])
    # After an authenticated reviewer chooses an answer:
    # result = answer(certificate, {view["field"]: selected_value},
    #                 expected_certificate_digest=view["certificate_digest"])
```

The helper verifies the certificate before displaying any options. It never
changes a hypothesis, machine code, policy decision or certificate. Labels refer
to what the message expresses; truth remains unverified. `UNCERTAIN` describes
uncertainty expressed in the message. A reviewer who cannot decide should leave
the question unanswered. Concern-content labels include denied or fictional
content, with assertion and source handled by their separate fields.

These are project-authored interface labels, not a new independent linguistic
review. Four tests cover bilingual code coverage, exact branch/binding retention,
unchanged certificates and tampered-input rejection. The helper is currently a
source-checkout addition, not part of the published alpha.
