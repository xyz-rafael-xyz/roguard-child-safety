# V9 decision assurance

RoGuard **9.0.0** includes a local assurance workflow, bilingual clarification
labels, a local review workspace and a report-to-planner bridge. The new optional
review model has its own [qualification and setup guide](V9_MODEL.md).
[Acceptance plan](V9_PLAN.md) · [Previous experiments](CONTEXTUAL_MODEL_STUDY.md).

The retained six-field interpreter remains experimental. Its bridge verifies
the conversation, focal turn, configured model identities and interpretation
contract before constructing a joint-hypothesis plan. Invalid output preserves
all possible interpretations. Qualifying the new binary review model does not
establish accuracy for every interpreter field or older model backend.
See the [interpreter integration guide](INTERPRETER_RUNTIME.md).

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

Install the stable release with `pip install --upgrade roguard-child-safety`.
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
The original assurance-alpha software suite passed 479 tests, with 14 private-artifact tests skipped; these counts describe that recorded study.
Results and source hashes live in `research/v9-assurance/`. Generated structural
worlds are not child messages. Cost comparisons use an explicitly defined
cost-aware, one-step action-ambiguity greedy baseline. They are neither a reviewer
usability study nor evidence that the hypothesis set includes the truth.

The optional v9 review model passed the unchanged >=99% concern recall and
<=5% benign-review gates, followed by reserved evaluation and installed-runtime
checks. See the [separate model record](V9_MODEL.md); the assurance solver itself
does not establish language-model accuracy.

## Research position

The contribution is the implemented connection between preserved interpretation
correlations, adaptive policy clarification, reviewer bindings and local replay.
Optimal decision trees, provenance and structured uncertainty are established
techniques. See [structured prediction sets](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1868a3c73d0d2a44c42458575fa8514c-Abstract-Conference.html)
and [OPA decision logs](https://www.openpolicyagent.org/docs/management-decision-logs).
No first-of-kind, conformal coverage, real-world accuracy or Eastern European
market-leadership claim is made.

## Localized answer choices

V9 includes a display helper for Romanian and Ukrainian
reviewers. It supplies localized questions and answer labels while retaining
the certificate's exact branch codes and parent binding:

```python
from roguard.review_labels import question_view

print(certificate["status"])
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

A `None` view means there is no available next question. Inspect the certificate
status to distinguish `ACTION_DETERMINED` from `UNRESOLVED`; neither grants
permission for an external action.

These are project-authored interface labels, not a new independent linguistic
review. Four tests cover bilingual code coverage, exact branch/binding retention,
unchanged certificates and tampered-input rejection. The helper is included in v9.0.0.

## Local clarification workspace

After installing RoGuard, open the local review UI:

```sh
python -m roguard.studio --review --open
```

Use the built-in fictional example, import a standalone assurance certificate,
or paste a structured problem. The page recomputes the plan, shows Romanian or
Ukrainian answer choices, and updates the remaining interpretations after each
answer. You can leave a question unanswered, restart, or export the current
certificate. Editing the input invalidates the displayed result.

The example's interpretations are supplied manually, not predicted by a model.
Import a certificate from a host integration to examine its actual hypotheses.
The interface does not establish the truth of those hypotheses or authenticate
the reviewer. It sends no notifications and grants no permissions. Source and
model bindings still need to be checked by the host. The local server does not
persist submitted data. No new language-model accuracy result is implied.
