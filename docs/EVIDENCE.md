# Experimental evidence workspace

This workflow proposes separate interpretations for subject, time, assertion,
source, relation and concern mention. It exposes ambiguity and candidate text
references. It does not verify a disclosure, establish legal authority, or
notify anyone.

## Try it locally

```sh
pip install --upgrade --pre roguard-child-safety
roguard-studio --evidence --open
roguard-evidence examples/evidence_ro.json
roguard-evidence examples/evidence_uk.json
```

The CLI example paths refer to a repository checkout; the browser workspace needs no example files.

The core character model runs locally without optional runtime dependencies.
It is useful for trying the interface but performed materially worse on the
prose probes; it is not an accuracy-equivalent substitute for the adapted model.
The guarded evidence mode preserves the released context workflow’s review
signals alongside the experimental factor model. It requires the adapted extra
and the existing pinned base model:

```sh
pip install --upgrade --pre 'roguard-child-safety[adapted]'
roguard-model fetch
roguard-studio --evidence --evidence-backend guarded --open
roguard-evidence examples/evidence_ro.json --backend guarded
```

The optional `--evidence-cascade` / `--cascade` mode performed worse than the
direct adapted model on the prospective probe and is retained for comparison.
Its challenger is invoked only when the primary model leaves an attribute
unresolved. A decided conflict remains unresolved. An overlapping singleton can
resolve an ambiguous primary prediction set. Neither model is an oracle. The
combined cascade has no established conformal coverage guarantee.

Guarded mode evaluates both models. It keeps their interpretations separate and
requests review when either workflow requests it; it cannot reduce the released
workflow's review workload on the same input. It does not grant authority or
verify a concern. `--backend semantic` (Studio: `--evidence-backend semantic`)
runs the factor model alone for experiments, with known missed concerns.

```python
from roguard import EvidenceSession, load_evidence_model

model = load_evidence_model("guarded")
with EvidenceSession("uk", model) as session:
    update = session.append("minor", "Мені потрібно з кимось поговорити.")
```

## References and context

The input uses the same `language` and `turns` schema as `roguard-context`.
Limits: 12 turns, 2,000 characters per turn, 6,000 characters total, and 48
nonempty minor paragraphs. Original Unicode code-point offsets are retained;
the browser converts them correctly to JavaScript selection offsets.

Each paragraph is a proposed evidence unit. Text inside French-style quotation
marks receives an additional candidate claim reference. This is a deterministic
boundary heuristic, not learned or verified predicate extraction. The character
model reads the paragraph. A contextual semantic artifact reads its bounded
conversation prefix as well as the focal paragraph. The semantic token limit is
256; exceeding it produces an explicit unresolved result rather than truncation.
Latin-only Ukrainian remains unsupported.

Matching wording and predicted subject/time/source may propose repeated,
contradictory or retracted claim links. These are candidate edges. Matching
wording does not establish that two statements describe the same real event.
No candidate link deletes or dismisses earlier review signals.

## Host-reviewed revisions

`EvidenceState.update(report)` requires append-only evidence for a fixed model
and language. `confirm_revision(new_id, old_id, reviewer_id=...)` changes the
active review view while retaining original evidence and the review record.
The host must authenticate and authorize its reviewer. This Python method must
not be exposed as a model tool. It does not authenticate a reviewer itself.
`close()` drops the session's retained report and revision metadata.

## Response proposals and permissions

`plan_response(report, candidates, ledger, purpose=..., recipient=...)` compares
host-declared alternatives. Each candidate supplies artifact IDs, coverage of
`task` / `support` / `review`, and an integer disruption cost. The planner checks
current source permissions and required coverage, then chooses the eligible
candidate with the lowest declared cost. Ties use the candidate ID.

This is an exact check of **declared** coverage and costs. It does not generate a
response or prove that its prose is supportive. It cannot grant permission or
execute an action. Permissions must be checked again immediately before actual
execution through the existing host gateway. See `examples/evidence_tutor.py`.

## Integration

```python
from roguard.evidence import EvidenceCascade, EvidenceState, analyze_evidence

model = EvidenceCascade()
report = analyze_evidence(payload, model)
state = EvidenceState()
state.update(report)
```

The authenticated gateway exposes `/v1/analyze-evidence` only when configured
with an explicit `evidence_model`. CLI flags are `--evidence` and
`--evidence-cascade`. The endpoint returns interpretation metadata without the
conversation text. The local Studio uses the existing origin/token protections.
It has no review-confirmation endpoint and no automatic external notifications.

## Interpretation limits

Training and calibration use project-authored fictional scenarios. The large
row count consists of combinations of a much smaller number of content families
and phrasing choices. Source and subject are correlated in this first training
study. The models can misinterpret informal language, ambiguous scope, mixed
subjects, quotation, negation and topic transitions. Independent native-language
quality and authentic-child accuracy are not established.

Attribute calibration uses separate language/task/class groups at nominal
alpha 0.05. That value is neither an individual probability of harm nor a
real-world error guarantee. Scenario-family shifts can invalidate calibration
assumptions. See the experiment report for held-out results and release status.

## Decisions under uncertainty

`decision_envelope(attributes)` enumerates the finite attribute hypothesis space.
It reports every reachable response category, one witness per category, and
reviewer attributes ranked by worst remaining action ambiguity after resolution.
For example, unresolved current/past timing need not create an extra alert if
both imply the same supportive review response. An affirmation/negation dispute
can change the action and therefore remains unresolved.

This guarantee is conditional on the supplied hypotheses and policy. A model can
exclude the correct interpretation. The enumeration cannot repair that error or
prove the underlying situation safe. Suggested questions are for the reviewer;
the toolkit does not automatically question a child.

To inspect the semantic model directly instead of the cascade:

```sh
roguard-evidence examples/evidence_ro.json --backend semantic
roguard-studio --evidence --evidence-backend semantic --open
```

The fast primary, semantic model, cascade and guarded mode have different error/latency
tradeoffs. A cascade can increase review workload; it is not automatically the
best configuration. Consult the per-backend results before choosing a deployment
configuration. Existing v7 workflows remain available.

### Mixed adaptation experiment

The follow-up training experiment adapts a rank-4 LoRA encoder and six heads
using 5,000 sampled structured scenarios and 3,000 existing fictional
conversational training examples. Existing context frames provide coarse
subject/time/source supervision for the conversational examples; correction
frames are excluded because they do not uniquely identify those attributes.
This projection is a training assumption, not verified event annotation.
The original controlled and prose evaluations become regression sets for this
iteration. A separate 56-case prose diagnostic is frozen before inspecting the
new candidate's results.

## Integrated sessions

`EvidenceSession(language, model=None)` connects the evidence state and a local
`FlowLedger`. `append(role, text)` records an entire source turn and computes its
paragraph evidence. `derive_from_evidence(id, text, evidence_ids)` creates a
host-provided response artifact that inherits restrictions from every complete
parent turn. Selecting a benign paragraph cannot declassify the rest of its turn.

`session.state.confirm_revision(new_id, old_id, reviewer_id=...)` records an
explicit host-reviewed correction. The host must authenticate that reviewer.
This changes the active interpretation used by `propose_response`, while retaining
history and leaving source permissions intact. No model prediction grants consent.

Run `PYTHONPATH=src python examples/evidence_tutor.py` for a fictional tutoring
session. It first proposes a permitted supportive reply, then revokes source
permission and demonstrates that the derived reply is blocked. Candidate coverage
is declared by the host; this API does not verify that the reply is clinically
appropriate or execute it. Recheck permissions immediately before execution.
The context manager clears the session's in-memory state on exit.

The integrated session memoizes interpretations of immutable conversation
prefixes in memory, up to the same 48-paragraph limit. Appending 12 one-paragraph
messages evaluates 12 units instead of repeatedly evaluating 78 prefixes.
This is an evaluation-count reduction, not a measured 6.5× wall-clock speedup.
Later messages produce new evidence; changing the active interpretation of an
older claim still requires explicit host review. Closing clears this cache too.
