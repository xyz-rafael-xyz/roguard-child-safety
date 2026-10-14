# Conversation evidence and permission-aware tutoring

The new context workflow is **experimental**. It proposes interpretations of a bounded Romanian or Ukrainian conversation, links them to turn references, and connects a host-owned transcript to existing permission controls. It does not verify what happened or establish legal authority.

## Install and open

For the jointly trained encoder and two prediction heads, use Python 3.10+:

```sh
pip install --upgrade 'roguard-child-safety[adapted]'
roguard-model fetch
roguard-studio --context --context-backend joint --open
```

The pinned base download is approximately 1.2 GB. Inference is local. The adapter is bundled. The browser offers Romanian and Ukrainian controls, editable messages, reference navigation, explicit uncertainty, clear/reset, and a metadata-only report download. It does not store a transcript in browser storage or execute external actions.

The `semantic` context backend uses the frozen base with fitted heads; `joint` additionally adapts attention through LoRA. The dependency-free `character` backend is a weaker comparison model, available explicitly with `--context-backend character`. These **context** backend names are separate from the existing conversation API's `adapted` backend.

CLI:

```sh
roguard-context examples/context_ro.json --backend joint
roguard-context examples/context_uk.json --backend joint
```

The example paths are in the repository; installed users can supply their own JSON file containing `language` and `turns`, with `role` and `text` per turn. Supported declared roles are `minor`, `assistant`, `adult`, `other`, `unknown`. Roles are caller declarations, not authenticated identities. Only minor turns receive evidence proposals; all earlier turns remain context.

## What is predicted

Two tasks are trained together: the conversational frame and whether a support concern is mentioned. A frame alone is not a risk label.

| Frame | Candidate interpretation |
| --- | --- |
| `SELF_CURRENT` | A current first-person account |
| `SELF_PAST` | A past first-person account |
| `OTHER_PERSON` | A statement about another person |
| `QUOTED` | A quotation, fictional passage or role |
| `CORRECTION` | A revision of an earlier interpretation |
| `UNCERTAIN` | Expressed uncertainty about the account |
| `ORDINARY` | Ordinary content without those contextual distinctions |

A positive mention in a current/past/third-person account recommends review. An unresolved mention also recommends review. A negative mention does not require review merely because its frame is ambiguous. A later correction never automatically dismisses an earlier review recommendation.

Each candidate includes a zero-based focus-turn index, Unicode code-point range in the original text for the **entire turn**, preceding context indices, prediction sets and a reviewer question. These are candidate references, not verified evidence spans, causal explanations, or extracted personal identities. Scores are not individual safety probabilities. Calibration is pooled within language, not class-conditional, and does not establish coverage on real child messages.

The hard limits are 12 turns, 2,000 characters per turn and 6,000 characters total. Semantic inference also has a 256-token bound per full prefix. Unsupported prefixes are reported explicitly; they are not truncated. Latin-only Ukrainian transliteration is currently unsupported and requests review. Code-switching, spelling errors and longer contexts do not have an established performance guarantee.

## Integrate a tutoring workflow

`ContextSession` holds an ephemeral transcript and a `FlowLedger`. Appending a turn registers the complete turn as a source and returns the evidence report and changes since the previous report. The session is owned by the trusted host; it is not a durable storage or authorization service.

```python
from roguard.context import ContextModel
from roguard.context_session import ContextSession

model = ContextModel('joint')
with ContextSession('ro', model) as session:
    session.append('minor', 'Am rezolvat exercițiul: 4 + 3 = 7.')
    review = session.append('minor', 'Mă simt singur și aș vrea să vorbesc cu cineva.')

    # An explicit host policy declaration, never inferred from model output.
    session.ledger.permission('turn_0', 'tutoring', 'teacher', 'export', 'grant')
    proposal = session.propose_summary(
        [0, 1], {'0': ['learning'], '1': []}, ['learning'],
        purpose='tutoring', recipient='teacher', draft_id='learning_summary')
```

The host reviews the coverage mapping. The planner selects permitted complete turns that cover the declared task, and the optional draft concatenates their exact text. It does not generate new prose, verify semantic sufficiency, redact a sensitive fragment or automatically declassify an excerpt. A turn mixing lesson and private material stays one source. Every derived draft retains its selected parents; revocation continues to apply. Nothing is sent by this workflow.

[Run the complete fictional tutoring example](../examples/context_tutor.py), including an allow check followed by revocation and a block. The example uses the character baseline for a dependency-free integration demonstration; pass an explicit `ContextModel('joint')` for the trained semantic model. Integration correctness is distinct from language-model accuracy.

For actual delivery, use the existing [authenticated durable gateway](RELIABILITY.md). Recheck live scope and permissions at dispatch; a proposal or an old model report is not an execution capability. The host must separately approve the action and configure the trusted callback. Existing gateway receipts record status without storing another copy of message text in monitoring.

## Authenticated analysis endpoint

`GatewayApp(..., context_model=ContextModel('joint'))` enables `POST /v1/analyze-context`. The CLI equivalent is `roguard-gateway ... --context-backend joint`. Without an explicit context model, this endpoint is disabled.

It uses the existing bearer-token, tenant, body-size and concurrency boundaries. Any authenticated reader may analyze. Requests have the same `language` / `turns` shape as the CLI. Responses contain references and metadata rather than raw messages. Optional monitoring records model identity, language, decision and latency; unresolved mentions are counted as insufficient evidence. It never stores the transcript through the monitoring API.

## Evidence and limitations

See [the context release report](V7_RELEASE.md) for the frozen comparisons, calibration failures, training revisions and generator-quality problems. All training and evaluation examples are fictional. Translation and new label quality have not received a new independent native-speaker review. The model is an optional research aid, not a diagnosis, an emergency detector, or a replacement for a safeguarding workflow.
