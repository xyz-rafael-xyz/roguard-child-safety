# Conversation workspace

RoGuard 4 adds a local conversation review workflow. It accepts declared speaker roles, a Romanian or Ukrainian conversation, an optional draft reply, and optional policy facts. It returns an advisory, explicit uncertainty, turn references, descriptive response checks and a separate action review. No function sends a notification, contacts a guardian, stores messages, infers age, or grants authority. Application callers control their own files and logs.

## Start locally

```sh
pip install --upgrade roguard-child-safety
roguard-studio --conversation --open
```

This starts the small, dependency-free **fast** model. It is useful for exercising the workflow, but its measured accuracy is lower than the optional semantic model. The UI states the active backend. Choose Romanian or Ukrainian, load a fictional example, edit the turns, and run the analysis. Add a draft response or a proposed-sharing check when needed. The detailed symbolic workspace remains available at `/`.

For the stronger semantic backend, use Python 3.10 or newer (the tested semantic environment is Python 3.12). The basic package still supports Python 3.9–3.14.

```sh
pip install --upgrade 'roguard-child-safety[semantic]'
roguard-model fetch
roguard-studio --conversation --open --backend semantic
```

The explicit fetch downloads roughly 1.2 GB of pinned mmBERT-base weights once. Inference resolves cached files only; it does not download implicitly or send input text to Hugging Face. A missing or modified base fails with an error. `roguard-model locate` prints the cached directory. `--model-path PATH` selects a verified local copy. `--device mps` or `--device cuda` requests an available accelerator; CPU is the default. The upstream model card at [the pinned revision](https://huggingface.co/jhu-clsp/mmBERT-base/tree/c5955035435e2bf121cde7f3c8863ef52ff35d82) declares MIT licensing. The base weights are not bundled in the Python wheel. RoGuard includes its own fitted classification head and character models, trained only on fictional project data.

The optional `hybrid` backend combines semantic and character scores. It reduced false alerts but missed more positives than standalone semantic inference on the second holdout; it is not described as universally stronger. All three backends use separately frozen Romanian/Ukrainian thresholds. [Measured results](V4_RELEASE.md) include misses, false alerts and abstentions.

## Reading the result

- **Possible support signal:** a reviewer should inspect the conversation. This is not a finding that an event occurred.
- **No model signal:** the model did not flag this input. This cannot close a concern.
- **Insufficient evidence:** the score is near its development threshold, roles or language are outside the evaluated scope, or the optional encoder cannot handle the context within its token limit.

Before semantic encoding, Unicode is normalized to NFC and repeated horizontal whitespace is collapsed. Entirely capitalized lines are converted to sentence case while preserving speaker roles and turn boundaries; this cannot reconstruct proper-name casing. The score is not a probability of danger. The uncertainty band is a fixed development-score margin, not a statistical coverage guarantee. Language checks detect some script mismatches and mixtures; they do not verify language or detect every mixture. Unknown speaker roles are not converted into minor roles.

Evidence buttons reference the input turn indices. They show which turns most change the score when omitted. This sensitivity probe is not proof of correct interpretation; a fictional framing turn can suppress a score, and removing it can change the meaning. Reports include offsets and score changes but no quoted text. Offsets use Python Unicode character indices, not JavaScript UTF-16 positions; the UI highlights entire turns.

## Response and accessibility checks

The draft checker flags a limited set of Romanian/Ukrainian phrases associated with promises of secrecy, unsupported guarantees, pressure for details, requests for identifying data and dependency pressure. A match requires contextual review; quoted criticism of such a phrase can also match. Absence of a match does not certify a reply. The human S1 contract remains separate and can record judgments about assumptions, support and appropriateness that phrase rules cannot establish.

Word counts, sentence lengths, long-word counts and a small jargon list are descriptive indicators. They do not estimate reading age or comprehension. A revised draft can be compared for number and negation changes; this catches some meaningful edits but cannot prove meaning preservation. Review both texts side by side. The tool does not automatically rewrite a child's words or a proposed response.

## From a signal to a proposed action

`contract` uses the existing six-category declared-contract schema. `audit` uses the v2 uncertainty schema. Both must have the same language as the conversation. The model never fills these facts or changes them. In the UI, the proposed-sharing form asks about recipient scope, permission, a human reviewer and whether the recipient could be the source of concern.

The combined action summary requires the existing proposed-use contract plus a resolved R1 concern audit before returning **ready for human review**. Missing or conflicting facts require review. Even the ready result does not establish legal authority or take an action. Unknown facts produce the existing exact symbolic decision sets and ranked clarification questions. Unbounded missing scope stays explicitly unresolved; no model score supplies a substitute.

## Library and CLI

```python
from roguard import analyze_conversation

report = analyze_conversation({
    "case_id": "fictional_1",
    "language": "ro",
    "turns": [{"role": "minor", "text": "Aș vrea să vorbesc cu cineva despre grijile mele."}],
})
print(report["signal"]["decision"])
```

For semantic inference, construct `ConversationScorer("semantic")` once and pass it as `scorer=`. This loads cached verified weights and keeps them in memory. `analyze_batch(payloads, scorer)` accepts up to 200 conversations, reuses embeddings within that request and discards text/embedding lookup data afterward.

```sh
roguard-conversation examples/conversation_ro.json
roguard-conversation --backend semantic examples/conversation_uk.json
roguard-conversation --jsonl examples/conversations.jsonl > current.jsonl
roguard-compare-reports previous.jsonl current.jsonl
```

Examples are in the GitHub checkout. Omit the input path to read JSON from standard input. Exported reports can be compared in Studio or through the CLI. Case IDs must be unique opaque identifiers and match across reports; callers are responsible for matching the same cases. Decision changes without reference labels are not an accuracy improvement.

The [input schema](../schema/conversation-input.schema.json) and runtime enforce 1–16 turns, at most 2,000 characters per turn, 12,000 total turn characters, and 6,000 characters per draft. The optional semantic encoder accepts at most 256 tokens, including roles and context; longer inputs abstain rather than being silently truncated. Studio retains its loopback-only origin, host and session-token checks. No remote service endpoint is exposed. Reports omit conversation and draft text, but users should still treat workflow metadata as potentially sensitive.

## Validation scope

The native-language UI wording and new message banks were authored for this project; the earlier specialist taxonomy review does not validate them. Every model result is synthetic and author-reported. The current models do not establish performance on authentic child language, all dialects, transliteration, complex third-party accounts or all forms of harm.
