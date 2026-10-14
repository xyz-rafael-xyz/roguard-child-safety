# RoGuard 7: conversation evidence review

## What ships

- A local **joint** model: rank-4 attention LoRA plus two jointly trained heads for conversational framing and concern mentions, in Romanian and Ukrainian.
- Seven candidate frames: current/past personal account, other person, quotation, correction, uncertainty and ordinary content. These are interpretations, not risk categories or verified facts.
- Incremental review with original turn references, explicit unsupported prefixes, prediction sets and retained earlier signals.
- A bilingual browser workspace and `roguard-context` CLI.
- `ContextSession` for host applications: transcript-to-source registration, change reports, permission-checked extractive summary proposals and inherited revocation.
- An opt-in authenticated `/v1/analyze-context` endpoint and metadata-only monitoring integration.

[Setup and runnable integration](CONTEXT.md). Existing v6 APIs and models remain available. The package is usable software; the new language interpretation model remains **experimental**.

## Training and selection

Seven development iterations were run. The first fitted frozen encoder heads; later iterations expanded context wording, tried and rejected a transliteration augmentation, repaired class-conditional calibration, jointly adapted the encoder, crossed benign/concern content with each frame, and finally trained on intermediate prefixes actually encountered by the runtime.

The released model continues the factorial joint candidate using 5,088 training exposures representing **4,261 unique contexts**. Repeated prefixes receive an explicit training weight. This is authored fictional data, not thousands of independently sampled child messages. Epoch 1 was selected by minimum per-language development frame macro F1; epoch 2 was retained in the record and not selected. The final model has pooled per-language prediction sets. Neither a prediction set nor a model score guarantees safety or real-world coverage.

The [protocol](../research/v7-context/PROTOCOL.md), seven development records, training rows, candidate hashes, earlier test failures and generator-quality failures are committed. Candidate changes made after inspecting a test make that test a diagnostic for later candidates. We do not relabel repeated tests as fresh.

### Why the factorial revision mattered

The initial generator coupled almost every non-ordinary frame with a concern. Models could therefore mistake “this is a quotation” for evidence that a concern was present. The later generator puts benign and concerning content in the same frames, separating the tasks. Prefix training then addressed unnecessary review of context-only opening messages.

These are engineering and evaluation improvements. LoRA, multitask learning, conformal prediction, source provenance and minimum-disclosure planning are established methods. The contribution is their implemented connection and inspectable bilingual workflow, not an invention claim for the individual techniques.

## Released model: factorial regression

The following **previously inspected synthetic regression** has 468 cases per language, derived from six content families and repeated context wrappers. It is not an independent final accuracy estimate for the released candidate. Metrics concern the latest turn's predicted frame/mention, not the correctness of an entire safeguarding response.

| Task | Romanian macro F1 | Ukrainian macro F1 |
| --- | ---: | ---: |
| Conversational frame, argmax | 0.9871 | 0.8674 |
| Concern mention, argmax | 0.9764 | 0.9206 |

Abstention changes coverage and error counts:

| Task/language | Decided / total | Wrong decided labels | Abstained |
| --- | ---: | ---: | ---: |
| Frame / Romanian | 446 / 468 | 1 | 22 |
| Frame / Ukrainian | 355 / 468 | 11 | 113 |
| Mention / Romanian | 386 / 468 | 0 | 82 |
| Mention / Ukrainian | 388 / 468 | 3 | 80 |

The earlier factorial candidate had 38 wrong decided Ukrainian frame labels out of 457 decided; the released candidate has 11 out of 355. This trades coverage for fewer errors, and some errors remain. Empty prediction sets can exclude the true label; abstention is not a coverage guarantee. [Exact regression outputs](../research/v7-context/factorial-regression-07.json).

The unchanged character and frozen-semantic comparison heads had substantially weaker mention results on the expanded factorial task, but they had narrower training exposure. That comparison does **not** isolate a LoRA-only gain. [Earlier fresh factorial comparison](../research/v7-context/factorial-final-test.json). V6 has no seven-frame output, so assigning it a seven-frame score would be invalid.

## Fresh incremental session probe

After selecting the released model, a separate newly authored probe tested 12 two-turn conversations per language. It was not used for fitting or calibration. It is still small, same-author fictional material without independent native-speaker review.

| Probe category, per language | Released joint model | V6 |
| --- | --- | --- |
| 6 benign conversations | 1 sent to review | 1 sent to review |
| 3 personal-support conversations | All 3 sent to review | All 3 sent to review |
| 1 clear fictional quotation | Not sent to review | Not sent to review |
| 1 later correction of a personal statement | Earlier signal retained for review | No review recommendation |
| 1 uncertain concern | Sent to review | Sent to review |

These counts were the same in Romanian and Ukrainian. Retaining the correction case is a deliberate workflow policy, not proof that a real disclosure is true. This small probe does not demonstrate a general false-alert reduction over v6. [Every input and output](../research/v7-context/final-session-probe.json).

## Integration and usability verification

The regression suite covers source restrictions, absent grants, revocation of derived summaries, malformed/oversized input, model mismatch, disabled endpoints, authentication, metadata-only monitoring and browser request boundaries. The local suite ran 387 tests: 373 passed and 14 optional/historical tests were skipped. Browser checks exercised Romanian and Ukrainian results, reference navigation, clearing stale exports, and a 390-pixel mobile layout without overflow or browser errors. The release wheel was built, validated and installed outside the checkout, including actual bilingual joint-model inference.

The tutoring example produces a permitted learning-only proposal, excludes an unpermitted personal source, and blocks the derived summary after source revocation. This verifies a declared-policy integration. It does not validate inferred authority or semantic sufficiency.

## What remains limited

- Ukrainian frame errors remain, especially ordinary content and some corrections. Romanian is stronger on the recorded regression.
- The latest-turn interpretation and the conservative whole-conversation review policy are different measurements. Earlier positive signals are retained; uncertainty can increase reviewer workload.
- Candidate references cover complete turns. No independently verified span extraction, causal attribution or named-person resolution is claimed.
- Latin-only Ukrainian transliteration is unsupported. Code-switching, typos, dialects and distant context have no established accuracy guarantee. Inputs exceeding the semantic bound are explicitly unresolved.
- Summary construction concatenates host-reviewed, permitted turns. It does not generate a semantically verified new summary or automatically remove restrictions from an excerpt.
- Three local generative models produced poor-quality diagnostic material; none of those sets was accepted as labelled training/accuracy evidence. [Quality gate](../research/v7-context/GENERATOR_QUALITY.md).
- There is no new independent specialist/native-language validation, real-child deployment study, clinical certification or evidence of improved real-world outcomes. Earlier project reviews do not validate this new task.

## Reproduction

Use the optional training dependencies, the pinned mmBERT base and a checkout. The exact selected training exposures are in `research/v7-context/selected-training.jsonl`. The final training script starts from the archived iteration-6 adapter by default:

```sh
PYTHONPATH=src:. python training/adapt_context_v7.py --base /path/to/pinned-model --output /tmp/context-retrain
PYTHONPATH=src:. python eval/context_session_probe_v7.py --base /path/to/pinned-model --output /tmp/context-session-replay.json
PYTHONPATH=src:. python eval/context_v7.py --factorial --skip-v6 --backend joint --base /path/to/pinned-model --output /tmp/context-regression-replay.json
```

Add `--device mps` or `--device cuda` where available; the default is CPU. Replay scripts refuse to overwrite an existing output. Training selects on development, not the final probes. Hardware/library differences can change floating-point results; the committed artifact hashes identify what was actually evaluated and shipped.
