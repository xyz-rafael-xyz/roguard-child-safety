# Context evidence study, candidate v7

This study adds advisory interpretation of the latest turn in a bounded conversation. It does not infer legal authority or grant permission. All generated material is fictional and mild, with no real child records.

## Frozen design before fitting

Two independently fitted heads sharing an encoder on pinned, frozen mmBERT features predict (1) one of SELF_CURRENT, SELF_PAST, OTHER_PERSON, QUOTED, CORRECTION, UNCERTAIN, ORDINARY and (2) whether a support concern is mentioned. These are limited operational labels, not diagnoses. The previous turns are context, and the focus turn is the candidate reference; neither attention nor a reference proves causality. Corrections never erase a previous review signal automatically.

Compare a character TF-IDF classifier and the semantic heads on the same frame task. Compare the support head against v6's adapted binary output only on the binary task; v6 cannot output the new frame classes. Select regularization on development macro F1, then fit class-conditional prediction sets using a separate calibration partition. Report abstention and decided errors, including per-language results and pair completion. Prediction sets require exchangeability, which generated template families do not establish for deployment.

Training, development, calibration and final tests use disjoint phrase families and disjoint context wrappers. Surface variants remain grouped with their source family. The deterministic authoring generator is one source, not multiple independent authors. Separately generated model-authored probes must be recorded with generator identity and kept separate from the primary test; their labels are not expert judgments. No public or private reviews of older taxonomies validate this new task.

Development iteration 2 adds training-only paraphrases and concatenates whole-prefix and focus-turn embeddings; iteration 1 artifacts and scores are retained. This change is chosen before opening the final test.

Try C in 0.1, 1, 10 for both heads; choose minimum per-language macro F1, then average F1. A candidate is useful as an experimental feature if it improves on the same-task character baseline or provides inspectable context functionality, but it must not be described as a validated accuracy improvement unless the fresh comparison supports that claim. Final tests are evaluated once per frozen candidate; any later use makes them development data and requires a new final test.

The application layer must reject malformed or oversized input, report unsupported context explicitly, preserve source restrictions, treat classification as advisory, expose unresolved questions, and never execute a sharing proposal. Host-reviewed sources and coverage remain authoritative. Runtime tests cover input limits, correction retention, model identity, source derivation, revocation and absent grants. A runnable tutoring workflow and local review interface must show model outputs separately from host policy and actual execution.

## Calibration repair after first final diagnostic

The first final evaluation exposed badly shifted class-conditional sets (a small ordinary class forces a quantile of 1). That test is now a diagnostic, not an untouched final test for subsequent candidates. Iteration 3 uses per-language pooled nonconformity across classes and requires a singleton to agree with argmax. This provides no class-conditional coverage claim. Training adds ordinary practical questions and deterministic Ukrainian transliteration. The next frozen test uses `context_fresh_v7.py`, six newly authored phrase families per language. The original test stays visible; it is not used in fitting. Model-authored probes stay outside training.

Iteration 3 reduced development frame performance after adding transliterations. Iteration 4 retains ordinary-question augmentation but removes those transliterations from training. Original-script performance selects the candidate; transliteration will be an explicitly unsupported input, not an untested quality claim. This decision uses development only, before the fresh final test.

Iteration 5 jointly trains a rank-4 attention LoRA and the two heads, using a seeded 2,500-row training subset, two epochs, learning rate 0.0001, and frame cross-entropy plus 0.25 times binary mention loss. The first and second final tests remain diagnostics after inspection. Select epochs on development only. A joint candidate must improve minimum per-language development frame macro F1 over iteration 4 (0.7711) before it can replace the frozen encoder candidate. Independent model-authored probes remain diagnostics, not independently labelled ground truth.

The joint candidate improved development minimum macro F1 to 0.8966 and is selected for the third final test. The third test has six new content families per language and disjoint contextual wording in `context_final_v7.py`. Compare character, frozen semantic and joint models on exactly the same rows. This is still one authored synthetic source with repeated wrappers. The earlier model-generated probes contain severe instruction-following and language-quality problems and are excluded from labelled accuracy/training; they are preserved as generator failures. No independent language-quality validation is claimed.

## Factorial mention correction

Inspection of the third final test and the dataset design exposed a shortcut: all non-ORDINARY frames in the primary generator mention a concern. Iteration 6 adds independently worded benign content to every frame, including self/current, past, quotation and correction, in training/development/calibration. This decouples the two tasks. Joint training uses 4,000 seeded rows and two epochs; the final holdout must cross benign/concern content with frame. Previous final tests stay diagnostic. No prior final-test wording is included in the added training material.

Iteration 6 selects epoch 2 (new factorial development frame macro F1: RO 0.8167, UK 0.8181). These values are on a broader development task and are not directly comparable to iteration 5's narrower development values. Freeze the candidate and evaluate `context_factorial_final_v7.py` once. Report frame and mention tasks separately, plus end-to-end reviewer workload. Old baselines are evaluated unchanged on the expanded task, with their narrower training exposure disclosed.

## Incremental-prefix correction

The end-to-end workload audit found unnecessary reviews on intermediate context-only messages: fitting only final turns mismatched the incremental runtime. Iteration 7 continues the iteration-6 joint adapter at learning rate 0.00005 for two epochs. Training adds all derived intermediate prefixes with an explicit fourfold weight, while development/calibration use unique prefixes. A standalone concern is SELF_CURRENT with mention=1; a scope-only wrapper has mention=0. Prior final sets and workload results are now diagnostics, not untouched final evidence for this candidate. A final independent-of-fitting authored conversation probe will report actual incremental behavior and retain failures.

The final incremental probe is `context_session_probe_v7.py`: 12 newly authored two-turn cases per language (six benign, three personal concerns, one quoted concern, one correction and one uncertain concern). Report category counts and every output against v6. This small same-author probe is not independent expert evidence and must not be converted into a general accuracy percentage. Replays of prior factorial tests are explicitly regressions, not fresh final tests for iteration 7.

Iteration 7 selects epoch 1 by the registered minimum-language development frame criterion (RO 0.8509, UK 0.8451); epoch 2 has a lower minimum. The released artifact hash is frozen before the newly authored session probe. The existing factorial test is replayed as a regression, explicitly not fresh evidence for this candidate.
