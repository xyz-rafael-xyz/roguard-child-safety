# v3 fictional-message research release

RoGuard Local 3.0.0 adds an **optional, offline D1 support-signal scorer** for Romanian and Ukrainian text. It is a research aid for a human reviewer, not a child-safety decision service. The CLI takes a JSON object with `language`, `declared_source_role: "minor"`, and `text`; it returns `REVIEW_SUPPORT_SIGNAL` or `NO_MODEL_SIGNAL`, a synthetic score and cutoff, and the exact model digest. It does not echo input text, infer age or identity, route to a guardian, grant access, or contact anyone. A `NO_MODEL_SIGNAL` result must not be used to dismiss a concern.

The model is a character n-gram TF–IDF logistic classifier. The shipped inference implementation is pure Python, has no runtime dependency, and runs without a network connection. The two fitted models are small compressed JSON files. Training is opt-in via `pip install '.[message-train]'`. Both the source banks and fitted weights are made from locally authored fictional text, so there is no third-party corpus or base-model weight license in this component.

## What was measured

The first [registered protocol](V3_MESSAGE_STUDY_PROTOCOL.md) fixed the generator, six candidate settings, selection rule, and test target in commit `49b8f72`. Its selected model was frozen in `9c761c6` before opening that test. The generator produced **5,120 training rows from 40 distinct paired stems** (20 per language), 384 development rows from 12 distinct pairs, and 384 test rows from another 12 distinct pairs. Prefixes and punctuation/case variants create many correlated rows; “5,120” does not mean 5,120 independently authored messages.

| Language | Development complete pairs | Once-opened test complete pairs | Test positive recall | Test false reviews |
|---|---:|---:|---:|---:|
| Romanian | 85/96 | 96/96 | 96/96 | 0/96 |
| Ukrainian | 96/96 | 91/96 | 91/96 | 0/96 |

The first test met its prespecified target, but a separately frozen [colloquial probe](../research/v3-message-selection/colloquial-probe-result.json) found a gap: 9/10 Romanian and 7/10 Ukrainian positive messages were recognized. The probe had no false reviews in ten negatives per language. All 20 probe pairs were then **consumed as training material** for a second iteration; they are not a test of the shipped model.

The second [prospective protocol](V3B_MESSAGE_STUDY_PROTOCOL.md) and new development/test banks were frozen in `647cd69`. It trains on 42 distinct paired stems per language, including all previously consumed stems, expanded to 10,752 training rows. Unlike the first study, the new development and test splits contain one unmodified message per pair. The selected second model was frozen in `154ec63` before opening its test. **These second model bytes are shipped in v3.0.0.**

| Language | Second-model development complete pairs | New test complete pairs | Test positive recall | Test false reviews |
|---|---:|---:|---:|---:|
| Romanian | 6/8 | 7/8 | 7/8 | 0/8 |
| Ukrainian | 8/8 | 7/8 | 7/8 | 0/8 |

The second model met the prespecified minimum of 7 complete pairs and 7 positive recalls, with no more than one false review, in each language. One positive was missed in each language. [The second frozen selection and once-opened result](../research/v3b-message-selection/) include hashes and per-family errors. Always-review would catch 8/8 positives but falsely review 8/8 negatives; never-review would miss all eight positives. Eight pairs per language are too few for a stable accuracy estimate. Both iterations are **author-reported synthetic studies**, with the same author designing every bank. The first test and probe are retained; their scores do not transfer to the second model.

The messages use plausible everyday wording but have not been reviewed by native-speaking children or an independent language panel. The specialist approvals of the corrected taxonomy do not cover these new message stems. The test contains no real child messages. Therefore no authentic-child accuracy, demographic fairness, prevalence calibration, or deployment safety has been established. The output score is a ranking on this synthetic task, not a probability that a child is unsafe. Human review and the existing R1/P1/G1 contract gates remain separate.

## Reproduce the study

From a checkout with the optional training dependencies installed, regenerate each split with `python -m training.generate_messages_v3b --split train --output /tmp/roguard-train.jsonl`, replacing `train` with `dev` and `test` for the other files. Run `python -m training.fit_messages_v3b --train /tmp/roguard-train.jsonl --dev /tmp/roguard-dev.jsonl --output /tmp/roguard-fit`. Compare the new `selection.json` and two weight hashes to the frozen v3b files. Reproduce the second test record with `python -m eval.run_messages_v3b --selection-dir research/v3b-message-selection --test /tmp/roguard-test.jsonl --output-dir /tmp/roguard-check`. Use a fresh output directory because the evaluator refuses to overwrite a result. The first iteration remains reproducible with its own `v3` generator, fitter, and evaluator.

The released CLI uses the exact frozen bytes. The Python API is `from roguard.message_model import assess_message`; pass the three-field JSON-equivalent dictionary. Do not pass personally identifying or confidential real child material into a research workflow unless your organization has an appropriate governed process. The CLI reads input locally and does not store it; operators control any shell history, files, or logs around it.
