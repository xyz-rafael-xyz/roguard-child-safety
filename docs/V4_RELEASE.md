# RoGuard 4.0: conversation review and semantic transfer

RoGuard 4 adds a bilingual local conversation workspace, an optional semantic classifier, explicit uncertainty, turn sensitivity references, draft-response phrase checks, descriptive complexity measures, revision checks, independent permission/action summaries, batch inference and report comparison. The [workspace guide](CONVERSATION_WORKSPACE.md) gives complete setup and API examples. The earlier contract APIs and v3 message scorer remain available.

The new semantic classifier fits a logistic head on frozen, pinned mmBERT-base embeddings. This is representation transfer with a trained head, not full encoder fine-tuning. The hybrid combines semantic and character scores. The default fast model requires no extra dependencies; semantic and hybrid require an explicit optional model download. All inference is local. Nothing in the model creates permission to disclose information or contacts another person.

## Two iterations and what changed

The first registered iteration used 8,784 training rows from template combinations of 24 settings and 12 need phrases per language, with three training styles. Its first test used distinct settings, need phrases and frames. Standalone semantic inference reached 96.2% Romanian and 95.1% Ukrainian balanced accuracy on that restricted test. A separate Qwen-generated diagnostic exposed source sensitivity, but also unnatural wording and unvalidated negative labels. Those rows were not used for training or treated as authentic-language evidence. The initial nested generator failed its schema on all 16 requests; a simpler registered interface produced 32 valid objects. Both records remain in Git.

The second iteration added complete native-language phrase pairs and canonical, previously consumed v3/v3b stems. It used **13,776 training exposures, 9,408 unique rendered conversations**, including many related wording variants. Those counts are not independent authors or scenarios. The new bank has 16 training, 12 development and 20 test phrase pairs per language. Each pair supplies five discourse forms. The fitted heads and language-specific thresholds were frozen before opening the new test. Hyperparameters and ensemble weights were chosen on development data only.

## Fresh second-iteration test

The test contains **200 conversations: 100 per language, formed from 20 phrase families per language**. Each language has 60 positive and 40 negative rows. These rows share stems and discourse templates, so row-level intervals are descriptive and optimistic about independence.

These are forced-threshold results, before treating the uncertainty band as abstention:

| Frozen model | RO recall | RO false alerts | RO balanced accuracy | UK recall | UK false alerts | UK balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Released v3, final minor message only | 28/60 | 17/40 | 52.1% | 22/60 | 12/40 | 53.3% |
| Earlier V4 semantic candidate | 60/60 | 15/40 | 81.2% | 60/60 | 12/40 | 85.0% |
| New fast character model | 45/60 | 8/40 | 77.5% | 38/60 | 2/40 | 79.2% |
| New standalone semantic model | **59/60** | **2/40** | **96.7%** | **57/60** | **1/40** | **96.2%** |
| New hybrid model | 52/60 | 1/40 | 92.1% | 46/60 | 0/40 | 88.3% |

The second semantic model reduced false alerts substantially compared with the earlier semantic candidate, with a small recall reduction. Its complete-family counts were 17/20 Romanian and 16/20 Ukrainian. The hybrid's lower false-alert count came with more missed positives. The fast model remains substantially weaker; it should not inherit semantic-model accuracy claims.

The standalone semantic backend performed best on this opened test and is the documented choice when users want the stronger evaluated model. That recommendation is informed by these test results; it is not a new independently validated model selection. All candidate results are disclosed. Comparing v3 with a conversation model also changes the information available: v3 receives the last minor message, while the new models receive context. This measures workflow improvement, not a controlled architecture-only gain.

For standalone semantic inference, descriptive 95% Wilson intervals are 91.1–99.7% for Romanian recall and 86.3–98.3% for Ukrainian recall. False-alert intervals are 1.4–16.5% and 0.4–12.9%. The observed 2/40 and 1/40 rates correspond to 50 and 25 alerts per 1,000 negatives, but the small correlated sample cannot establish those rates in deployment.

## Uncertainty and review workload

The API abstains when the synthetic score is within 0.08 of the frozen development threshold. This is not conformal calibration or a safety probability. It also abstains for unsupported roles, certain language/script mixtures, insufficient text and semantic token overflow. Language heuristics are incomplete and may reject valid inputs.

| Standalone semantic model | Decided rows | Unresolved rows | Missed positives among decided rows | False alerts among decided rows | Review signal or unresolved |
|---|---:|---:|---:|---:|---:|
| Romanian | 97/100 | 3/100 | 1 | 1 | 62/100 |
| Ukrainian | 95/100 | 5/100 | 1 | 1 | 60/100 |

Abstentions are not counted as correct classifications. The workload counts apply to this deliberately enriched synthetic test, not ordinary-message prevalence. A negative decision cannot close a concern.

## Speed and quantization

On one ARM Mac, with four CPU threads, the final semantic head and float encoder took about 3.6 seconds to load. Warm single-input scoring had a 56.8 ms median and 61.4 ms 95th percentile on a fixed development diagnostic. A 48-row batch took 1.03 seconds. The fast model had a 0.17 ms median and 0.27 ms 95th percentile. These timings exclude turn-ablation evidence probes, network downloads and browser rendering. Process peak memory before the quantization trial was about 1.6 GB. Other hardware and longer conversations will differ.

An initial quantization attempt found no selected ARM quantization engine. Selecting QNNPACK resolved execution. The final-model int8 diagnostic changed no decisions among 48 examples, but shifted some scores by as much as 0.20 and took 2.51 seconds for the same batch—slower than float. The earlier head had two decision changes. Quantization remains an experiment and is not exposed as a supported runtime mode. Weight caching and request-scoped batching are implemented without retaining input text between requests.

## Surface-form robustness repair

A development-only diagnostic uses 20 existing development conversations and eight variants each (160 per backend). Before repair, all-capital input changed 6/10 Romanian and 5/10 Ukrainian semantic decisions; extra spaces changed 2/10 Romanian decisions; decomposed Unicode changed 1/10 Ukrainian decisions. This was an input-sensitivity defect.

The runtime now normalizes Unicode to NFC, collapses repeated horizontal whitespace, and converts wholly capitalized turn lines to sentence case. Speaker roles and turn boundaries remain. After repair, the same casing, spacing and Unicode variants produced identical semantic scores to their ordinary versions. No semantic decisions changed in the tested punctuation, Romanian missing-diacritic or Ukrainian mixed-Latin-i variants, although scores still shifted. The fast backend retained one decision change for Ukrainian mixed-Latin-i. Both backends abstained on all tested English mixtures and unknown speakers; the semantic backend abstained on all oversized token contexts. These are small development diagnostics, not fresh accuracy results or guarantees for all spelling variations.

The repair leaves **all 200 frozen holdout encoder inputs byte-for-byte unchanged**, verified without choosing new thresholds or fitting weights. The published holdout results therefore still apply to exactly the same encoder inputs. Registered fit reproduction explicitly retains the original raw-input encoder mode; runtime preprocessing is identified as `surface-v1` in reports. Before/after diagnostics and the input-equivalence check are retained in the second selection directory. The repair cannot reconstruct proper-name casing or establish that mixed-language inputs are understood.

## What is and is not established

Model loading explicitly selects the hash-verified PyTorch weight file and rejects unregistered adapter/tokenizer overrides; auxiliary special-token metadata is also checked. Software tests cover strict input limits, role handling, uncertainty, text-free outputs, independent authority checks, batch equivalence, revision signals and loopback request protections. Browser checks exercise both languages, desktop/mobile layout, policy integration, clearing inputs and invalidating stale exports. These tests verify behavior, not child-safety accuracy.

The new UI wording, message banks and response patterns have not received independent native-language validation. Prior specialist review applies to the exact reviewed taxonomy, not these models or texts. Every example is fictional. No authentic-child accuracy, demographic fairness, comprehension, universal harm coverage or readiness for autonomous deployment is established. Response phrase checks can miss paraphrases and can flag quoted criticism; complexity counts and revision checks do not prove comprehension or preserved meaning.

The contribution is an inspectable connection between bilingual conversational suggestions and independently declared action constraints, accompanied by frozen comparisons that expose context, source shift, abstention and review workload. These components build on established methods; no generic first-of-its-kind claim is made.

## Records and reproduction

- [First protocol](V4_STUDY_PROTOCOL.md), [first selection and results](../research/v4-conversation-selection/).
- [Second protocol](V4B_STUDY_PROTOCOL.md), [second selection and results](../research/v4b-conversation-selection/).
- [Second-source protocol and interface repair](V4_SECOND_SOURCE_PROTOCOL.md).
- The training generator is `training.conversations_v4b`; the fitter is `training.fit_conversations_v4 --corpus v4b`. Install `.[message-train,semantic]` for experiments.
- `python -m eval.run_conversations_v4 --corpus v4b --base MODEL_PATH --output NEW_FILE --device cpu` verifies packaged artifact digests and reproduces the second test. Use a clean full Git checkout: the earlier comparators are read from frozen commit `c2a6f96`. The evaluator refuses to overwrite results.
- The base model ID, exact revision, training/development digests, fitted artifact hashes and source commits are recorded. Selection was made on MPS; reproductions on different libraries/hardware may differ numerically. Do not silently replace the published results with a rerun.
