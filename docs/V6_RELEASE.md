# RoGuard 6: adaptation, durable execution and reliability

V6 implements all six approved directions. [Usage and integration](RELIABILITY.md) · [registered plan](V6_PLAN.md) · [complete records](../research/v6-reliability/).

## Capabilities

- A bundled, hash-verified LoRA adaptation of pinned mmBERT, explicitly selected with `--backend adapted`, plus model-bound class-conditional prediction sets. Existing backends remain available.
- Metadata-only monitoring with retention, per-language distributions, latency and descriptive shift alerts; optional standard OpenTelemetry spans.
- An authenticated ASGI service with tenant/role credentials, encrypted persistent policy sessions, a bounded job queue, explicit approvals, cancellation, restart behavior and one-attempt execution.
- Explicit chat-completions adapters and comparisons against two actual cached generative models, retaining malformed outputs and task failures.
- Bounded overlapping long-conversation analysis with original character references and conservative aggregate decisions; calibrated prediction APIs and an optional feature-distance diagnostic.
- Permission-aware repair using declared alternative fields, preserving requirements/scope and rechecking permissions before execution.

## Encoder training: three iterations, all retained

Each iteration trained two rank-4 LoRA candidates for three epochs with different learning rates, selected on development only. All use synthetic, project-authored families; none has independent native-language validation. First/second tests were consumed only for the explicitly registered third fit. Training counts are 400, 400 and 580 exposures, not independent authors. Shared discourse frames make these narrow family tests correlated.

| Fresh test at that iteration | Model | RO recall / false alerts | UK recall / false alerts | Promotion gate |
|---|---|---|---|---|
| First: 10 families/language | v4 semantic | 19/30; 1/20 | 19/30; 0/20 | comparator |
| First | adapted candidate | 30/30; 2/20 | 29/30; 0/20 | failed: RO false alerts increased |
| Second: 8 families/language | v4 semantic | 14/24; 0/16 | 15/24; 0/16 | comparator |
| Second | negative-weight candidate | 24/24; 1/16 | 24/24; 1/16 | failed: false alerts increased |
| Third: 6 families/language | v4 semantic | 10/18; 0/12 | 12/18; 0/12 | comparator |
| Third | released adapted candidate | 18/18; 0/12 | 18/18; 0/12 | passed the registered threshold-score gate |

The selected third candidate uses learning rate 0.0001, epoch 2, negative-example loss weight 2. The export preserves its tensors and head; installed-backend replay differs from frozen scores by at most about 8.8e-8 on the recorded hardware. These small synthetic tests do **not** establish perfect accuracy, real-child performance, multilingual fairness or superiority to industry systems.

### The deployed adapted API abstains

The table above uses fixed development cutoffs. The actual adapted conversation API uses prediction sets calibrated separately from training, with 18 positives and 12 negatives per language. On the third test it returned 29/30 singleton decisions in Romanian and 26/30 in Ukrainian, with no wrong singleton decisions; the remaining 1 and 4 cases were unresolved empty sets. They are not counted as correct. Empirical set coverage was 29/30 and 26/30, respectively; a nominal alpha of 0.1 is not a guarantee on this small shifted sample. Calibration was reused across iterations and is not independent evidence of deployment coverage. Long-window aggregates are explicitly uncalibrated.

## Actual-model integration comparison

Two cached quantized models, Qwen3-4B-Instruct-2507 and Phi-4-mini-instruct, received identical fictional scenarios with the host's minimum-disclosure proposal visible. Each ran two stochastic repetitions over 24 scenarios: 80 requests, 40 permitted-task opportunities, and 12 addition tasks per model. This measures following the supplied contract, not learning Romanian/Ukrainian policy from scratch. Exact model file hashes and language-specific results are recorded.

| Configuration | Permitted tasks completed / 40 | Addition tasks / 12 | Unresolved requests / 80 |
|---|---:|---:|---:|
| Qwen, initial output instructions | 28 | 0 | 0 |
| Phi, initial output instructions | 28 | 0 | 4 |
| Qwen, clarified required-answer instructions | 40 | 12 | 0 |
| Phi, clarified required-answer instructions | 28 | 1 | 5 |

All four runs proposed zero unauthorized disclosures in these cases. The before/after comparison reuses scenarios and is a development integration repair, not a new held-out model benchmark. Phi failures remain in the denominator and records. RoGuard cannot guarantee another model's correctness by wrapping it. No paid external model or real child data was used.

## Repair and durable execution

A controlled repair test covers 400 related variants: all original over-disclosing proposals were blocked; 200 had a permitted alternative and were repaired, while 200 correctly remained unrepairable. There were zero unauthorized repaired selections. Revoking the permitted source stopped all 200 previously viable proposals. Variants share simple templates; semantic sufficiency remains a host declaration.

Security tests cover tenant isolation, encrypted values, context-bound ciphertext, credential roles, binding-version changes, expiry, revocation, cancellation, concurrent execution, idempotent enqueue, restart and uncertain side effects. The service is a bounded single-database architecture, not a distributed transaction across external systems. A remote receiver still needs final authorization. Trusted callbacks cannot be forcibly cancelled or sandboxed by this library.

Monitoring stores fixed metadata, not message text. Statistical distribution change is not evidence of accuracy change without trusted labels. The feature-distance diagnostic can miss unfamiliar semantics; overlapping windows can miss relationships across distant turns. Reviewers' earlier approvals apply to their exact reviewed material, not the new texts or models.

## Reproduce

The plan and study code were committed before each fit. `training.adapt_v6`, `eval.evaluate_adaptation_v6`, `training.export_adapter_v6`, `eval.compare_local_targets_v6` and `eval.evaluate_repair_v6` contain the workflows. Output files refuse overwrite. Model training needs the `train` extra and the pinned base; the actual-model comparison needs `mlx` and compatible cached weights. See the commit/file digests in each record. The public adapter is an explicit JSON tensor export of this fictional-data study; private historical model artifacts remain excluded from Git.

### Earlier-test regression check

On the already published 200-row v4b test (consumed regression evidence), Romanian counts remained 59/60 signals and 2/40 false alerts. Ukrainian improved from 57/60 and 1/40 to 58/60 and 0/40. This check was run after selection; it is not a new independent test or a retuned cutoff.

### Verification notes

The complete pre-release suite passed 373 tests with 14 optional-environment skips. Executing the documented real-clock gateway example then exposed an immediate-execution timestamp rounding mismatch; enqueue timestamps now use the same persisted integer clock as execution, with a dedicated fractional-clock regression test. The corrected example executes once and rejects replay. Final release checks rerun that test and the complete suite. Browser verification covers Romanian/Ukrainian repair, long input, stale exports, invalid JSON, clearing and a 390-pixel mobile viewport. A fresh package installation and GitHub CI are required before publication.
