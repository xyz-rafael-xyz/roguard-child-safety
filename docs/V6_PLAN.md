# V6 implementation and experiment plan

Six approved directions: language generalization experiments; metadata-only observability and OpenTelemetry; authenticated durable application gateway; actual local-model comparisons; calibrated selective prediction and long-conversation evidence; permission-aware assistance repair.

## Release gates

- Existing APIs remain compatible; core installation remains dependency-free on Python 3.9+.
- Tenant isolation, authorization, expiry, revocation, cancellation, restart, approval replay, and uncertain callback outcomes have executable tests. No automatically retried side effect. HTTP clients cannot choose executable code or arbitrary destinations.
- Telemetry accepts a fixed metadata schema, excludes text and credentials, expires records, and reports language-specific distributions and latency. Drift is not accuracy without labels.
- Long-context analysis retains source turn/chunk references and reports incomplete coverage. Calibration is bound to model identity, language and task; exchangeability is an assumption, not a deployment guarantee.
- Repair preserves host-declared requirements and scope, never invents permission, and is rechecked at execution.
- Compare at least two cached actual local generative models on identical bilingual scenarios. Retain invalid replies and failures in denominators; distinguish proposals from enforced outcomes. Do not compare incompatible tasks as one accuracy number.
- Run two registered encoder-adaptation candidates, select on development only, freeze before fresh testing, retain all outcomes. Promote only if per-language recall does not decrease, false-alert rate does not increase, and mean balanced accuracy improves by at least 0.02 against v4 semantic on the new test. Otherwise retain the released model and distribute the experimental workflow honestly.
- Training/calibration/test splits separated by authored phrase families. Synthetic rows are correlated and author-generated; no authentic-child accuracy or independent language review claim.
- Verify packaged installation, documented workflows, CI and public release before completion.

## Model study fixed before execution

Seed 20261026. Frozen v4 semantic baseline; two LoRA encoder candidates (rank 4, alpha 8, dropout 0.05, query/value projections, learning rates 0.0001 and 0.0003, three epochs, batch 8, AdamW, BCE). Candidate head initialized from released semantic head, encoder pooling unchanged. Training uses a deterministic bounded sample of consumed v4b training rows plus new v6 training families. Development cutoff grid 0.1..0.9 step 0.02; prefer recall among cutoffs with FPR <= 0.1, then balanced accuracy, then proximity to 0.5; otherwise minimize FPR then maximize recall. Select by minimum per-language balanced accuracy, then mean, then earlier candidate/epoch. No test-based tuning. Calibration uses separate families and class-conditional split-conformal sets at alpha 0.1. Report sample sizes and no guarantees under shift.

Prior methods: https://opentelemetry.io/docs/languages/python/instrumentation/ ; https://arxiv.org/abs/2107.07511 ; https://arxiv.org/abs/2106.09685 . These components build on established work; the joint bilingual workflow is the project contribution, not a generic first-of-kind claim.
