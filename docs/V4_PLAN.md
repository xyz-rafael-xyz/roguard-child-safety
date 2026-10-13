# Conversation workspace implementation

This work implements the owner's approved October 2026 roadmap. The product remains RoGuard and the distribution remains `roguard-child-safety`. All new study text is invented, low-detail support-seeking material; no real cases, identifying details, or operational grooming scenarios are used. No function sends a notification or makes an external decision.

## Deliverables

1. Conversation API, strict schema, explicit uncertainty and source-role handling, evidence references without storing message text, batch CLI.
2. Draft-response review and descriptive language-complexity measures, with limitations and independent human contract judgments.
3. Evidence-to-action report combining model suggestions with separately declared permission/routing/gate facts and existing exact uncertainty questions.
4. Local Studio conversation editor, response editor, readable Romanian/Ukrainian results, report export, and regression comparison.
5. Broader synthetic corpus with scenario-family splits, training-only perturbations, hard negatives, context and language stress tests, published provenance and counts.
6. Controlled comparison of the released character model, a new character baseline, and pinned multilingual semantic features with fitted heads; development selection and frozen fresh testing, including uncertainty and operational metrics.
7. Weight caching, batch inference, measured CPU/GPU latency and optional quantization experiment; documented installation and release after checks.

## Research controls

Model selection uses training and development only. The first candidate experiment compares several regularization settings; a second development experiment adds lexical features to semantic features. Both are frozen before test scoring. Dataset code and protocol are committed before fitting, selection artifacts before test evaluation. Test results are retained; fixes based on an opened test require a new holdout. All generators share this project's author and therefore do not constitute independent human validation. Raw row counts, base scenarios, templates and dependency between rows are reported separately. No synthetic study establishes authentic-child accuracy or comprehension.

The model is advisory. Unknown, unsupported and ambiguous inputs produce an explicit uncertainty result. Tests will verify that changing model scores cannot grant permission, erase a declared concern or remove a human gate. A response lint pass does not certify a helpful or safe reply.

## Release criteria

All existing and new relevant tests pass; clean packaged CLI and Studio workflows work; evaluated artifacts match reported hashes; CI passes on supported Python versions; a stronger semantic candidate is offered only with its measured results and separate optional dependencies. No claim of universal superiority or first-of-its-kind status is made.

## Implementation record

The deliverables are implemented in v4.0.0. The [release report](V4_RELEASE.md) records two frozen training iterations, all comparator results, source-shift diagnostics and the quantization outcome. The [workspace guide](CONVERSATION_WORKSPACE.md) documents setup, APIs, input limits and current limitations.
