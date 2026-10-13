# RoGuard 5: interaction laboratory

## Scope

Implement all five approved directions: branching fictional simulations with reproducible failures; provenance and live permission checks through summaries, memory and tool boundaries; minimum necessary disclosure under explicit task requirements; decision consistency tests that distinguish invariance from required change; and useful-task preservation alongside unauthorized proposals and blocked actions.

The existing classifiers remain available and their v4 evidence remains separate. New exact policy mechanisms do not need learned permission inference. Training is warranted only if an experimental learned component beats a fixed baseline on an independently reserved task. No new classifier accuracy gain is promised by this release.

## Implementation

- An in-memory ledger owns immutable artifact provenance. Derived artifacts inherit all source restrictions. Grants are purpose/recipient/operation specific; expiry and revocation are checked against a monotonic host clock. Model output cannot grant permissions.
- A host-configured tool gateway binds destinations and purposes to callbacks. Every dispatch rechecks current source permissions and its human-approval gate. No live external tools are shipped. This is an integration boundary, not a sandbox for arbitrary Python or a verifier of legal authority.
- Exact subset optimization chooses minimum declared disclosure cost among currently eligible candidate fields while covering all declared requirements. It cannot infer semantic sufficiency or recover unregistered provenance.
- A scenario interpreter supports messages, sources, derived artifacts, grants/revocations, clock changes, requests and response-dependent branches. Target adapters are explicit Python callables or recorded proposals; browser JSON cannot load code, contact URLs or dispatch live tools.
- Counterexample reduction replays fresh target instances, preserves the same first-failure identity, reports its probe budget and distinguishes bounded one-deletion minimality from global minimality.
- Consistency evaluation supports model-message invariance and required policy decision changes. Reports retain abstentions and paired failures rather than counting agreement as accuracy.
- Utility reports separate unauthorized proposals, permitted task completion, unnecessary refusal, unresolved requests, and independently checkable arithmetic answers. Declared field coverage does not establish human helpfulness.
- Romanian/Ukrainian Studio and packaged CLI/API examples expose the complete workflow.

## Evaluation protocol

Commit the generator and protocol before measuring the final implementation. Use fixed development seeds and a separate final evaluation seed. Evaluate deliberately permissive and overblocking targets alongside the guarded reference integration. These are controlled reference policies, not trained language models. Include exhaustive bounded policy transitions, generated scenario combinations, derived-source revocations, expiry, missing provenance, model-supplied permission claims, stale selection, and preservation of authorized utility.

The same engine is not independent evidence of its own policy correctness: dedicated tests compare it with explicit expected states and a separately written bounded transition oracle. Paired surface diagnostics use the frozen v4 classifier without tuning on their results. All examples are synthetic, project-authored and not independent linguistic or clinical validation. Preserve observed failures and their repairs; never re-label failures to obtain a passing result.

## Release gates

Meaningful new adversarial/regression tests pass; existing APIs remain compatible; installed wheels work on supported core Python versions; browser workflows run in both languages; public-tree audit passes; main CI is green before a release tag. Publish scope-specific counts and limitations. No claim that a simulated or exact policy result proves authentic-child accuracy or universal security.

## Implementation record

All five components are implemented. See the [integration guide](INTERACTION_LAB.md) and [release report](V5_RELEASE.md) for executable examples, measured results and boundaries. The learned v4 classifiers are unchanged; no model is used to infer execution permission.
