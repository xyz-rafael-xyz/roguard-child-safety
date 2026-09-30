# Decision architecture

RoGuard separates **declared facts** from **model observations**. This distinction came from the first two measured pilots: the first adapter learned a card-type shortcut, and the second reached 23/40 exact matches on a new abstract test, below the prompt-only base model's 25/40. A code emitted by a language model cannot establish who may receive information or whether a permission remains current.

## Questions answered in order

1. **What facts did the caller supply?** Typed cards state a fictional policy, scope, event history, workflow, word cap, or caller-reviewed D1/S1 evidence fields. Missing facts produce an `incomplete` finding, not a negative example or a guessed grant.
2. **Which parts are mechanically decidable?** `assess_contracts()` calls the pure reference checks. It preserves separate findings when several contracts apply: a routing conflict does not disappear because a gate passes. An explicit violation produces `review`; a fully specified matching contract produces `pass`. `check_readability_contract()` tests a caller-supplied numerical cap only. D1 and S1 checks evaluate **declared reviewer judgments**; they do not inspect a message or response or calculate a Romanian comprehension level.
3. **Which observations need language interpretation?** `screen()` can return experimental category codes. `assemble_report()` keeps `D1`, `A1`, and `S1` as advisory labels. It records model `R1`, `P1`, and `G1` labels as **unverified policy labels**: a label without a contract suggests collecting the required facts. When a complete declared contract passes, a conflicting model label remains visible in `policy_disagreements` but does not override the contract's review status. The model never grants permission or runs a workflow.
4. **Who acts?** RoGuard returns data to its caller. `review_suggested` is a local return value. No function contacts a guardian, changes access, sends a report, or executes a proposed workflow. A child-safety practitioner and the organization's policy must make any real-world decision outside this package.
5. **What has been measured?** [BENCHMARK.md](../BENCHMARK.md) reports exact-match, parsing, and per-category precision and recall on abstract, generated Romanian cards. [The provenance method](METHODOLOGY.md) binds each batch to its approved taxonomy and either a historical human approval or a reproducible generator attestation. These results do not estimate performance on real child language.

The CLI report schema now makes the evidence boundary visible in each result: `checked_categories` lists the assessed scope, each finding names its `evidence_basis`, and both `caller_facts_verified` and `language_verified` are false. A passing local word-cap or route check should be read with those fields, because the checker cannot verify the real age, policy, relationship, or language tag supplied by a caller.

## Example

```python
from roguard import (ReadabilityCard, RoutingCard, assess_contracts,
                     assemble_report)

routing = RoutingCard(
    principal="minor", item="element-fictiv", purpose="A",
    proposed_recipient="tutore", recipient_roles=frozenset({"minor", "tutore"}),
    allowed_by_scope={("minor", "element-fictiv", "A"): frozenset({"minor"})},
)
contracts = assess_contracts(
    routing=routing, readability=ReadabilityCard(declared_age=9, measured_words=76, max_words=75)
)
report = assemble_report(contracts=contracts)
assert report.review_suggested
assert [finding.category for finding in report.contracts.findings] == ["A1", "R1"]
```

The system still lacks a validated Romanian age-to-reading-level scale, a reliable classifier for authentic disclosures, and the expert review needed for deployment. Improving synthetic-card scores alone would not resolve those gaps.

For a caller that has both typed declarations and an experimental model, `assess_case()` runs the declared checks first and then the optional text scorer. Invalid cards stop before the backend receives text. The resulting `GuardReport` retains advisory D1/A1/S1 labels, exact declared-contract findings, and any model-policy disagreement. The function accepts Ukrainian contracts without a model; the Romanian research adapter remains the only active model language.

The same declared-contract path is available locally through `python -m roguard examples/routing_ro.json` and `python -m roguard examples/routing_uk.json`. This is a bilingual interface to structured rules, not evidence that a Ukrainian language model has been validated.
