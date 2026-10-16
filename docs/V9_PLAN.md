# V9 decision assurance: acceptance plan

V9 develops a bounded, local decision-assurance workflow. It must not promote the
failed v8 language models or imply that a symbolic guarantee measures real-world
child-safety accuracy. The existing >=99% concern recall and <=5% benign review
targets, fresh evaluation, and integrated runtime gates remain necessary for a
stable language-model release. Initial delivery may be an explicit prerelease.

## New implementation requirements

1. Preserve explicitly supplied joint interpretation hypotheses, including their
   correlations. Never silently intersect disagreeing model hypotheses.
2. Compute an optimal adaptive clarification tree under positive integer costs:
   minimize worst-case total reviewer cost until the policy action is determined.
   The result is exact only over the supplied finite hypothesis set and policy.
3. Bind each report to normalized structured input and the complete finite policy
   truth table. Recompute to verify; a hash is not a signature or authority proof.
4. Bind reviewer answers to the report being answered. Reject stale bindings,
   invalid answers, and conflicting answers; explicitly report answers outside
   the proposed hypotheses. No model output grants permission or sends messages.
5. Connect the actual evidence API to the assurance engine without dropping
   unresolved attributes or a retained-model review floor. Supply a dependency-free
   CLI and Romanian/Ukrainian reviewer labels and examples.
6. Verify the policy table independently, tree leaves against every supplied
   world, minimax costs against a separate small-instance exhaustive solver,
   correlation handling, tampering, invalid inputs, and installed CLI behavior.

## Evidence and release requirements

Publish deterministic synthetic structural evaluations, actual runtime examples,
test results, limits, and reproducible commands. Measure reviewer effort against
the existing greedy question strategy, not as classifier accuracy. Do not claim
that generated worlds contain the true interpretation or that authored language
has acquired new independent review. Preserve reserved language-model probes.

Novelty is the implemented combination, not an assertion that optimal decision
trees, structured prediction sets, or decision provenance were invented here.
Relevant primary references: [structured prediction sets](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1868a3c73d0d2a44c42458575fa8514c-Abstract-Conference.html)
and [OPA decision provenance](https://www.openpolicyagent.org/docs/management-decision-logs).
