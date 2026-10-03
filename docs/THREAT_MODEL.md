# Threat model

## Assets and failure modes

- Children and families can be harmed by missed signals, false alarms, an incorrect recipient, or a response that omits a necessary support element.
- Model outputs can be over-trusted. This prototype returns labels and human review signals only.
- Any real child text in a dataset would violate the synthetic-only rule and could expose private information. Registered generators read only fixed abstract templates. Automated validation can prove the stored rows reproduce those generators; it cannot prove ethical provenance or real-world validity for arbitrary new text.
- A batch can change after validation. Its attestation binds exact batch, preview, audit, generator, and taxonomy bytes; any change blocks ingestion. Historical human-approved batches retain their original exact-byte approvals.
- A train/test leak can inflate reported metrics. The gate checks duplicate normalized text and IDs across selected splits and batches; the generator validator checks earlier batches and same-kind surface similarity. Semantic overlap remains and must be disclosed.
- `D1` must not characterize a child as unsafe. Routing decisions require a supplied fictional policy and scope; no family role grants access by default.
- `G1` treats an automatic notification or external action as a workflow defect. A model score or a category name never authorizes disclosure.
- Readability scores can miss semantic errors and are not portable from English to Romanian or Ukrainian without validation.
- The old Ukrainian seven-category draft is archived. A new six-category Ukrainian draft exists, but its language and subject-matter boundaries are not independently reviewed. The CLI can check structured Ukrainian-labelled fields; the learned Ukrainian model path remains disabled.
- An approved older taxonomy cannot silently authorize a revised one: v0.1 and its batch are archived. Rafael approved v0.2 and batches 0004–0007; batch 0008 has a distinct automated validation record under the later user-authorized workflow.
- Training on abstract decision cards can reward keyword recognition without teaching real language understanding. Real disclosure recognition and child comprehension remain unmeasured.
- The v0.2 Decision Studio is a loopback-only editor for declared symbolic facts, not a child-message intake service. A random token, exact Host and Origin checks, a request-size limit, no-cache headers, and a restrictive content security policy reduce cross-site and accidental-exposure risks. The local machine and browser session remain inside the trust boundary; do not proxy the port or paste real child disclosures into it. The counterfactual explorer enumerates only selected declared-fact variants and cannot prove policy completeness.
- The first adapter learned a card-type shortcut. The second adapter lost to the prompt-only base model on the fresh abstract test. No generated-card score establishes deployment readiness.

No code sends a report, changes account permissions, or contacts a third party. The repository does not claim compliance with a law or safety standard.
