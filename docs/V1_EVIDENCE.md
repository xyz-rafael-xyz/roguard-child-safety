# V1 evidence ledger

`roguard-v1-review --root .` is the gate for the corrected [Romanian](../taxonomy/taxonomy_ro_v1_candidate.md) and [Ukrainian](../taxonomy/taxonomy_uk_v1_candidate.md) candidates. Both exact-byte review records are pending. The new review memo requested the revisions; it did not accept the resulting bytes. `roguard-v1-evidence --root .` still audits the historical study pipeline bound to the earlier taxonomy paths. It reports both earlier independent review records pending and all four D1/S1 cells blocked. The study pipeline needs a new preregistration and exact-version packet binding before it can support v1 model claims.

The evidence command also reports `candidate_v1_taxonomy`, `candidate_v1_study_pipeline_bound`, and `candidate_v1_release_gates_passed` separately. An old abstract-study pass can never turn the candidate release flag on: the current evaluator does not bind its packets to the corrected candidate. The owner-supplied [four-role submission](REVIEW_SUBMISSION_V1.md) matches the new digests, but the candidate review records remain pending until reviewer provenance and the category explanations are resolved.

Separate private synthetic studies V30 and V31 failed their once-scored transfer targets after passing development gates, at 129/192 and 143/192 complete pairs. Those results cannot fill the independent-study cells and rule out a high-accuracy claim for the current text readers.

The following **historical pipeline** manifest format remains available for older taxonomies. It must not be used as evidence for the corrected v1 candidate. A new study must register and bind the candidate digests before any candidate predictions are scored:

```json
{
  "schema_version": 1,
  "studies": {
    "ro:D1": {
      "packet_dir": "review_runs/ro-d1",
      "reviewer_a": "review_runs/ro-d1/a-answers.jsonl",
      "reviewer_b": "review_runs/ro-d1/b-answers.jsonl",
      "adjudicated": "review_runs/ro-d1/adjudicated.jsonl",
      "predictions": "eval/prospective/ro-d1-frozen-predictions.json"
    }
  }
}
```

For a historical replay only, run `roguard-v1-evidence --root . --study-manifest review_runs/v1-studies.json`. Each supplied cell is **reevaluated** with `roguard-human-eval-registered`; the manifest cannot assert that it passed. The command checks the language/category, both private packet bindings, packet and answer alignment, adjudication and prediction alignment, and frozen model identifier before accepting the evaluator's preregistered numeric result. A missing cell remains pending; malformed or unverifiable evidence is marked invalid. Paths cannot leave the repository. The output gives only summary counts and rates, without cards, answers, reviewer names, or custom identifiers.

`machine_abstract_gates_passed: true` would mean only that the supplied abstract-study inputs passed structural and numeric checks under the historical taxonomy paths used by this evaluator. It does not apply to the corrected v1 candidate. It never verifies who reviewed or authored them, whether they worked independently, whether an adjudication was valid, whether the frozen model actually produced the supplied predictions, or how the model will behave on authentic child language. The study owner must document those facts separately. A public weight release, package-index registration, or external submission remains a separate decision under the project master plan; this ledger does not take any such action or authorize operational child-safety use.
