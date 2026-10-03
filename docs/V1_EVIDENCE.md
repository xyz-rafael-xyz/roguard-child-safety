# V1 evidence ledger

`roguard-v1-evidence --root .` reports the four independent abstract-study cells needed for a bilingual D1/S1 research release: `ro:D1`, `ro:S1`, `uk:D1`, and `uk:S1`. It first checks the exact-byte Romanian and Ukrainian independent taxonomy review records. It currently reports both as pending, and all four study cells as blocked. It cannot manufacture reviewers, authors, adjudications, or model accuracy.

Separate private synthetic studies V30 and V31 failed their once-scored transfer targets after passing development gates, at 129/192 and 143/192 complete pairs. Those results cannot fill the independent-study cells and rule out a high-accuracy claim for the current text readers.

Once the independently authored, approved, blinded, and adjudicated studies exist, the owner may keep a **private** manifest under `review_runs/` with paths relative to the repository:

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

Then run `roguard-v1-evidence --root . --study-manifest review_runs/v1-studies.json`. Each supplied cell is **reevaluated** with `roguard-human-eval-registered`; the manifest cannot assert that it passed. The command checks the language/category, both private packet bindings, packet and answer alignment, adjudication and prediction alignment, and frozen model identifier before accepting the evaluator's preregistered numeric result. A missing cell remains pending; malformed or unverifiable evidence is marked invalid. Paths cannot leave the repository. The output gives only summary counts and rates, without cards, answers, reviewer names, or custom identifiers.

`machine_abstract_gates_passed: true` means only that the four supplied abstract-study inputs passed the structural and numeric software checks under the current taxonomies. It never verifies who reviewed or authored them, whether they worked independently, whether an adjudication was valid, whether the frozen model actually produced the supplied predictions, or how the model will behave on authentic child language. The study owner must document those facts separately. A public weight release, package-index registration, or external submission remains a separate decision under the project master plan; this ledger does not take any such action or authorize operational child-safety use.
