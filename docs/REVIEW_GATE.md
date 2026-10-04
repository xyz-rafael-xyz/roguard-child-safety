# Taxonomy review records

The Romanian taxonomy has a project approval in `taxonomy/review_ro.json`. Independent language and child-safety reviews are still pending for Romanian and Ukrainian. The pending records are `taxonomy/review_ro_independent.json` and `taxonomy/review_uk.json`. These records bind a review to the exact taxonomy bytes through `taxonomy_sha256`.

The first owner-supplied [specialist memo](REVIEW_FEEDBACK_V2.md) gave README-focused feedback and says its reviewers did not read the taxonomy. A later [taxonomy review memo](TAXONOMY_REVIEW_V1.md) reviewed the earlier Romanian and Ukrainian texts, requested changes, and calls for a new review of the corrected exact bytes. Neither memo fills an approval record for the v1 candidates.

The [Romanian review packet](RO_REVIEW_PACKET.md) and [Ukrainian review packet](UK_REVIEW_PACKET.md) ask reviewers to inspect all six categories and the language-specific role terms. Reviewers should return corrections in their own words, without child disclosure text or case paraphrases. If any category needs revision, keep `status` as `pending`, revise the taxonomy, recompute its digest, and obtain reviews of that version.

The [prospective human validation protocol](HUMAN_VALIDATION_PROTOCOL.md) separates this taxonomy signoff from a later blinded annotation study of abstract cards. The local batch agreement command can measure reviewer disagreement but cannot create an approval record or establish independence.

To record a completed review, replace `reviews: []` with exactly two entries, one with `kind: "language"` and one with `kind: "child_safety"`. Each entry needs a distinct `reviewer`, an ISO `reviewed_at` date, `independent_of_author: true`, a nonempty `summary`, and `category_decisions` containing `"accept"` for each of `D1`, `R1`, `A1`, `P1`, `G1`, and `S1`. Set the top-level `status` to `"approved"` only after both reviews are complete. The names, qualifications, and independence of reviewers must be checked by a human; JSON validation cannot establish them.

`verify_taxonomy(root, "uk")` opens the **training-data taxonomy gate** only when the Ukrainian record passes these checks. The Romanian training gate retains its historical project approval; `verify_independent_reviews(root, "ro")` is a separate release review check. Neither review enables Ukrainian model inference or proves authentic-language accuracy. Those require separate data and evaluation evidence.

Run the local gate explicitly with:

```bash
python -c 'from pathlib import Path; from roguard.review import verify_independent_reviews; print(verify_independent_reviews(Path.cwd(), "ro"))'
python -c 'from pathlib import Path; from roguard.review import verify_taxonomy; print(verify_taxonomy(Path.cwd(), "uk"))'
```

Both commands currently raise `ReviewError` because the reviews are pending.

For the corrected v1 candidates, run `roguard-v1-review --root .`. It checks [Romanian](../taxonomy/review_ro_v1_candidate.json) and [Ukrainian](../taxonomy/review_uk_v1_candidate.json) candidate records against the candidate bytes. Both remain pending. Historical batch and study attestations retain the old taxonomy files and digests.
