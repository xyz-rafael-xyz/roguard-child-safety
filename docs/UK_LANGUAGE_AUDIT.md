# Ukrainian terminology and review brief

**Status:** source-assisted internal audit, not an independent fluent or child-safety review. The [Ukrainian taxonomy](../taxonomy/taxonomy_uk.md) remains a draft and is barred from training by `verify_taxonomy`. No Ukrainian model result is advertised as validated.

The [Ukrainian UNICEF pocket guide for supporting a child or adolescent who seeks help](https://www.unicef.org/ukraine/media/22116/file/GBV_PocketGuide_UA_17.05.21.pdf) uses terms such as *дитина/підліток*, *звернутися по допомогу*, *зберігайте конфіденційність*, *зрозумілий їй спосіб*, and *дорослий, якому вона/він довіряє*. It notes that children may seek help indirectly, asks adults to avoid interrogation and judgment, recommends asking permission before sharing information even with a trusted person, and warns against promises one cannot keep. Those are useful language and boundary checks for D1, R1, A1, P1, and S1. The guide addresses human service providers in a specific context; it does **not** validate RoGuard's category thresholds, authorize automated routing, or supply training examples.

The taxonomy deliberately uses *піклувальник* as a **fictional role in a declared policy card**. A fluent reviewer should check whether this term misleadingly suggests actual legal standing in ordinary Ukrainian. If so, the final draft should use a clearer role label while retaining the rule that a model label grants no access. The current draft makes no legal determination.

The [frozen Ukrainian probe](UK_PROBE_PROTOCOL.md) has a separate, concrete A1 language defect: five of its eight A1 cards use `слів` after 54, 62, 63, or 92 where the quantity phrase requires `слова`. Seven occurrences are affected. The [registered diagnostic](UK_NUMERAL_DIAGNOSTIC.md) preserved the original bytes and corrected these forms only in memory. Qwen still returned `A1` on all eight cards, with 4/8 correct cards and 0/4 complete pairs. This flaw further limits interpretation of the pilot; its correction did not fix the observed model failure and is no substitute for fluent review or a new, independently authored card set.

## Independent review requested

Please review the Ukrainian text as Ukrainian, including its category names, field labels, code meanings, and example descriptions. Keep examples abstract; do not add a child's realistic utterance, a case paraphrase, or a ready-to-send response. For each item, provide the exact phrase to change, a proposed Ukrainian replacement, and a short reason:

1. Does D1 distinguish an indirect support-seeking pattern with a separate safety marker from an ordinary pause or topic change?
2. Does R1 avoid implying that a parent, guardian, or fictional *піклувальник* automatically receives private information?
3. Does A1 avoid claiming that a word-count rule proves Ukrainian comprehension or that an English readability scale transfers?
   Check numeral–noun agreement in any age, word-count, or length examples; the frozen probe contains documented errors.
4. Does P1 distinguish revocation, pause, renewal, and needless withholding of permitted information?
5. Does G1 clearly require a separate human decision before external action and keep malformed model output visible?
6. Does S1 describe applicable support fields without prescribing one universal response or an impossible promise?

After a fluent language and child-safety reviewer responds, revise the taxonomy, record their name/date and the exact reviewed file hash in `taxonomy/review_uk.json`, and then run a separately authored Ukrainian abstract-card study. Until then, the draft gate stays closed.
