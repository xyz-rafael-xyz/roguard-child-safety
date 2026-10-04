# Offline kit for independent abstract-card authors

**Version boundary:** This command and the workflow below are bound to the historical `taxonomy_ro.md` and `taxonomy_uk.md` files. Those historical independent review records remain pending, so `new` currently refuses both languages. The approved v1 candidate records do not unlock this command. Do not issue a v1 author assignment from this kit until a candidate-digest-bound study pipeline and model freeze are registered.

`roguard-author-kit` prepares historical D1/S1 study workbooks without writing unreviewed cards to Git. It does not contact authors, review their work, or generate card text. That historical workflow requires a reviewed taxonomy, a frozen model and evaluation plan, two fluent outside authors, and a separate content reviewer. The [intake contract](INDEPENDENT_BATCH_INTAKE.md) and [human validation protocol](HUMAN_VALIDATION_PROTOCOL.md) describe the requirements; the [v1 evidence ledger](V1_EVIDENCE.md) tracks the corrected candidate separately.

## 1. Create separate author tasks

After the taxonomy reviews and model freeze, the study owner runs:

```sh
roguard-author-kit new --batch batch-XXXX --language ro --category D1 --freeze-record eval/prospective/ro-v16-registered-study-freeze-v5.json --output-dir review_runs/batch-XXXX-kit
```

Use `S1` for a separate Romanian study. A future Ukrainian study needs its own verified Ukrainian model freeze; the Romanian V16 record cannot be reused for `uk`. The [freeze format](STUDY_FREEZE.md) lists the required fields and limits. The command checks that the record matches the language and category, and that its named model weights, manifest, inference code, and evaluator code still have their registered hashes. The new V16 study record includes the registered evaluator and every current RoGuard Python source file; the older V16 record remains historical preregistration evidence. It writes the freeze path and digest into private `author-a.json` and `author-b.json` files plus an owner-only `owner-assignments.json`. Each author file has 24 independently shuffled factor assignments, balanced at six per D1 factor or three per S1 factor. Give each author only their own file and the reviewed taxonomy; keep the owner assignment map private. The owner must arrange any communication separately; this command sends nothing. Keep the files and real identity records outside Git.

Each author fills `native_language`, `independent_of_project`, and `authored_at` with their own truthful declarations, then writes one `negative_abstract_card` and one `positive_abstract_card` for every assigned factor. Cards describe invented attributes or response-field properties only. They must contain no real or realistic child utterance, case paraphrase, dialogue, or proposed reply. A changed factor assignment, missing pair, quoted text, repeated card, or malformed declaration fails assembly. These structural checks cannot determine whether a pair actually changes only the stated fact.

Assembly also checks that each card has at least 20 letters and that at least half its letters use Latin script for `ro` or Cyrillic script for `uk`. This catches an obvious language-tag mix-up while allowing short symbolic field names. It does not establish Romanian or Ukrainian grammar, naturalness, or meaning; the fluent authors and reviewers must judge those.

## 2. Assemble a private candidate

When both workbooks return, the owner runs:

```sh
roguard-author-kit assemble --author-a review_runs/batch-XXXX-kit/author-a.json --author-b review_runs/batch-XXXX-kit/author-b.json --owner-assignments review_runs/batch-XXXX-kit/owner-assignments.json --output-dir review_runs/batch-XXXX-candidate
```

The output remains private and unapproved. It contains 96 source rows, a readable preview, author and factor provenance with all safety-review fields set to false, a pending review record, an assembly manifest bound to both workbooks, the original owner assignments, and the same model freeze, and `content-review-template.json`. The manifest includes a content-free overlap audit. Its top-level `flagged` list gives 1-based candidate card numbers, prior batch names, and normalized character five-gram similarity. The nested `within_candidate_cross_pair` report identifies near copies among different pairs in this candidate, excluding each pair's deliberate negative/positive mate. Neither report copies card text. A score of at least 0.8 flags wording for manual inspection, not automatic rejection or proof of semantic leakage. The command refuses an output inside tracked repository paths. Its result explicitly says `approved_for_repository: false`.

## 3. Review, declare, then seal

A **different** content reviewer inspects every candidate card and adjacent pair, checks the no-realistic-content rule, confirms the declared factor and balance, and decides whether to approve. They inspect prior-batch and cross-pair overlap flags against the private preview and, where relevant, earlier batches. Repeated abstract scaffolding can legitimately trigger many flags, while a low similarity score does not establish semantic novelty. The reviewer should ask whether ostensibly different pairs make the same lexical shortcut available to the model. If anything fails, revise privately and assemble a new candidate; leave review flags false until checks are genuinely complete. The reviewer copies `content-review-template.json` to a private declaration file, fills their own pseudonymous `reviewer_id` and ISO `reviewed_at` date, and changes each of the five review flags, including `prior_overlap_checked`, to `true` only after completing **both** overlap checks. The template binds the exact assembly manifest, batch, preview, and overlap audit. Preserve the original author workbooks and reviewer declaration privately as provenance evidence.

Only after that separate declaration exists, run:

```sh
roguard-author-kit seal --candidate-dir review_runs/batch-XXXX-candidate --review-declaration review_runs/content-review.json
roguard-independent-check --batch batch-XXXX
```

`seal` refuses changed candidate bytes, model-freeze bytes or named code and weight files, a changed prior-card corpus or overlap audit, a reviewer matching either author ID, a review dated before authorship, missing review flags, reused batch IDs, and failed taxonomy or batch checks. It writes the four approved batch files under the owner's local `data/synthetic/` and removes its newly written files if final validation fails. Follow the [prediction-seal order](STUDY_PREDICTION_SEAL.md): commit and push the opaque blind prediction **before** committing or pushing these labeled batch files. Then follow the blinded adjudication steps in the [intake contract](INDEPENDENT_BATCH_INTAKE.md). Commit only approved abstract batch files and suitable hash records, never private workbooks, reviewer contact details, or realistic source material.

Neither a completed workbook nor a passing seal proves when a person actually authored cards, their independence, reviewer competence, semantic one-fact validity, or model accuracy. The freeze binding prevents accidental model or evaluation-code drift; human chronology and the identity of the eventual prediction file need separate evidence. Author-intended labels are not adjudicated labels.
