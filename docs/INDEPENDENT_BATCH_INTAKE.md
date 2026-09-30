# Intake for independently authored abstract test cards

This path lets independent Romanian or Ukrainian authors supply a **new abstract D1 or S1 test batch** for blinded annotation. It does not admit real or realistic disclosure/grooming text, case paraphrases, dialogue, or proposed replies. A human content reviewer must inspect every card **before** it is placed in the repository. The software checks hashes, declared provenance, pair structure, and the taxonomy review gate; it cannot establish that the named people are independent or that their intended labels are correct.

The [offline author kit](AUTHOR_KIT.md) can prepare two private, balanced workbooks and assemble the four files below without putting unreviewed text in Git. It requires a separate content-review declaration before sealing those files into the repository. Manual authoring remains possible under the same contract.

## Prerequisites

1. Complete the [independent taxonomy review](REVIEW_GATE.md) for the batch's language. The loader rejects both Romanian and Ukrainian independently authored cards while that review remains pending.
2. Freeze the model, prompt, cutoff, and evaluation code before authors see the task. Give `roguard-author-kit new` the exact committed `--freeze-record`; it binds that record and every named file hash into the workbooks and sealed provenance. A previous test or development batch cannot become this new test. Human timing still needs independent evidence.
3. Engage **two different fluent authors** independent of the project. Each writes 24 adjacent negative-to-positive one-fact pairs in their own language. For D1, each author declares six pairs for each of the four D1 facts; for S1, each declares three pairs for each of the eight response fields. A separate content reviewer checks every card for the no-realistic-content rule, abstractness, intended pair logic, and declared factor balance. Record their actual identities, qualifications, independence, and review decisions privately; use pseudonymous IDs in Git. Do not claim that the software verified those facts.

## Batch files

Choose an unused `batch-XXXX` identifier that has no registered local generator. Place four files under the owner's local `data/synthetic/` after content review. Follow the [prediction-seal order](STUDY_PREDICTION_SEAL.md): commit and push the blind prediction before these labeled files.

| File | Required content |
|---|---|
| `batch-XXXX.jsonl` | Exactly 96 JSON objects. Use only `id`, `language`, `source_kind`, `split`, `text`, `labels`, and `origin`. Set one language (`ro` or `uk`), one source kind (`message` for D1 or `response` for S1), `split: "test"`, and `origin: "independent_abstract_v1"`. Each adjacent pair has intended labels `[]` then `["D1"]` or `["S1"]`. Text is an abstract attribute description, never speech. |
| `batch-XXXX.preview.md` | A review table with one row per card in the exact form `| ID | language | source_kind | comma-separated labels or none | text |`. No `|` or newline may appear in card text. |
| `batch-XXXX.provenance.json` | The declarations below, with 48 pair-author IDs and 48 pair-factor codes in pair order. Each author contributes 24 balanced pairs. The content reviewer must differ from both authors. |
| `batch-XXXX.review.json` | Human approval record with `status: "approved"`, `reviewer`, `reviewed_at`, `reviewed_ids` in source order, `language`, `batch_sha256`, `preview_sha256`, `taxonomy_sha256`, and `provenance_sha256`. `reviewer` and `reviewed_at` must match the safety review. Do not set approved before that review occurs. |

The provenance file has this structure; replace all identifiers and dates with real declarations. This **schema example is not a completed review**:

```json
{
  "schema_version": 2,
  "batch": "batch-XXXX",
  "language": "ro",
  "category": "D1",
  "model_freeze_path": "eval/prospective/ro-v16-registered-study-freeze-v5.json",
  "model_freeze_sha256": "digest-of-that-exact-record",
  "authors": [
    {"author_id": "author_a", "native_language": "ro", "independent_of_project": true, "authored_at": "YYYY-MM-DD"},
    {"author_id": "author_b", "native_language": "ro", "independent_of_project": true, "authored_at": "YYYY-MM-DD"}
  ],
  "pair_authors": ["author_a", "... 23 more author_a IDs ...", "author_b", "... 23 more author_b IDs ..."],
  "pair_factors": ["source_role", "... 47 more factor codes in pair order ..."],
  "safety_review": {
    "reviewer_id": "content_reviewer", "reviewed_at": "YYYY-MM-DD",
    "abstract_only_checked": true, "no_realistic_content_checked": true,
    "pair_logic_checked": true, "factor_balance_checked": true,
    "prior_overlap_checked": true
  },
  "label_status": "author_intended_not_adjudicated"
}
```

The `pair_authors` and `pair_factors` illustrations abbreviate their arrays; the actual file must contain **48 literal entries in each**. The freeze digest is also a placeholder: supply the verified record's real SHA-256. D1 factor codes are `source_role`, `safety_or_support_anchor`, `indirect_or_repeated_support_pattern`, and `explicit_support_request` (six of each per author). S1 factor codes are `acknowledgement`, `next_step`, `human_support`, `no_blame`, `no_secret_promise`, `no_leading_question`, `no_unsupported_guarantee`, and `no_dependency_pressure` (three of each per author). The validator checks these declared counts. It cannot read the card's meaning or establish that the named fact truly changed; the content reviewer must check each pair. Calculate file hashes with `shasum -a 256`. RoGuard's taxonomy digest includes a filename separator; compute it with `roguard.review.taxonomy_sha256`, not a plain SHA of the Markdown file. The loader checks that the preview, cards, provenance, model freeze, and taxonomy still match the approved bytes and that no card exactly duplicates another current or earlier batch card. It cannot establish semantic novelty or actual authorship chronology from those checks. Author-intended labels remain **unadjudicated** even after the content reviewer approves the batch.

The direct loader and offline author kit require each card to be 20–1000 characters, stripped, single line, and free of quotes, table separators, and category-code tokens. They also reject cards with fewer than 20 letters or fewer than half their letters in the script expected from the language tag: Latin for `ro`, Cyrillic for `uk`. This is a narrow structural check. It cannot distinguish English from Romanian, prove natural Ukrainian phrasing, or replace either independent language review or content review.

The offline kit computes a content-free overlap audit before content review. A character five-gram Dice score of at least 0.8 flags a candidate against an earlier batch or another pair in the same candidate; the nested `within_candidate_cross_pair` check excludes the intentional negative/positive mate. These are warnings, not rejections or proof that meaning was reused. The separate reviewer records `prior_overlap_checked: true` only after inspecting **both** warning lists and relevant earlier cards. `seal` recomputes the audit and refuses a changed prior corpus or candidate. For manually assembled batches, run `roguard-independent-check --batch batch-XXXX` and inspect its `prior_overlap_audit` before accepting the batch; the declaration still cannot prove human inspection. Exact repeated cards remain rejected by the loader.

After approval, run `roguard-independent-check --batch batch-XXXX`. It prints hashes and structural status without displaying card text. It explicitly reports that actual author independence, content safety judgment, and label truth are **not verified by software**. A failed check must be resolved before packet generation.

## Blind review and scoring

Run `roguard-blind-packets --batch batch-XXXX --category D1 --output-dir review_runs/batch-XXXX` (or `S1`) only after the approval record is complete. This makes two differently ordered packets with opaque IDs and no source labels. Keep `owner-map.json` away from annotators. Two fluent annotators independently answer their packets with `roguard-annotate-packet`; each returns the private answer JSONL and its `.packet.json` binding together. The binding records the exact packet bytes shown and prevents resuming against a changed packet. `roguard-review-audit` requires both bindings for an independent batch, checks alignment and agreement, and leaves a separate blinded adjudication stream for disagreements. The [human validation protocol](HUMAN_VALIDATION_PROTOCOL.md) specifies the order.

`roguard-pairs` refuses this batch while its labels are merely author intended. Give a separate model operator only `reviewer-a.jsonl`, never the owner map or source file, and have `roguard-predict-abstract` freeze opaque-ID predictions before the labeled files are pushed or adjudications are opened. Use `roguard-human-eval-registered` to bind that packet hash to the approved source, check that its prediction identifier matches the recorded pre-author freeze, and score the frozen predictions against separately adjudicated declarations. Run the [Git-order verifier](../eval/verify_independent_study_git_order.py) on the final history as a separate chronology check. Neither check proves remote push order or which code actually produced the file. The original `roguard-human-eval` command remains pinned for reproducibility of the earlier input-integrity record. For every adjudicated pass-to-review pair the evaluator checks whether the two structured declarations differ on exactly the declared factor; mismatched or multi-fact pairs remain visible and cannot count toward the prospective numeric target. It reports parse failures, incomplete judgments, human disagreement, and model errors separately. Even a strong result would concern abstract descriptions only; it would not prove sensitivity to authentic child disclosures or authorize automatic action.
