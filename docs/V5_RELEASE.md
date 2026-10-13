# RoGuard 5.0: interaction testing and permission-aware execution

Version 5 implements all five approved directions: branching scenario replay and reduction; source restrictions that survive transformations; minimum necessary disclosure; paired decision tests; and useful-task preservation metrics. The [guide](INTERACTION_LAB.md) contains browser, CLI and Python examples. The [protocol](V5_LAB_PROTOCOL.md) was committed before final evaluation.

## What is implemented

| Component | Working capability | Boundary |
|---|---|---|
| Simulation laboratory | 12 bilingual scenario families, forward branches, fresh callable or recorded targets, first-failure traces, bounded counterexample reduction | Browser and CLI simulate actions; Python target adapters are explicit trusted integrations |
| Provenance and permissions | Immutable source IDs, derived-source unions, purpose/recipient/operation scopes, expiry, pause/resume, live revocation, host-bound callbacks | Complete provenance and authoritative policy must be supplied by the host |
| Minimum disclosure | Exact optimization over up to 18 candidate fields and 16 goal IDs, with explanations of retention/removal | Costs and coverage are declared, not semantic or legal judgments |
| Decision consistency | Invariance and directional transitions, expected labels, abstention handling, optional frozen-classifier surface probes | Agreement alone does not establish correct interpretation |
| Usefulness | Permitted-task completion, unnecessary refusal, unresolved requests, checkable arithmetic and violation/utility comparisons | Synthetic contract/task success does not measure benefit to real children |

Tool destinations and purposes are fixed by trusted bindings. External-operation bindings require host approval. Approval tokens are opaque, single-use, and bound to the exact tool, artifacts and current ledger revision. A change invalidates a previous token; dispatch rechecks permissions. Callback failures are uncertain rather than silently retried. Unknown external results retain their parent restrictions and are marked incomplete unless the trusted host explicitly verifies their provenance. The browser exposes no live callback route.

The existing v4 character models and semantic head are unchanged. This release adds exact mechanisms and evaluation infrastructure; it does not claim a newly trained language model or improved authentic-language accuracy.

## Registered controlled evaluation

The final seed is **20261025**. The suite contains **2,400 scenario variants**, 100 per family/language, from **12 policy families in two languages**. Each operating point encounters 4,000 requests, including 2,000 opportunities for permitted task completion and 600 checkable arithmetic opportunities. Variants share templates and change operands or harmless dialogue length; these are not 2,400 independent situations.

| Programmed reference behavior | Unauthorized proposals / 4,000 | Unauthorized simulated deliveries / 4,000 | Permitted tasks completed / 2,000 | Checkable tasks completed / 600 |
|---|---:|---:|---:|---:|
| Permissive, no enforcement | 2,400 | 2,400 | 1,600 | 200 |
| Permissive, enforced | 2,400 | 0 | 1,600 | 200 |
| Refuse everything, enforced | 0 | 0 | 0 | 0 |
| Minimum-disclosure reference, enforced | 0 | 0 | 2,000 | 600 |
| Reuse first choice, enforced | 1,200 | 0 | 1,800 | 600 |

These outcomes show the intended differences between permissive behavior, boundary enforcement, unnecessary refusal and minimum disclosure. The guarded reference follows the same exact proposal mechanism being demonstrated; its perfect reference score is **not independent AI accuracy evidence**. The unchanged unauthorized-proposal count under enforcement makes clear that stopping an action does not repair the target's reasoning. A permissive proposal containing unnecessary restricted fields can lose otherwise legitimate utility when the entire action is stopped.

There were no disagreements between the generated expected policy opportunities and the implemented policy checks. A separately written transition table matched the ledger on **5,456 bounded checks**: every one-to-five-event sequence of grant, revoke, pause and resume, evaluated at four time points including expiry. Dedicated tests also compare the disclosure optimizer with exhaustive subset enumeration and exercise derived-source revocation, incomplete provenance, approval replay, fabricated approval booleans, concurrent dispatch/revocation, malformed target outputs and callback uncertainty. This is bounded software evidence, not an unbounded proof or complete security assessment.

## Pairs and reduced failures

Twelve reference pairs cover four required policy changes and two invariant changes per language. The permissive target without enforcement passed 4 and failed 8; adding enforcement passed all 12. The overblocking target failed all 12 when both expected endpoints were checked, despite some decisions remaining consistent. The guarded and stale-choice integrations passed all 12. The stale target still failed longer temporal scenarios, illustrating why paired checks cannot replace sequence tests.

The registered stale-choice example was reduced from 8 to 6 events in 16 probes. The smaller scenario preserved the first failing event, reason codes and source identities. No additional single-event deletion within the valid scenario grammar preserved that failure. This establishes bounded one-deletion minimality for the observed target behavior, not a globally smallest counterexample or proof of determinism.

## Unchanged classifier diagnostics

A separate development-only diagnostic applies five surface changes to existing v4 development conversations. The initial direct-message sample contains 12 base inputs per language and 60 pairs per backend/language. The fast model produced 44 decided matching pairs and 16 unresolved Romanian pairs; Ukrainian had 55 matching and 5 unresolved. The semantic backend had 60 matching and no unresolved pairs in each language. No decided pair disagreed. Because this first sample uses only the direct form, a second recorded sample cycles through direct, ordinary, personal, fictional and retraction forms. In that second 60-pair-per-language sample, the fast backend had 50 matching/10 unresolved Romanian pairs and 48 matching/12 unresolved Ukrainian pairs, with no decided disagreements. The semantic backend had 60 matching/0 unresolved in each language. Both samples and all pair records are retained. Neither diagnostic changes models, establishes semantic equivalence or estimates accuracy.

## Verification and practical limits

Browser verification covers Romanian/Ukrainian simulations, minimum disclosure, counterexample download preparation, reference comparisons, classifier variants, mobile layout, stale-export invalidation and malformed input. A mobile overflow found during development was fixed by containing wide result tables; the same workflow then passed. Ordinary reports omit scenario values, while reduced reproducer exports explicitly contain their input text. The SDK ledger retains values in memory until closed; caller callbacks, files, browser extensions and logs remain outside its control.

The local logical clock must be advanced by the host. The gateway lock protects only that process's synchronous boundary; remote queues and external services need their own final authorization check. Callbacks are trusted code, not sandboxed programs. Host approval is an attestation, not authenticated identity or legal authority. New UI wording, scenarios and coverage declarations have not received independent language or specialist validation.

## Records and reproduction

- [Plan](V5_PLAN.md), [registered protocol](V5_LAB_PROTOCOL.md), [results directory](../research/v5-interaction-lab/).
- Core reference evaluation: `PYTHONPATH=src python -m eval.evaluate_interaction_lab --output NEW_FILE.json`. Use a clean full checkout; the command refuses to overwrite an existing record. The suite stores the source commit, corpus digest and implementation hashes.
- Classifier surface diagnostics: `PYTHONPATH=src python -m eval.evaluate_lab_model_consistency --base MODEL_PATH --sampling discourse --output NEW_FILE.json`. Install the semantic extra and the pinned base as documented. `--sampling direct` reproduces the earlier selection of direct forms.
- The base model and v4 fitted-artifact provenance remain in the [v4 release record](V4_RELEASE.md).

Prior work includes [Petri's automated interaction auditing](https://www.anthropic.com/research/petri-open-source-auditing), [CaMeL's separation of control/data flow](https://arxiv.org/abs/2503.18813), and established [privacy engineering](https://www.nist.gov/privacy-engineering). RoGuard's contribution here is the connected bilingual workflow, source-linked permission changes, exact disclosure proposals and joint violation/utility records. No generic first-of-its-kind claim is made.
