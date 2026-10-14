# RoGuard 7.1: bilingual false-alert repair

This update targets the optional **joint context model**. It does not retrain the older `adapted` conversation backend or the character/semantic comparison models. The decision-logic and bounded-language-rule fixes apply to all context backends. All language evidence below is synthetic and authored within the project; it does not establish near-perfect performance on real child messages or independent translation quality.

## 7.1.1 negation correction

A final rule audit after 7.1.0 publication found that Romanian negated support and solitude statements could match affirmative rules. Version **7.1.1** adds conservative clause-level negation checks in both languages, whitespace normalization, and tighter absent-care wording. Negated exits from fiction cannot trigger the quotation-conflict rule. This corrects **rule-derived** alerts; model predictions remain independent and may still request review.

Patch verification: **402 tests run: 388 passed, 14 skipped**; fresh-wheel negation and affirmative phrase checks passed. The model weights and calibration are unchanged. All six existing decision suites were replayed on their saved prefix predictions with **identical counts** to 7.1.0. Added tests cover negated support, negative belief wording, negated solitude, unrelated “nobody” statements, affirmative contrasts, and negated fiction exits. The new policy freeze is `release-freeze-7.1.1.json`; the original freeze is preserved. Final3 was fresh for the 7.1.0 evaluation; the patch replay is regression evidence.

## What changed

- Added ordinary task-help, theatre-writing, hobbies, scope-only introductions, correction/quotation contrasts and emotional-support examples in Romanian and Ukrainian.
- Paired ordinary contexts with explicit transitions to personal concerns, so a schoolwork or fictional preface cannot be treated as permission to dismiss a later disclosure.
- Increased the concern-mention objective weight from 0.25 to 1.0. Development selection considers both tasks and both languages; calibration remains separate.
- Recalibrated the selected concern head using label-conditional split conformal at a 95% synthetic calibration target. Frame calibration stays pooled within language at 90%. This is a calibration target under exchangeability, not a real-world accuracy guarantee. The existing requirement that a decided singleton agree with the probability winner remains enforced.
- Fixed a policy contradiction: a positive concern prediction paired with an `ORDINARY` frame now requests review as model disagreement instead of silently clearing the turn.
- Added transparent, separately reported language-rule signals for explicit emotional-support requests, a conjunction of prolonged lack of food and absent care, and explicit personal-context contradictions of a positive quotation prediction. These are bounded patterns, not verified facts. Known quotation frames are excluded from the support/basic-needs rules; quotation conflicts require all three conditions.
- Added `review_kind` to distinguish model uncertainty from a detected concern. Both can request human review. A harmless turn no longer displays review instructions. No earlier signal is automatically erased.

## Iterations

The released v7 model is the baseline. The old factorial and session probes are **regression sets**, already inspected before this work.

| Candidate | Romanian benign alerts / 126 | Ukrainian benign alerts / 126 | Non-quoted concern reviews, RO / 90 | Non-quoted concern reviews, UK / 90 | Disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| v7 baseline | 20 | 29 | 90 | 90 | Comparison |
| 02 | 12 | 33 | 90 | 87 | Rejected: Ukrainian regression |
| 03 | 0 | 5 | 89 | 90 | Rejected: missed Romanian third-person concern |
| 04 | 3 | 16 | 90 | 90 | Improved weights; overly frequent abstentions |
| 05 | 0 | 2 | 90 | 90 | Rejected after new-topic and stress tests |
| 07 | 1 | 2 | 90 | 90 | Revised training/calibration; remaining transfer misses |
| 09 (released) | 1 | 2 | 90 | 90 | Same model as 07, with separately reported rules and policy fixes |

An initial run was stopped before completing an epoch after manual inspection found ambiguous topic-transition labels. The corrected examples explicitly distinguish a shift to the speaker's actual experience from continuation of fictional dialogue. No test predictions had been inspected at that point.

The exact sampled training rows, rejected-candidate results, selected epochs, model hashes and test freezes are retained in [`research/v7-1-context`](../research/v7-1-context). Repeated templates and oversampling are not independent observations.

Candidate 05 reduced new-topic benign alerts from 42 to 10 in Romanian and 52 to 6 in Ukrainian (97 benign cases per language), but missed one Romanian and three Ukrainian topic-shift concerns. It also missed one Ukrainian unmet-needs case in the 24-row direct stress test. These failures prompted topic-transition and direct-concern training. The initial new-topic/stress sets became regression sets after inspection; a separate 40-row final set was frozen before candidate 06 training.

## Released results

Counts include model abstentions as review requests. Previous failures remain in the records; inspected sets are regression evidence.

| Set | Romanian benign alerts, v7 → v7.1 | Ukrainian benign alerts, v7 → v7.1 | Non-quoted concerns reviewed, RO | Non-quoted concerns reviewed, UK |
| --- | --- | --- | --- | --- |
| Original factorial regression | 20/126 → 1/126 | 29/126 → 2/126 | 90/90 → 90/90 | 90/90 → 90/90 |
| Initial new-topic set, now regression | 42/97 → 10/97 | 52/97 → 7/97 | 12/12 → 12/12 | 10/12 → 12/12 |
| Direct stress set, now regression | 2/4 → 1/4 | 2/4 → 0/4 | 8/8 → 8/8 | 8/8 → 8/8 |
| Original session regression | 1/6 → 1/6 | 1/6 → 1/6 | 5/5 → 5/5 | 5/5 → 5/5 |
| Final2, now regression after fixes | 2/7 → 0/7 | 3/7 → 0/7 | 10/12 → 12/12 | 11/12 → 12/12 |
| **Final3 fresh decision spot check** | **4/6 → 1/6** | **2/6 → 0/6** | **6/6 → 6/6** | **6/6 → 6/6** |

Candidate 08 initially missed three non-quoted final2 cases. Candidate 09 adds qualified emotional-support wording and explicit fiction-to-personal conflict checks; final2 is therefore regression evidence. Final3 was frozen after those changes and before inference. It is a tiny same-author synthetic spot check, not independent native-speaker validation. No fitting or policy changes followed final3 results.

The original session set still has one benign alert in each language. Final2 also retains a Romanian review request on its one explicitly fictional concern case (both versions flag it); its Ukrainian counterpart is not flagged. Zero benign alerts on a small subset does not mean zero false alerts generally.

Final policy changes were rechecked on existing prefix predictions using `eval/replay_context_policy_v71.py`; these replay records preserve the source result hash and model identity. Final2/final3 used actual encoder inference. The final comparison baseline uses the archived v7.0 policy, not the revised rules applied to old weights.

## Classification tradeoffs

On the same original-case regression rows, Romanian frame macro F1 changes from 0.9902 to **0.9412** and Ukrainian from 0.8726 to **0.8999**. Concern-mention argmax macro F1 changes from 0.9785 to **0.9914** in Romanian and from 0.9311 to **0.9216** in Ukrainian. On the less templated final2 set, released raw frame macro F1 is only **0.4800 Romanian / 0.4855 Ukrainian**. The separate rule layer does not change those model predictions.

The update improves conversation-level review behavior; it does **not** improve every classification metric or make frame interpretations reliable facts. Do not summarize these results as a single overall accuracy number. The fresh final3 check has decision-level labels only, so it is not scored as a seven-way frame benchmark.

## Evaluation boundaries

The new 218-row topic set holds out benign content and topic-transition wording. Its concern phrases and frame wrappers reuse historical v7 held-out vocabulary: it is not a wholly independent language benchmark. A separate 24-row stress set introduces direct concerns and benign contrasts. Both were initially frozen before inference and excluded from fitting. Their inspected failures informed subsequent repair design, so their final results are regression evidence. New final2 and final3 files have separate freezes and provenance. The old 24-session probe is regression evidence.

Benign alerts count all `REVIEW` outcomes, including abstention. Concern review counts measure routing for human inspection, not verified harm detection. Quoted passages appearing before their fictional context can still trigger a retained earlier review; quotation totals must not all be treated as errors.

The seven-way frame task remains imperfect, and performance on multilingual slang, misspellings, real child disclosures and long contexts is not established. The 256-token semantic limit and explicit unsupported Ukrainian transliteration handling are unchanged. Human review and source permissions remain independent of model output.

## Reproduce

Use the pinned encoder installed by `roguard-model fetch`, Python 3.10+ and the `adapted` extra plus scikit-learn for evaluation. Set `BASE_MODEL` to its local snapshot directory; replace `mps` with `cpu` or a supported CUDA device as appropriate.

```sh
PYTHONPATH=src:. python -m eval.context_v71 --base "$BASE_MODEL" --device mps \
  --artifact src/roguard/context-joint.json.gz --suite regression --output /tmp/roguard-regression.json
```

Other suites: `fresh` (now regression), `stress`, `session`, `final2` (now regression), `final3`. Pass `--legacy-policy` with `research/v7-1-context/baseline-joint.json.gz` to reproduce the original v7 policy comparison. Evaluators refuse to overwrite existing result files. Fresh and stress input hashes are checked before inference. Candidate training commands and hashes are recorded in the protocol and candidate records; do not mix a later generator revision with an earlier candidate's archived trainer when claiming exact reproduction.

## Installation and verification

```sh
pip install --upgrade 'roguard-child-safety[adapted]'
roguard-model fetch
roguard-studio --context --context-backend joint --open
```

The base encoder is unchanged; an existing pinned base download can be reused. The new adapter is bundled in the package. Core Python 3.9 support is unchanged; joint inference requires Python 3.10+.

Local software verification: **399 tests run: 385 passed, 14 skipped**. A fresh wheel installed on Python 3.9; artifact digests and bilingual rule integration passed. Six actual joint-backend cases from the built wheel matched the frozen results outside the checkout. Software tests, package checks and GitHub CI are separate from language-quality evaluation. Browser checks cover Romanian/Ukrainian rendering, mobile layout, reference navigation, clearing stale reports, and separate no-review/uncertainty/concern/rule-signal presentation.
