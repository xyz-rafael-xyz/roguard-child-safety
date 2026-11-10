# RoGuard v10 release candidate

RoGuard v10 adds a local joint feature reviewer for Romanian and Ukrainian. It reads the pinned v9 Gemma 4 model's final token feature and focal message feature, then applies two small, checksum-bound RBF heads for the review decision and concern field. The v9 model and threshold remain available through the existing interface. V10 is an opt-in candidate until independent review and publication are complete.

## Install and run locally

The v10 wheel includes the four RBF heads. It reuses the existing v9 model download, which must be fetched separately on Apple Silicon with Python 3.12 or later.

```sh
pip install 'roguard-child-safety[gemma4-v10]==10.0.0'
roguard-fetch fetch
roguard-review --v10 --language ro --text 'Pot cere ajutor când sunt speriat.'
roguard-review --v10 --language uk --text 'Можу попросити допомоги, коли мені страшно.'
```

The Python API is `Gemma4RBFV10Model.from_installed_v9(language='ro')` from `roguard.gemma4_rbf_v10`. The output is an uncalibrated review proposal. It does not verify an event or authorize a notification or other external action.

## Engineering result

The frozen candidate has zero binary and concern errors on all 39,010 exposed fit cases. On a separate 120-case authored mixed-scope panel, v10 had zero errors in both outputs for both languages. V9 had six binary and six concern errors per language on the same panel. An installed wheel replayed all 120 cases with the same results.

The panel was frozen before scoring. Its first authored labels mistakenly marked 20 hypothetical personal concerns as negative. The preexisting v9 policy explicitly counts hypothetical personal concerns as review proposals. A separate correction manifest records the 20 case IDs, original and corrected input hashes, and the policy hash; the original panel and first evaluation remain intact. Neither model was fitted on this panel.

The four packaged head hashes and the fit and panel report hashes are bound in `src/roguard/v10_models/manifest.json`. The source passed 64 focused package and runtime tests. An installed candidate wheel replayed the 120-case panel, and its four model assets matched the packaged manifest hashes.

These are project-authored synthetic cases. The result establishes an engineering comparison on this finite panel, not perfect accuracy on open-ended child language. The other five interpretation fields are inherited from the v9 parent and have no new standalone accuracy claim.

## Independent review and release gate

Two fluent reviewers per language should annotate independently, with a separate adjudicator resolving disagreements. The review packet should hide authored labels and model predictions, record every raw vote, and retain adjudication reasons. Reviewers should check label meaning under the pinned policy, especially hypothetical personal concerns, reported harm, fiction, negation, and positive statements about seeking support.

Public promotion remains closed until the independent labels are available, the installed wheel is checked against them and the v9 baseline, and no required output regresses under the accepted release gate. The v9 package remains the published stable release while this review is pending.
