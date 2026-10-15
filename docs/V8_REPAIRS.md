# v8 input and interpretation repairs

These repairs address reproduced runtime errors. The separate encoder training
study continues under `research/v8-stabilization/review-03`; it does not change
the bundled model until evaluation supports a replacement.

## Equivalent text must produce equivalent predictions

The evidence character model previously processed decomposed Romanian and
Ukrainian accents differently from their composed equivalents. The context
character model also omitted the fitted character analyzer's whitespace
normalization. Both models now normalize their internal feature inputs to NFC;
the context analyzer also collapses repeated whitespace as its training analyzer
does. Original input text and Unicode code-point references remain unchanged.

Candidate repeated-claim links now compare canonically normalized text while
keeping offsets into the original text. These links remain unverified proposals.

A deterministic sample of 400 previously inspected v8 synthetic test cases
(200 per language) was checked under decomposed Unicode and repeated-space
transformations. Each comparison includes the complete prediction, including
attribute sets, rather than only the final action.

| Model / variation | Romanian mismatches before → after | Ukrainian mismatches before → after |
|---|---:|---:|
| Evidence / decomposed Unicode | 150/200 → 0/200 | 76/200 → 0/200 |
| Evidence / repeated spaces | 0/200 → 0/200 | 0/200 → 0/200 |
| Context / decomposed Unicode | 71/200 → 0/200 | 51/200 → 0/200 |
| Context / repeated spaces | 126/200 → 0/200 | 102/200 → 0/200 |

There were no prediction changes on the original 400 inputs. This demonstrates
invariance to these transformations, not correctness of every original prediction.
The full record is `research/v8-stabilization/surface-regression.json`.

## Explicit denials must not become affirmative rule evidence

The bounded support and basic-needs rules could flag statements explicitly
introduced as false. Repairs recognize Romanian and Ukrainian denial scopes and
apply them to the food-absence condition as well as the support/care conditions.
A later affirmative sentence or contrast clause can still produce its own signal.
Negation inside an affirmed concern, such as having no food, remains intact.

Regression tests cover explicit denials of support, food and care claims, later
affirmative clauses, and genuine concerns preceded by ordinary negative wording.
These are bounded lexical rules, not general-purpose parsing; model proposals
remain separately reported and no rule authorizes an external action.

## Interpretation labels

The Ukrainian evidence interface no longer labels an affirmed statement as
“confirmed.” It describes what the speaker asserts, without suggesting that the
system verified it. The Romanian label is also explicit about assertion, and the
role selector's accessible label follows the selected language.

## Verification and reproduction

The full Python suite ran 435 tests: 421 passed and 14 historical private-artifact
tests were skipped. The evidence JavaScript passes Node's syntax check. Runtime
hashes are recorded in `research/v8-stabilization/normalization-freeze.json`;
historical release freezes are retained unchanged.

```sh
PYTHONPATH=src:. python -m training.surface_regression_v8 --output /tmp/roguard-surface-regression.json
PYTHONPATH=src python -m unittest discover -s tests -v
```
