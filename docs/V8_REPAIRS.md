# v8 input and interpretation repairs

These repairs address reproduced runtime errors. The separate encoder training
study continues under `research/v8-stabilization/review-04`; it does not change
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

The personal-deprivation rule also excludes explicitly named pet-food recipients.
Regression pairs retain the rule when the speaker lacks food personally, including
when a different sentence mentions available pet food.

## Evaluation partition isolation

A follow-up audit found shared ordinary-prefix inputs in the conversational
supplement: the first review-head study had one training/evaluation overlap and
five development/calibration overlaps. Its nonlinear study reused those inputs.
The first direct encoder pass had five overlaps between each pair of partitions.
Those old counts remain diagnostic records, not clean independent validation.

The corrected continuation removes duplicates by the exact normalized context
and focal input consumed by the encoder, checks for contradictory targets, and
retains the first partition's ownership. Its 20,199 training inputs, 4,845
development inputs and 4,846 calibration inputs have zero pairwise normalized
input overlap. This does not establish independent authorship or semantic-family
independence. The first completed encoder pass is only an initialization; the
corrected continuation supplies the selection and validation results.

## UTF-8 input on Windows

The evidence, context, operations, service-configuration and independent-review
commands now explicitly read UTF-8 files. Their previous locale-dependent reads
could decode Romanian or Ukrainian JSON incorrectly on a legacy Windows code
page. Installed-package checks exercise both languages and compare CLI results
against the same Unicode input passed directly to the Python API. CI includes
Windows Python 3.9 and 3.14 checks with UTF-8 mode disabled.

## Interpretation labels

The Ukrainian evidence interface no longer labels an affirmed statement as
“confirmed.” It describes what the speaker asserts, without suggesting that the
system verified it. The Romanian label is also explicit about assertion, and the
role selector's accessible label follows the selected language.

## Verification and reproduction

The full Python suite ran 440 tests: 426 passed and 14 historical private-artifact
tests were skipped. The evidence JavaScript passes Node's syntax check. Runtime
hashes are recorded in `research/v8-stabilization/normalization-freeze.json`;
historical release freezes are retained unchanged.

```sh
PYTHONPATH=src:. python -m training.surface_regression_v8 --output /tmp/roguard-surface-regression.json
PYTHONPATH=src python -m unittest discover -s tests -v
```
