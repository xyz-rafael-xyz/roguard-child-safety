# Semantic teacher-mask ablation 131 and evaluation 132

Evaluation 132 passes its integrity checks and rejects escalation. Extending the
teacher disagreement mask to all six semantic fields on newly admitted FIT rows
did not improve the registered transfer criterion or preserve the required DEV
field accuracy. The earlier assertion-only control was reproduced exactly: all
26 control checkpoints and their saved DEV logit arrays are byte-identical to
study 125.

| Primary result | Romanian | Ukrainian |
| --- | ---: | ---: |
| Transfer assertion errors, out of 768 rows | 107 | 104 |
| Assertion-only control transfer errors | 106 | 102 |
| Original DEV binary errors | 11 | 1 |
| Original DEV six-field tuple errors | 684 | 641 |
| Control DEV six-field tuple errors | 675 | 634 |
| Full panel binary errors, out of 20 responses | 0 | 0 |
| Full panel six-field tuple errors | 6 | 2 |

Zero panel binary errors in both languages is a real improvement over the earlier
primary, whose Ukrainian panel had one binary error. It does not satisfy the
zero-tuple-error gate. Romanian joint DEV errors exceed the fixed ceiling of 673;
Ukrainian joint DEV errors exceed 634 and one binary error is newly introduced
relative to v9, against a ceiling of zero. Transfer assertion errors also regress
on the Romanian both-held block and Ukrainian operator-held block. The exact
22 failed predicate entries are retained; several repeat the same failure against
different references or fixed floors and are not independent experiments.

Study 131 used the same 576 unique generated FIT inputs, repeated 32 times,
alongside 28,929 original TRAIN rows. Both arms used the same 12-epoch schedule,
seed, optimizer, fixed thresholds and frozen encoder features. The control kept
the earlier assertion-only teacher mask. The treatment additionally suppressed
teacher KL terms whose semantic argmax disagreed with the authored labels on new
FIT occurrences only. Original TRAIN masking and the legacy binary teacher term
were unchanged, with the fixed seven-term denominator retained. The scheduled
total was 17,784 updates and 1,136,664 presentations across both arms; repetitions
are not independent examples. Removing contradictory terms also reduces
regularization, so this comparison cannot isolate those two explanations.

Original DEV selected the checkpoints before any new trained probe evaluation.
Evaluation 132 independently replayed all 52 checkpoints across 229,762 old DEV
row presentations, verified mask and training-order provenance, and recounted the
saved comparisons. A separate saved-logit recount checked 18,480 new-input
predictions, 240 mapped panel responses, 44,185 selected DEV predictions and all
1,332 grouped metric objects. Integer and reference-error counts agree exactly;
the maximum cross-entropy difference is about 3.55e-15. That additional recount
uses the pinned portable policy scorer and does not replay the head, encoder or
optimizer. Neither audit independently reproduces optimizer steps.

The 3,080 unique scoring inputs include 3,072 main-bank rows and eight supplemental
controls. All 40 panel responses remain visible, comprising 32 main-bank probe
aliases and eight supplements. They are not 40 additional independent examples.
The probe families were already exposed in earlier studies, and generated labels
have no independent human validation. No reserved TEST was opened, no secondary
candidate replaces the failed primary, and these studies do not qualify v10.

Exact registered trainer and evaluator sources and the exact semantic loss helper
are included. Reports, registrations, selection records and reviews have local
paths redacted while retaining original values and digests. The additional recount
script is explicitly path-redacted rather than claimed byte-identical. Private
arrays and checkpoints remain represented by their original hashes; the redacted
registrations are provenance records, not executable run plans.
