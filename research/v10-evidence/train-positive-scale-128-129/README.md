# TRAIN-only positive scaling study 128 and saved-output audit 129

Study 128 selected one bounded positive logit scale per language using all
28,929 existing TRAIN rows and the unchanged study 75 head outputs retained by
study 117. Seven fixed candidates were evaluated. A candidate was eligible only
if it introduced no binary errors on previously correct TRAIN rows; selection
then minimized total binary errors and preferred the scale closest to identity.
The frozen scales are 2.0 for Romanian and 1.5 for Ukrainian.

The selected scales correct all 11 TRAIN binary false negatives, leaving zero
binary errors and no introduced errors. All nine wrong unary-field tuples remain:
three in Romanian and six in Ukrainian. The same TRAIN labels were used to select
and measure the scales, so zero TRAIN errors are not an independent performance
estimate or a semantic correction. Time logits, legacy binary logits, policy
couplings and decision thresholds remain unchanged.

Audit 129 independently checks all 202,503 saved candidate-row transformations,
outcomes, groups and selection decisions across the seven scales. It verifies
transformed bytes, all seven argmax predictions and tie sets, fallback status,
binary decisions and integer counts. Floating-point loss and Brier diagnostics
use a 1e-10 absolute tolerance. The audit does not regenerate the parent head or
load the encoder, and it makes no calibration claim.

These studies score no DEV or new probe rows. They do not establish DEV
improvement, native-serving parity, inherited joint-set calibration or a qualified
v10 release. The scales must remain frozen before any separately registered
DEV evaluation. Exact producer and auditor sources, path-redacted registrations,
reports and the frozen selection are included; private arrays are represented by
their original digests.
