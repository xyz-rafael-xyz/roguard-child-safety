# Shared VALUE resource study 134 and saved-output audit 135

Study 134 completed all 16 synthetic cases while preserving the unchanged study
75 baseline through the qualified capture/replay path. It pools normalized
attention VALUE vectors already retained at global layer 23, without a second
encoder pass. These 1,024-dimensional values are not final hidden states. Native
qualified serving-score differences and repeated qualified feature/logit
differences were zero on all cases, with exact zero-residual byte identity. This
is bounded engineering evidence, not universal equivalence or learned accuracy.
No residual model was trained or evaluated.

Audit 135 independently retokenized all 16 inputs, recomputed all 80 pool means
from the 16 saved VALUE arrays, and replayed the study 75 head on saved qualified
final vectors using CPU arithmetic. The maximum pool-mean difference was
4.77e-7, within the fixed 1e-5 limit. Maximum CPU logit differences were 1.91e-6
in Romanian and 2.86e-6 in Ukrainian, within the 1e-3 tolerance, with zero
seven-head argmax or binary-decision differences. CPU score differences reached
1.42e-9 and 2.22e-8; these are not native 1e-10 score parity. The audit does not
replay the encoder or independently establish native VALUE origin, installation,
repeated-feature identity or resource measurements.

Within this run, mean primary versus baseline-component times were 1.04598 versus
1.04377 seconds in Romanian and 1.16539 versus 1.16374 seconds in Ukrainian.
The measured additional conversion and pooling components averaged about 2.22
and 1.65 milliseconds. These are paired component timings, not end-to-end
speedups. Loading, prefix construction, fresh repeats, policy checks, CPU audits,
host copies and persistence have separate costs. The complete producer took
138.997 seconds. Earlier study 127 had a baseline component near 0.205 seconds;
cache warming, host state and synchronization were uncontrolled across studies,
so their absolute timings do not establish a cross-study speedup. Only three
cases per language have different aligned and chronological prior masks.

The original study 134 registration was not executed. Before the actual run,
revision 2 corrected only root-review prose from 950 to 951 input bindings and
recorded that correction. All 951 bindings had already been checked; source,
inputs, numerical gates and limits stayed fixed. There was no failed actual run
associated with this registration correction.

Before study 135 was registered, independent synthetic tests found a missing
post-import input/source/artifact verification barrier. The preserved first draft
failed three injected-mutation cases; revision 2 rejects them before any tokenizer
call and passed all 30 independent tests. Actual study 135 used the corrected
source. This was a pre-registration integrity repair, not a change to pool means,
accuracy labels, tolerances or actual results.

Exact registered producer and auditor sources are included with path-redacted
reports, registrations, design and reviews. All original numeric, boolean, null
and digest values are retained. Private arrays remain represented by their
hashes. Neither study establishes predictive improvement, semantic grounding,
calibration or a qualified v10 release.
