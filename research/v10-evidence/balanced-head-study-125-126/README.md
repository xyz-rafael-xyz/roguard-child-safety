# Larger four-arm training125 and independent evaluation126

The fixed four-arm study completed 35,568 updates and 2,273,328 example
presentations using 576 distinct new FIT inputs. All 104 checkpoints were
retained. Eight selections used original exposed DEV only and froze before
trained-probe scoring. Independent evaluation replayed all checkpoints and
459,524 DEV outputs, verified selection, and scored the larger generated bank.

The primary candidate reduces transfer assertion errors to 106 Romanian and
102 Ukrainian. Matched control has 345 and 296; unchanged75 has 357 and296;
original v9 has 343 and274. Every per-partition binary/joint/assertion
nonregression comparison against the matched control passes. All 576 FIT rows
have zero binary and joint-field errors. These are measured improvements on
exposed generated engineering data, not independent final validation.

Escalation is rejected. Romanian original DEV joint-field errors rise to675
against control673. On the full40 panel, Romanian has zero binary but six
joint-field errors; Ukrainian has one binary and three joint-field errors.
The primary has13 original DEV binary errors in total and fails the user's
zero-error requirement. Secondary runs cannot replace the failed primary.

A separate root recount checks18,480 saved predictions,240 panel responses
and72 partition tables. It confirms all error counts and the four gate failures.
It does not replay optimizer steps or the encoder. Its first reporting attempt
failed because its hash helper received a string path; the preserved second
version fixes that reporting-only bug and reruns the complete recount.

Exact executable trainer/evaluator sources and path-redacted reports are
included. No model weights or private arrays are published. Package metadata
stays9.0.0 and no v10 release qualifies.
