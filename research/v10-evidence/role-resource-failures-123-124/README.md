# Failed bounded resource pilots123 and124

Study123 stopped before checking a case because the runtime tokenizer wrapper
was not callable by the offset-alignment helper. The versioned fix uses the
underlying fast tokenizer for offsets and independently compares its IDs with
the actual runtime wrapper. Two reviewers reran the real-wrapper tests.

Study124 then checked seven Romanian cases. On the eighth, native-final versus
serving score drift was8.484203506842897e-9, above the frozen1e-10 tolerance.
The run stopped with exit1; Ukrainian cases were not run. The tolerance was not
relaxed. Partial measurements and both exact sources are retained. No completed
16-case audit, accuracy improvement, fitting or broad extraction is claimed.

A separate proposed architecture keeps the exact qualified final-state path
for baseline predictions and adds native pooled states only as residual inputs.
This must account for the cost of both paths; it is not a silent repair or an
already validated model. Neither pilot qualifies a release.
