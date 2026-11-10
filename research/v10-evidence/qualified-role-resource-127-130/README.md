# Qualified baseline resource study 127 and saved-output audit 130

Study 127 completed all 16 synthetic cases, eight in Romanian and eight in
Ukrainian. It retained the qualified capture/replay path and unchanged study 75
head for baseline predictions, while computing native pooled vectors separately.
The maximum qualified serving-score difference and repeated qualified feature
and logit differences were zero on these cases. This is bounded engineering
evidence; it does not establish universal equivalence, an accuracy improvement,
or a trained residual model. Native final states are not claimed to replace the
qualified baseline path.

The primary two-path phase took about 3.24 times the qualified baseline phase
in Romanian and 2.94 times in Ukrainian. These ratios compare measured phases,
not end-to-end latency or throughput. Loading, prefix construction and additional
audit calls have separate timings in the producer report. The small synthetic
sample does not establish production performance.

Audit 130 independently retokenized all 16 inputs and replayed the study 75 head
on the saved qualified-final vectors using CPU arithmetic. Maximum logit errors
were approximately 1.91e-6 and 2.86e-6, within its 1e-3 logit tolerance, with zero
seven-head argmax or binary-decision differences. Its maximum CPU score differences
were approximately 1.42e-9 and 2.22e-8. This CPU check does not claim native
1e-10 score parity or independently replay the encoder. Pooling, repeated-feature,
resource and installation observations are recounted producer evidence.

The first registration contained a null binding for the earlier study 120 report.
Preflight rejected it before output creation or encoder loading. Both registration
summaries and the redacted failure log are retained. Revision 2 supplies that
completed report binding without changing source, cases, heads, tolerances or
resource limits.

The exact producer and auditor sources are included with path-redacted reports
and registrations. Required private assets remain represented by their digests.
Neither study qualifies a public v10 release.
