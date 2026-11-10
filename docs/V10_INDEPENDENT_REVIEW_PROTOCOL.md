# V10 independent review protocol

The authored 120-case panel selected the current candidate. Rechecking its labels can audit the policy and catch annotation mistakes, but it cannot qualify the model independently. The existing local blind packet drawn from that panel is a **label audit only**.

For release qualification, the Romanian and Ukrainian reviewers must provide a new case set whose text and labels have not been seen by model developers or used for model selection. Freeze its content and SHA-256 digest before running v9 or v10. Include positive, negative, quoted, fictional, denied, uncertain, hypothetical, past, third-party, and practical-help cases in each language. Preserve the natural distribution and report every stratum; do not discard hard cases after scoring.

Two fluent reviewers per language annotate the frozen inputs independently for `binary_review` and `concern`. They receive the policy definitions and input text, without model outputs or each other's labels. A third person adjudicates every disagreement while blind to model outputs. Preserve the original two labels, reasons, adjudicated label, and all exclusions with reasons. People fluent in both languages may cover both, but no one may adjudicate their own disagreement.

After adjudication, score the exact release-candidate wheel in a fresh installed environment and the pinned v9 baseline on the same frozen inputs. Record the wheel, model, dataset, and policy digests. Report confusion matrices by language and output, all individual failures, and the paired v9/v10 disagreement table. The user gate requires zero v10 errors on the accepted set, no regression against v9, and at least one v9 error corrected on that same set; a tie does not demonstrate improvement. Any v10 error fails the candidate and starts a new development cycle. A revised candidate requires a new untouched qualification set.

The five inherited structured fields need separate qualification before any improvement claim about them. The review heads are uncalibrated and do not confer permission for notifications or other external actions.
