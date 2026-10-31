# Independent v10 study handoff

The supplied practice packets are developer-authored synthetic exercises. They
are **not** independent validation. No human participation or identity has been
verified by the software.

## Roles and separation

Use Romanian- and Ukrainian-fluent adult scenario authors, two independent
annotators per case, and a separate adjudicator. Store identifying details outside
Git; use opaque participant IDs in data. Authors and generation sources for the
final test must be disjoint from training and development. Keep translations,
paraphrases and variants of the same scenario in one family and one partition.

Decide sample size from a separate pilot, the minimum useful paired effect and
family dependence before authoring the final test. The current four-family
diagnostic cannot supply an adequate population estimate.

## Authoring task

Write fictional conversations with ordinary task assistance and varied wording.
Cover long-range references, indirect concerns, person/time ambiguity, correction
versus topic change, fiction, reported accounts, informal language, code switching,
transliteration, and meanings outside the six-field ontology. Include benign
matched cases; avoid making every unfamiliar expression a positive example.

Use `roguard.trajectory_data.FORMAT`, roles `minor`, `assistant`, or `other`, and
at most 512 turns / 500,000 characters. A conversation must have explicit case,
family, author and generator identifiers. Labels belong in `checkpoints`, never
inside `turns`. Annotate multiple prefixes to measure detection delay; the final
label must never be copied backward onto earlier prefixes without review.

Each checkpoint records focal turn, review label, joint interpretation when
expressible, and supporting spans with turn/start/end code-point offsets.
Unresolved and outside-ontology judgments stay explicit; adjudication must not
force ambiguous language into a fabricated certainty. Current model experiments
do not qualify an outside-ontology detector.

## Annotation and adjudication

`trajectory_review.annotation_packet()` removes model outputs and expected
labels, shuffles items, and supplies a private coordinator key. Give each
annotator only their packet. `validate_annotations()` rejects changed inputs,
missing items and malformed spans. `annotation_agreement()` lists disagreements
for adjudication; agreement is not evidence of event truth or reviewer expertise.

Practice material is in `research/v10-evidence/reviewer-practice-kit`. The files
named `PRIVATE-key` are practice mappings, not credentials. Keep actual study
keys and participant returns outside the public repository.

## Frozen final comparison

Select the model and all hyperparameters on development data. Obtain fresh
calibration families for the final frozen pipeline: the current exploratory
calibration has already been inspected during development. Use `trajectory_study.freeze()` to bind model,
calibration and source artifacts plus the final test's file digest without parsing
its contents. The independent custodian controls access to that file.

`freeze()` also binds the exposure-log path; `open_once()` rejects an alternative
path and records exposure before parsing or evaluation. A failed run consumes
the test just as a successful run does. Any further improvements require a new
independent final test; the old one becomes regression evidence. Custodian-held
records are necessary because local hashes cannot prove blindness.

Run v9, v9 with simple retrieval, a full-context neural comparator and v10 on the
same prefixes. Report all abstentions, review workload, delayed/missed detections,
joint coverage, set size, evidence errors, paired family-bootstrap differences,
memory usage and latency. Review results by language and scenario group.

## Human review study

`reviewer_packet()` counterbalances model condition across paired reviewers and
shows each conversation once per reviewer. It removes model names and identities.
Use separate training/practice examples. Record active elapsed time, completion
and decision accuracy; interruptions remain incomplete instead of disappearing.
`score_reviewer()` reports descriptive condition summaries. Aggregate according
to a preregistered participant/case design before claiming an improvement.

Symbolic question costs are not measured human time. The planned comparison must
test whether assistance improves reviewer accuracy or effort at matched error
rates. No child-facing intervention or external notification is authorized by a
model result or this study tooling.
