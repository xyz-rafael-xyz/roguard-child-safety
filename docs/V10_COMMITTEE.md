# Experimental shared-encoder review committee

The committee is a binary review component under development. It has passed the
recorded decision-combination checks and all 6,002 fresh installed-runtime
inferences; it is not a qualified v10 release.

It combines the published v9 classifier, semantic readout 11, and attention
adapter 14. Each retains its frozen threshold. A review proposal requires all
three positive decisions. This can remove false alerts, but can also miss a
concern when any component misses it. A negative result does not establish
safety. Component disagreement remains visible in the output.

## Recorded comparison

| Exposed population | Committee errors | Fixed v9 errors | New errors |
|---|---:|---:|---:|
| Eligible development, 6,668 inputs | 0 | 2 | 0 |
| Broad regression, 5,022 checkpoints | 0 | 2 | 0 |
| Original regression, 600 checkpoints | 0 | 0 | 0 |
| Challenge, 32 checkpoints | 0 | 0 | 0 |

These populations have been used during development. The agreement hypothesis
was motivated by exposed errors, and the earlier two-model rule failed its
development gate. The table therefore establishes an engineering result, not
independent superiority or unseen recall. The populations are not asserted to
be independent samples of one another.

[`review-unanimity-28`](../research/v10-evidence/review-unanimity-28/report.json)
retains all component decisions and the frozen rule. The separate
[`assessment`](../research/v10-evidence/review-unanimity-assessment-28/report.json)
recomputes the votes, gold labels, fixed errors and new errors from bound parent
records. Fresh installed-runtime verification in
[`experiment 29`](../research/v10-evidence/committee-runtime-29/report.json)
reproduces every component score exactly, with zero binary errors across all
6,002 checkpoint inferences. The run took 1,655 seconds on the local MLX
environment; its batch runtime is not a per-request latency benchmark.

## Runtime

`roguard.gemma4_committee.Gemma4ReviewCommittee` accepts three explicitly
configured, checksum-bound `Gemma4ReviewModel` objects for one language. Loading
checks their tensor inventories. Readout 11 may change only its language head;
attention 14 may additionally change the final 18 layers' query/output adapters.
Changes to the shared body or inactive language head are rejected.

The runtime captures the frozen body once per prefix, applies v9 and readout 11
to the original representation, then replays the attention tail on that same
captured state. It restores original weights even after an inference exception.
A shared inference lock protects these temporary overlays. Conversation text
is not retained in a cache between calls.

The component accepts at most 12 turns and 1,024 model tokens, matching the
underlying contract. Token overflow produces an explicit abstention that requests
review. The longer-conversation experiment supplies the same separately bound
memory view used by every comparator. Earlier unresolved reviews in a durable
session are not closed by later negative predictions.

The top-level `score` is a binary unanimity indicator, not a calibrated
probability. Each component's score, decision, logits and identity are recorded.
Outputs remain unqualified and authorize no notification or external action.

## Verification and artifacts

The focused tests cover every vote combination, focal-prefix isolation, invalid
outputs, abstention, and forbidden tensor changes. A small actual Gemma model
checks shared replay against separate full forwards and checks restoration after
an injected attention failure. Both languages' real 80-tensor inventories also
pass the compatibility check.

An isolated wheel installation passes all 20 committee and bundle tests on the
local neural environment. Portable installed smoke checks pass on Python 3.9,
3.10 and 3.12. The full-model run verifies all 6,002 exposed checkpoint outputs
against the frozen parent scores and decisions. Both hosted CI runs for the
bundle implementation pass all 16 jobs, including Windows and Linux checks.

The research adapters remain outside Git; the public metadata and training
entry points are in [neural reproduction](V10_NEURAL_REPRODUCTION.md). No new
package-index version or default model profile is published by this experiment.
Semantic sets, response checking, fresh calibration, independent final evaluation
and measured reviewer benefit remain separate qualification requirements.

## Local research bundle

After fresh runtime verification passes, `scripts/package_committee_v10.py`
copies the exact six adapters and metadata files into a new directory. It checks
that both packaged committee identities match the recorded runtime evaluation.
The pretrained Gemma base stays separate. The manifest identifies this as an
unqualified research component, and the exporter refuses to overwrite a bundle.

The installed package can load a relocated bundle with:

```python
from roguard.committee_bundle import load_committee

model = load_committee(
    "research-bundle/committee.json",
    expected_sha256="<manifest SHA-256 from the packaging report>",
    base="/path/to/the/pinned/gemma-base",
    language="ro",
)
result = model.predict({
    "language": "ro",
    "turns": [{"role": "minor", "text": "Am nevoie de ajutor cu tema."}],
})
```

The equivalent JSON CLI is `python -m roguard.committee_bundle --bundle ...
--sha256 ... --base ... --input conversation.json`. Loading validates all six
adapter checksums and rejects missing assets, altered metadata, mixed languages
and paths escaping the bundle. It performs no download. Full inference requires
the existing Apple Silicon MLX environment and the pinned pretrained base.

The [packaging record](../research/v10-evidence/committee-packaging-29/report.json)
binds every exported file. A second installed-wheel verification loads this new
bundle and performs 21 fresh inferences spanning every observed vote pattern by
population/language and every fixed v9 error. Every output field matches the full
runtime study exactly. The
[verification record](../research/v10-evidence/committee-bundle-runtime-29/report.json)
also checks all 171 installed package files against the wheel. This is a packaging
and parity check on existing cases, not another accuracy population.
