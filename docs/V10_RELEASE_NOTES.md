# RoGuard 10.0.0 release record

RoGuard v10 adds an opt-in Romanian and Ukrainian review path. It reuses the pinned v9 Gemma 4 body and review head, extracts the final classifier token and focal message features, and applies four packaged, checksum-bound radial-basis-function heads. A guarded decision rule combines the v9 proposal with the new binary and concern margins. Narrow, auditable scope rules handle explicit basic-need deprivation, denial and media references. The other five interpretation fields use v9 logits with explicit v10 scope-cue corrections; they have no newly trained heads or standalone accuracy claim. The v9 command remains the default.

The v9 runtime file remains byte-for-byte identical to the v9.0.0 tag. The v10 wheel contains the four small heads; the approximately 5.18 GB v9 base bundle is downloaded separately with `roguard-fetch fetch`. Native model review was validated on Apple Silicon with Python 3.12. Model margins are not calibrated probabilities, and a review proposal neither verifies an event nor authorizes notification or other external action.

## Measured comparisons

| Authored synthetic population | V9 binary errors | V10 binary errors | V9 concern errors | V10 concern errors |
| --- | ---: | ---: | ---: | ---: |
| Romanian, 60-case policy-corrected panel | 6 | 0 | 6 | 0 |
| Ukrainian, 60-case policy-corrected panel | 6 | 0 | 6 | 0 |
| Romanian, 20-case fresh challenge O | 3 | 0 | 2 | 0 |
| Ukrainian, 20-case fresh challenge O | 2 | 0 | 2 | 0 |

The same installed v13e wheel scored all 160 cases above. Challenge O was authored and frozen before the preceding challenge N's result was inspected. It was not used to train the four heads or tune the final decision rule. V10 fixed five v9 binary and four v9 concern errors on O; v9's panel errors were false alerts, so the panel demonstrates improved specificity rather than recall. The earlier 120-case panel was also held out from head fitting.

The original panel labels mistakenly marked 20 hypothetical personal concerns negative. A separate correction manifest records each case and the preexisting policy basis; the original labels and first evaluation remain available. The final four heads fit 18,331 Romanian and 21,345 Ukrainian exposed cases with zero training errors per head. Raw heads still had a few errors on another diagnostic hold; the guarded final decision replayed that and the previously exposed A–N challenges without binary or concern errors. These replays are development checks, not independent validation.

## Failures and limits

Fourteen successive fresh authored challenges, A through N, exposed errors in earlier v10 candidates. Challenge N first found three binary misses and one concern miss; the v13e wheel fixed all four on replay. The [complete challenge record](../research/v10-evidence/fresh-self-challenges-20261110/README.md) retains each frozen source, original result, and the final N and O results. Earlier prompt, query-only feature and shortcut-fit candidates were rejected after failures. The passing O result should be read alongside that selection history.

All these examples are project-authored synthetic messages. They do not demonstrate perfect accuracy on real children's language. No independent case-level human annotation record was supplied; informal reviewer approval is recorded only as approval to report the model and its limitations transparently. The six-field interpretation task has no new standalone qualification for its five fields derived from v9 logits. The v10 path does not replace the v9 default or close a prior concern based on one negative prediction.

The [release evidence](V10_RELEASE_CANDIDATE.md), [independent review protocol](V10_INDEPENDENT_REVIEW_PROTOCOL.md), [earlier committee record](V10_COMMITTEE.md), and [research history](V10_RESEARCH.md) give the reproduction details and remaining validation work.
