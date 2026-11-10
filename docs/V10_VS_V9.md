# RoGuard 10.0 versus 9.0

This document describes the **published 10.0.0 release**, with emphasis on its optional Romanian and Ukrainian model review path. It distinguishes changes in that path from existing RoGuard features and from research components in the repository. Version 10.0.0 is available on [PyPI](https://pypi.org/project/roguard-child-safety/10.0.0/) and as a [GitHub release](https://github.com/xyz-rafael-xyz/roguard-child-safety/releases/tag/v10.0.0).

## The change in one paragraph

V9 uses a pinned Gemma 4 model and language-specific review head to propose whether a focal minor message needs human review. V10 keeps that model and head, then adds an **opt-in second decision path**. It reads both the final classifier-token representation and a mean representation of the focal minor message, applies four small packaged radial-basis-function (RBF) heads, and combines their binary and concern-topic margins with the v9 decision through a guarded rule. Explicit, narrow scope cues handle cases such as first-person food or water deprivation, denials and clearly framed media references. The default `roguard-review` command still runs v9; `--v10` selects the new path.

## What changed

| Area | V9 | Stable v10 addition |
| --- | --- | --- |
| Model body | Pinned Gemma 4 body with trained attention adapters. | Reuses the **same pinned body**; no new Gemma 4 training or replacement backbone in the released v10 path. |
| Readout features | Final classifier-token representation for the review proposal. | Uses that representation **plus the mean hidden state of the focal minor message's aligned tokens**. This gives the added heads a second, message-specific view. |
| Trained decision heads | V9 review head and auxiliary interpretation logits. | Four packaged RBF heads: a binary review head and a concern-topic head **for each language**. These are the newly fitted v10 heads. |
| Binary decision | V9 score and threshold. | A guarded combination of the v9 proposal and the v10 binary and concern margins. A v10 head does not unconditionally overwrite v9. |
| Scope handling | Existing v9 interpretation and review behavior. | Explicit, auditable cues for selected deprivation, denial and reference-scope cases. The output records which cues fired. |
| Concern topic | Derived from the v9 structured logits. | Uses the v10 concern head, with a narrow correction for explicit denied confinement and basic-need deprivation. A topic can be present when the binary decision is `no_signal`. |
| Other structured fields | V9 logits. | The other five fields still derive from v9 logits, with explicit v10 scope-cue corrections. They have **no new trained heads or separate accuracy qualification**. |
| Distribution | V9 base and adapters fetched explicitly. | The four small RBF heads ship in the 10.0.0 wheel; the approximately 5.18 GB pinned v9 base is fetched separately and reused. Model identities and assets are checksum-bound. |
| Entry point | `roguard-review` and the v9 Python model API. | `roguard-review --v10` and `Gemma4RBFV10Model.from_installed_v9(language=...)`. V9 stays the default. |

The feature extraction, decision guard, packaged asset checks and returned evidence are implemented in [`gemma4_review_v10.py`](../src/roguard/gemma4_review_v10.py), [`gemma4_rbf_v10.py`](../src/roguard/gemma4_rbf_v10.py), [`review_scope_v10.py`](../src/roguard/review_scope_v10.py) and the [model manifest](../src/roguard/v10_models/manifest.json). The v9 runtime file remains byte-for-byte identical to the v9.0.0 tag; the new path is selected explicitly.

## How a v10 review runs

1. The caller supplies a Romanian or Ukrainian conversation and a focal turn, just as for v9. Later turns are excluded from that focal input.
2. The unchanged pinned v9 model produces its logits and final classifier-token vector. The v10 feature reader also extracts the focal minor message's token mean from the same model pass, with token alignment checked before use.
3. The appropriate language's packaged binary and concern RBF heads each return a signed margin from the combined feature. These margins are **not calibrated probabilities**.
4. A guarded rule compares both margins with the v9 proposal. Stronger evidence is required to reverse an existing decision; the explicit scope cues can resolve the narrowly defined cases documented above.
5. The output reports the final binary proposal, concern topic and other structured fields, the v9 proposal, both v10 margins, cue evidence, input digest and bound model identity. An overlong input requests clarification rather than silently becoming a negative result. No proposal authorizes a notification or other external action.

The released guard uses a concern-margin cutoff of `0.0` for Romanian and `0.1` for Ukrainian. It can clear a positive v9 proposal only when the binary margin is at most `0.05` and either the concern margin falls below that language's cutoff or the binary margin is at most `-0.45`. It can raise a negative v9 proposal only when the binary margin is at least `0.35` **and** the concern margin exceeds the cutoff. Otherwise it keeps v9's proposal. Explicit first-person meal or water deprivation then requests review; a standalone meal-deprivation denial or a clearly framed, nonpersonal media reference can clear it. The concern field uses its own head and retains a topic for explicit denied confinement or basic-need deprivation. These are exact rules in the [released implementation](../src/roguard/gemma4_rbf_v10.py), not general language-understanding guarantees.

The binary output answers whether to **propose review under the declared policy**. The concern field answers whether a safety topic is present. For example, an explicit denial of confinement can retain `concern=1` while producing `binary=0`; those two fields have different meanings. See the [walkthrough transcript](V10_DEMO_VIDEO.md) for the recorded examples.

## Training and selection

V10 did **not** retrain the large Gemma 4 base or replace the v9 review head. The [release notes](V10_RELEASE_NOTES.md) report that the final four RBF heads were fitted on **18,331 exposed Romanian** and **21,345 exposed Ukrainian** project-authored cases per head, with zero training errors per head. Raw heads still made errors on a separate diagnostic hold; the final guarded runtime cleared that exposed set. Training-set fit and replay of an exposed diagnostic are engineering checks, not independent accuracy estimates. The fit report cited by the packaged manifest is not in the public Git tree.

Earlier prompt, query-only, committee and shortcut-fit candidates were examined and rejected where they failed their own checks. The [release notes](V10_RELEASE_NOTES.md) report that fourteen successive newly authored challenges, A through N, found first-pass errors in earlier v10 candidates. Challenge N initially found three binary misses and one concern miss; the final v10 wheel fixed those on replay. The passing final challenge must be read alongside that selection history. The older release notes link to a challenge-record directory that is **not present in the public Git tree**; readers cannot independently inspect those original challenge files there.

## Direct v9–v10 comparison on the released wheel

The comparison below evaluates the **same authored inputs under both versions**. Counts are **output errors**, not necessarily distinct cases: one case can contribute both a binary and a concern error.

| Authored synthetic population | Cases | V9 binary errors | V10 binary errors | V9 concern errors | V10 concern errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Romanian, policy-corrected panel | 60 | 6 | 0 | 6 | 0 |
| Ukrainian, policy-corrected panel | 60 | 6 | 0 | 6 | 0 |
| Romanian, fresh challenge O | 20 | 3 | 0 | 2 | 0 |
| Ukrainian, fresh challenge O | 20 | 2 | 0 | 2 | 0 |
| **Total** | **160** | **17** | **0** | **16** | **0** |

Thus the 120-case panel has **24 combined v9 output errors versus zero for v10**, and challenge O has **nine versus zero**. The panel's v9 errors were false alerts, so that panel demonstrates improved specificity on those inputs; it does not establish improved recall. The panel was held out from v10 head fitting. The [release record](V10_RELEASE_CANDIDATE.md) says its original labels mistakenly treated **20 hypothetical personal concerns** as negative and describes a correction under the preexisting policy. The correction manifest linked from that record is **not present in the public Git tree**, so that correction cannot be independently reconstructed from this repository alone. Challenge O was authored and frozen before challenge N's result was inspected and was not used to fit or adjust the final model, according to the release record.

The exact [public PyPI wheel verification](../research/v10-evidence/stable-release-20261110/publication.json) replayed both populations after installation. Its wheel SHA-256 is `c471e34518d9808b8d96e3ce759799e6e90679057c9169be8e341643ca1da34e`. The public [120-case result](../research/v10-evidence/stable-release-20261110/public-wheel-panel20b-result.json) and [40-case result](../research/v10-evidence/stable-release-20261110/public-wheel-challenge-o-result.json) contain case IDs, labels, predictions and margins, while the publication record supplies source and result hashes. The underlying case text is not included in those result files. V9's separate 600-case reserved evaluation and 5,022-case regression remain valuable v9 evidence, but they are **different populations** and should not be combined with the 160-case v10 comparison.

## What v10 inherits, and what it does not claim

- The core toolkit, conversation handling, permission checks, decision-assurance tools and local interaction lab existed before v10. The new RBF review path does not make these features new to v10.
- The default v9 review profile, threshold and public v9 behavior remain available. Installing 10.0.0 alone does not select the v10 model; it requires the optional extra, the explicit base-model fetch and `--v10` or the v10 Python API.
- The v10 path has no newly qualified response-generation model, long-conversation semantic memory guarantee, six-field interpretation score or independently measured reviewer benefit. Research prototypes for trajectory selection, committee review, response checking and durable sessions are documented in [v10 research](V10_RESEARCH.md); their experiments must not be mistaken for a qualification of those capabilities in the stable RBF reviewer.
- All quoted accuracy comparisons use finite, **project-authored synthetic messages**. There is no supplied independent case-level human annotation record and no demonstrated accuracy on real children's messages. Zero observed errors do not imply universal or real-world perfection.
- A model result is a review proposal. It does not verify that an event happened, close an earlier concern after one negative result, or grant permission to share data, contact anyone or take another external action.

## Other v10 research in the repository

The repository also contains v10-named research APIs and studies. They expand the research surface beyond v9, but **their presence in the source tree does not make them part of the qualified stable RBF reviewer**. The [research record](V10_RESEARCH.md) preserves their failures and experimental status; some older research pages were written before the 10.0.0 RBF release and still describe the earlier promotion state.

| Research area | Added capability or study | Release boundary |
| --- | --- | --- |
| Long-conversation trajectories | Prefix-level provenance, evidence offsets, split checks and explicit rejection of future evidence; a bounded extractive selector for focal and relevant earlier content. | The released RBF path retains the v9 input contract; longer trajectories are experimental. |
| Permission lineage | FlowLedger experiments propagate restrictions from every consulted source, including omitted sources, through derived views. | Hosts still bind source IDs and authorize use; this is not a new v10 model-accuracy claim. |
| Portable student | A standard-library sparse model explores review, interpretation, relevance and response heads without an accelerator. | Its response head and full pipeline did not qualify for the stable release. |
| Joint interpretation sets | Language-stratified, family-level split-conformal sets and a probability-floor numerical correction. | Historical sets were preserved, but deployment assumptions and fresh calibration remain unverified. |
| Shared-encoder committee | Three candidate models share one frozen body pass, retain component votes and expose disagreement. | A fresh authored diagnostic found seven misses for both committee and v9; the committee was not promoted as the stable v10 path. |
| Session state | An explicit review floor and durable SQLite sessions retain unresolved positives and abstentions across restarts, with host resolutions and atomic revision checks. | These host-facing research APIs do not change the released RBF review decision's evidence or accuracy claim. |
| Reviewer and promotion tools | Blinded annotation packets, disagreement records, counterbalanced studies, artifact binding and a single-opening final-test workflow. | No independently annotated case-level evaluation was supplied for the stable v10 model. |
| Response checking | Several neural, lexical and translated response experiments test permitted-task completion and privacy behavior. | Remaining errors prevented a new response-checker qualification. |

These lines of work are useful for research and integration, but their experimental measurements must be read in their own protocols rather than added to the released model's 160-case comparison.

## Install the new path

The core package supports Python 3.9–3.14 on macOS, Linux and Windows. Native v10 model review was validated on Apple Silicon with Python 3.12; the optional model extra requires that environment and an explicit download of the pinned v9 base.

```sh
pip install 'roguard-child-safety[gemma4-v10]==10.0.0'
roguard-fetch fetch
roguard-review --v10 --language ro --text 'Pot cere ajutor când sunt speriat.'
roguard-review --v10 --language uk --text 'Можу попросити допомоги, коли мені страшно.'
```

For applications, use `Gemma4RBFV10Model.from_installed_v9(language='ro')` from `roguard.gemma4_rbf_v10` after the explicit fetch. The [release guide](V10_RELEASE_CANDIDATE.md) and [v9 model guide](V9_MODEL.md) describe the installation contract and the inherited base.
