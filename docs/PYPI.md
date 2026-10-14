# RoGuard — Romanian and Ukrainian interaction testing and guardrails

RoGuard 7.1 provides a local bilingual laboratory for fictional AI interactions, source-linked permissions, minimum necessary disclosure, decision consistency and useful-task preservation. It includes an optional semantic text classifier and the existing declared-contract checks.

V7 adds an experimental, jointly trained conversation-context model, turn-reference review, separate concern-mention predictions, and permission-aware extractive tutoring proposals. [Context workflow and measured limitations](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/CONTEXT.md).

Version 7.1 retrains and recalibrates the joint context model, reduces synthetic Romanian/Ukrainian benign review requests, and adds separately reported language signals and clearer uncertainty handling. Errors remain, including frame-classification errors. [Measured improvements and limits](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/V7_1_RELEASE.md).

## Install and open

```sh
pip install --upgrade roguard-child-safety
roguard-studio --reliability --open
```

The core package has no runtime dependencies and supports Python 3.9–3.14. The browser and CLI simulate actions; they do not contact guardians or execute external tools. A separately configured Python gateway can call trusted host callbacks after live policy and approval checks.

V6 also adds authenticated tenant APIs, encrypted durable permissions and one-attempt queued execution, metadata-only monitoring/OpenTelemetry, overlapping long-context analysis, and permission-aware repair. Optional extras: `gateway`, `observability`, `adapted`. [Integration guide](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/RELIABILITY.md).

## What you can do

- Replay branching fictional scenarios against programmed references, recorded responses or your own explicit Python adapter.
- Identify the first failing step and reduce a scenario with a bounded replay budget.
- Retain source restrictions through summaries, proposed memory and tool results.
- Compute the exact minimum declared disclosure cost while covering required task fields.
- Test decision invariance and required policy transitions, retaining uncertainty.
- Compare permitted-task completion, unnecessary refusal and unauthorized proposals/deliveries.

```sh
roguard-lab demo --language uk --family revocation --target stale
roguard-lab compare --language ro
roguard-lab suite --fail-on-failure
```

For the new adapted text classifier, use Python 3.10+ (tested on 3.12):

```sh
pip install --upgrade 'roguard-child-safety[adapted]'
roguard-model fetch
roguard-studio --conversation --open --backend adapted
```

The pinned model download is approximately 1.2 GB; subsequent inference is local. V6 adds a LoRA-adapted encoder and calibrated abstention while retaining older backends. Its final test contains only six synthetic phrase families per language; no real-world child-language accuracy is established. Laboratory reference scores demonstrate controlled software behavior, not AI reasoning or human benefit. Provenance, coverage, costs and authority are host declarations; callback code is trusted, not sandboxed. Reduced scenario exports contain their input text.

[GitHub and full README](https://github.com/xyz-rafael-xyz/roguard-child-safety) · [Interaction guide](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/INTERACTION_LAB.md) · [v5 evidence](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/V5_RELEASE.md) · [Classifier evidence](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/V4_RELEASE.md)

## Context evidence review (v7, experimental)

```sh
pip install --upgrade 'roguard-child-safety[adapted]'
roguard-model fetch
roguard-studio --context --context-backend joint --open
```

The two prediction heads distinguish conversational framing from whether a concern is mentioned. Candidate references identify whole turns, not verified evidence spans. Training and tests are fictional; actual child-language accuracy and benefit are not established. Ukrainian interpretation errors and abstentions remain. Source permissions and human approvals stay independent of model outputs. [Full v7 evidence](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/V7_RELEASE.md).
