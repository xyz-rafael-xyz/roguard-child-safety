# RoGuard — Romanian and Ukrainian interaction testing and guardrails

RoGuard combines local model review with explicit permission checks and a laboratory for fictional AI interactions. Developers can test whether a system preserves source restrictions, respects revocation, requests human review and still completes permitted tasks. The browser and CLI simulate actions; external execution requires separately configured host callbacks and live approval checks.

## Install

```sh
pip install --upgrade roguard-child-safety
```

The core toolkit has no runtime dependencies and supports Python 3.9–3.14 on macOS, Linux and Windows.

```sh
roguard-studio --reliability --open
roguard-studio --review --open
```

## V9 local model review

The optional v9 model proposes whether a Romanian or Ukrainian focal message needs review under the project's declared policy. It uses earlier turns to distinguish an ordinary topic change, a support signal and a revision of an earlier statement. Review is a proposal, not a finding that an event occurred or permission to contact anyone.

This model requires native Apple Silicon and Python 3.12 or later. In a separate virtual environment:

```sh
pip install --upgrade "roguard-child-safety[gemma4]"
roguard-fetch fetch
roguard-review --language ro --text "Mă ajuți să înțeleg exercițiul?"
roguard-review --language uk --text "Допоможеш розібратися із завданням?"
```

The explicit download fetches pinned, checksummed model files. The base is approximately 5.18 GB; subsequent inference is local. Keep disk space for the installed base and a verified private loading copy. For applications, load the model once and reuse it. [Installation, conversation input and Python API](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/V9_MODEL.md).

The first neural evaluation of the reserved 600 synthetic cases passed the unchanged 99% recall and 5% false-review limits in both languages. The installed package also reproduced all 5,022 prior scores exactly. Those prior cases had zero missed concerns and two Romanian false alerts; all six language/group checks passed. Development selection used 6,341 cases and had zero errors across 14 groups.

| Language | Reserved cases | Missed concerns | False reviews | Recall | False-review rate |
|---|---:|---:|---:|---:|---:|
| Romanian | 300 | 0/150 | 0/150 | 100.00% | 0.00% |
| Ukrainian | 300 | 0/150 | 0/150 | 100.00% | 0.00% |

Measurements describe project-authored synthetic inputs. They do not establish accuracy on real children's messages or measured human benefit. Scores are not calibrated safety probabilities. A negative prediction cannot close a concern; an overlong input requests clarification, and malformed model output is rejected. The binary review task is qualified separately from the auxiliary interpretation heads.

## Decision assurance

V9 preserves complete supplied interpretation hypotheses, computes an optimal adaptive clarification tree under declared reviewer costs, and verifies reports bound to their context, model identities and policy. Romanian and Ukrainian question labels and a local clarification workspace are included. The guarantee applies to the supplied finite problem; it does not establish that the true interpretation is among the supplied hypotheses. [Assurance guide](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/V9_ASSURANCE.md).

## Test an application

```sh
roguard-lab demo --language uk --family revocation --target stale
roguard-lab compare --language ro
roguard-lab suite --fail-on-failure
```

- Replay branching fictional scenarios against programmed references, recorded responses or an explicit Python adapter.
- Find the first failing step and reduce a scenario within a bounded replay budget.
- Preserve source restrictions through summaries, proposed memory and tool results.
- Minimize declared disclosure cost while covering required task fields.
- Compare permitted-task completion, unnecessary refusal and unauthorized proposals or deliveries.

Authenticated tenant APIs, encrypted durable permissions, one-attempt queued execution and metadata-only monitoring are available through optional integration extras. Host declarations and callback code remain trusted inputs. [Integration guide](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/RELIABILITY.md).

## Retained experimental backends

The older Studio conversation, context and evidence screens retain their documented backends; they do not automatically use the new Gemma 4 model. The `semantic` and `adapted` extras require a different Transformers major version and belong in a separate environment from `gemma4`. Their historical model limitations remain documented. [Context workflow](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/CONTEXT.md) · [Evidence workspace](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/EVIDENCE.md).

[GitHub and full README](https://github.com/xyz-rafael-xyz/roguard-child-safety) · [Training method](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/V9_SHARED_ATTENTION.md) · [Evaluation record](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/V9_STABILIZATION.md)
