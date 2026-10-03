# RoGuard Local — Romanian and Ukrainian child-safety research toolkit

**Latest tagged release: v0.2.0. Main branch package version: 1.0.0.dev0. V1 evidence gate: open.** RoGuard Local is a local checker for caller-declared child-safety contracts and a research harness for Romanian and Ukrainian abstract-card evaluations. It does not read a child message to decide whom to notify, and it takes no external action. Its trained text models are experimental and have not been validated on authentic child language.

This project is independent of Roblox. [Roblox Guard 1.0](https://about.roblox.com/newsroom/2025/07/roblox-guard-advancing-safety-for-llms-with-robust-guardrails) is a separate LLM moderation toolkit; there is no affiliation, shared model, or compatibility claim. The package here is named `roguard-local`.

## What the checker does

Each check compares **facts supplied by the caller** with a declared rule. The tool marks those facts as unverified; it does not establish whether a person's account is true, who holds legal authority, or whether a proposed response is helpful in a real case.

| Code | Check | Python function | What the result means |
|---|---|---|---|
| D1 | Declared disclosure and support-signal fields | `check_disclosure_evidence` | Checks the caller's field declarations; does not interpret a child's words. |
| R1 | Recipient and purpose scope | `check_routing` | Checks whether the proposed recipient fits an explicit rule. A relationship alone grants no access. |
| A1 | Declared or locally counted words against a caller-supplied cap | `check_readability_contract` | Checks length only. It does not measure readability or comprehension. |
| P1 | Permission state and repeated boundary pressure | `check_permission`, `check_boundary` | Checks declared permission events and boundary facts. `check_boundary` is part of P1. |
| G1 | Human-review gate before an external action | `check_gate` | Flags an automatic external branch or a missing review owner in the declared flow. |
| S1 | Declared support-response fields | `check_support_contract` | Checks fields marked applicable by the caller; does not judge a real reply. |

The CLI can count A1 words locally when text is supplied; the cap still comes from the caller. The [Romanian taxonomy](taxonomy/taxonomy_ro.md) and [Ukrainian draft taxonomy](taxonomy/taxonomy_uk.md) give the six definitions and their limits. Romanian project approval is recorded, but neither language has a completed independent review of the exact taxonomy bytes. The [v1 evidence command](docs/V1_EVIDENCE.md) reports this gate. Some historical bibliography links inside the frozen taxonomy point to a private sibling workspace; the public [research note](docs/TAXONOMY_RESEARCH.md) summarizes the accessible sources. Changing taxonomy bytes would require a new review digest.

A **synthetic abstract card** is an invented description of annotation fields or policy facts, with no real or realistic child statement. A **complete one-fact pair** is a positive card and its minimally changed negative, both classified correctly. Those pairs measure an invented-card task, not accuracy on children's messages. When a professional has reporting duties, this software does not determine or replace them; follow applicable law, organizational policy, and qualified human judgment.

## Install and try it

The public [GitHub releases](https://github.com/xyz-rafael-xyz/roguard-child-safety/releases/latest) provide source archives and a wheel. Python 3.9 or newer is supported. On macOS or Linux:

```sh
git clone https://github.com/xyz-rafael-xyz/roguard-child-safety.git
cd roguard-child-safety
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m roguard examples/contracts_ro.json
.venv/bin/python -m roguard examples/contracts_uk.json
```

On Windows, create the environment with `py -3.12 -m venv .venv` and use `.venv\Scripts\python.exe` in the last three commands. The examples are fictional, caller-declared contracts; no model download is needed. For your own declared contract, copy an [example](examples/contracts_ro.json), edit its fields, and run the same command. The [JSON Schema](schema/contract-input.schema.json) and [CLI guide](docs/CLI.md) describe every field and status.

`roguard-studio --open` opens a loopback-only [Decision Studio](docs/DECISION_STUDIO.md) for checking contracts and exploring one-fact changes. The command-line `roguard-explore examples/proposed_use_ro.json` shows which declared changes alter a result. Both are local consistency tools, not model accuracy tests. `python -m roguard --jsonl examples/contracts_mixed.jsonl` checks multiple contracts after validating every line.

For a small routing check in Python:

```python
from roguard import RoutingCard, check_routing

card = RoutingCard(
    principal="minor",
    item="element-simbolic",
    purpose="A",
    proposed_recipient="părinte",
    recipient_roles=frozenset({"minor", "părinte"}),
    allowed_by_scope={
        ("minor", "element-simbolic", "A"): frozenset({"minor"})
    },
)
print(check_routing(card).reason_codes)  # ('PRINCIPAL_SCOPE_CONFLICT',)
```

`părinte` here is an example role label, not a finding about legal authority or safe disclosure. A parent or guardian can be the subject of a concern; the checker must not infer permission to notify them from that relationship. The responsible human must assess the actual policy and situation. The [combined-use guide](docs/CLI.md#check-one-proposed-use-across-contracts) shows how routing, current permission, and a separate human gate fit together without sending anything.

## What the research shows

**The mixed-input result is not six-category text understanding.** Four categories in the 72/72 workflow received typed facts; D1 and S1 used a frozen model on abstract-card text. The same text-only model scored 55/72 and missed every A1 positive. A separate frozen Romanian D1/S1 model scored 47/48 on one abstract wording surface and 23/48 on another. This sensitivity is why no trained adapter is offered for live screening.

| Study on synthetic abstract cards | Complete pairs | Descriptive 95% Wilson interval | Interpretation |
|---|---:|---:|---|
| Romanian mixed input, with four typed-fact categories | 72/72 | 94.9–100% | Passed its narrow registered target; text-only comparator was 55/72 (65.4–84.7%). |
| Romanian D1/S1, frozen model, batch 0025 | 47/48 | 89.1–99.6% | Passed its narrow target on one wording surface. |
| Same frozen model, batch 0027 | 23/48 | 34.5–61.7% | Failed on new abstract prose; S1 recall was 8/24. |
| Bilingual V25 sealed synthetic test | 165/192 | 80.3–90.2% | Failed its fixed per-cell target; Ukrainian D1 was 36/48 and S1 was 38/48. |
| Private V30 fresh synthetic test | 129/192 | 60.3–73.4% | Failed; Ukrainian D1 was 2/48. |
| Private V31 fresh synthetic test | 143/192 | 67.9–80.1% | Failed; Romanian D1 was 13/48. |

All model results in this table are **author-reported**. Selected adapter weights and their Git history are in a separate private research archive, so public readers cannot replay weight-dependent results from this checkout. The [benchmark ledger](BENCHMARK.md) retains the study-by-study results, including failures and development-only diagnostics; the [public-release note](docs/PUBLIC_RELEASE.md) explains what CI can reproduce. Wilson intervals are descriptive for each count and may be optimistic when paired cards share wording or authoring patterns. No row estimates performance on authentic Romanian or Ukrainian child language.

The research adapters are not included in the public wheel. Publishing weights is a separate decision involving base-model licences, privacy review, and the project's release rules. The local checker remains usable without them.

## Specialist feedback and v1 status

A review memo supplied by the project owner attributes README-focused feedback to a **Romanian-language child-protection practitioner**, a **Ukrainian linguist specializing in plain language and readability**, and a **child online-safety policy and evaluation specialist**. Their identities are withheld at their request. The memo reports conditional or negative judgments on D1 and S1, limits A1 to word counting, and calls for clearer evidence and naming claims. It says the taxonomy text was **not reviewed**. [Feedback disposition](docs/REVIEW_FEEDBACK_V2.md) records which edits were applied and what still needs a separate decision. This memo is not independent taxonomy signoff or model validation.

`roguard-v1-evidence --root .` currently reports both taxonomy-review records pending and four independent abstract-study cells blocked. A v1 model claim requires exact-version Romanian and Ukrainian language and child-protection review, followed by newly authored, blinded, independently annotated and adjudicated D1/S1 cards for both languages. The private archive has four pre-author V25 comparator freezes; the [bilingual prediction and replay guide](docs/BILINGUAL_BASELINE_PREDICTION.md) describes how the study owner can score an opaque packet and reproduce every decision. These steps measure only the abstract-card task. The [reviewer handoff](docs/REVIEWER_HANDOFF.md), [human validation protocol](docs/HUMAN_VALIDATION_PROTOCOL.md), and [v1 evidence ledger](docs/V1_EVIDENCE.md) state the evidence and claim boundaries.

## Pe scurt, în română

RoGuard Local verifică reguli și câmpuri **declarate de apelant**; nu citește singur relatarea unui copil și nu trimite notificări. A1 compară numărul de cuvinte cu un plafon ales de apelant, fără a măsura ușurința de înțelegere. Taxonomia română are aprobarea autorului, dar nu o revizuire independentă a textului exact; validarea automată a loturilor sintetice nu ține loc de evaluare de specialitate. Modelele au rezultate instabile pe fișe abstracte inventate și nu sunt validate pentru mesaje reale. Consultați [taxonomia](taxonomy/taxonomy_ro.md), [rezultatele complete](BENCHMARK.md) și [stadiul v1](docs/V1_EVIDENCE.md).

## Коротко українською

[Українська таксономія](taxonomy/taxonomy_uk.md) залишається проєктом: незалежні фахівці ще не перевірили її точний текст. Локальна програма перевіряє правила й поля, які задає користувач, але не є перевіреним детектором дитячих повідомлень. Публічний виклик `screen()` для експериментальної текстової моделі працює лише з румунською; приватний двомовний V25 є дослідницьким компаратором і не пройшов свій тест на вигаданих абстрактних картках. Жодна функція не надсилає повідомлень і не виконує зовнішніх дій; [результати](BENCHMARK.md) та [умови для v1](docs/V1_EVIDENCE.md) наведено окремо.
