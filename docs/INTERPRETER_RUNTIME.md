# Experimental local interpreter API

This source-checkout API runs an explicitly selected v9 structured adapter on
Apple Silicon with the `mlx` extra. It is under validation. It does not change
the stable installation or establish that the release accuracy requirements
have been met. The current interpreter-14 result is 371/400 on a reused
development screen; several language/stratum requirements still fail.

## What it returns

`InterpreterModel.predict` proposes the focal message's concern, subject,
source, assertion, relation and time. It derives a review proposal from those
six attributes. These are model interpretations, not confirmed disclosures.

- `predicted`: `0` for no signal, `1` for review, or `null` for abstention.
- `interpretation`: the six proposed labels, when decoding is available.
- `conditional_field_probabilities`: conditional scores, **not calibrated risk
  probabilities**. Earlier decoded fields influence later fields.
- `valid_label_tokens`: false if any full-vocabulary prediction was outside
  the allowed label tokens. Such a result remains an abstention.
- `model_identity` and `contract_digest`: identify the adapter and interpretation
  contract. Save both when comparing results across versions.

The report contains no message text. It never grants notification permission or
performs an external action. Applications must decide how to handle a review or
abstention; `no_signal` is not proof that a child is safe.

## Local setup and use

From this repository checkout, install the optional runtime:

Use Python 3.10 or newer; this runtime was exercised for packaging on Python
3.12. Keep it in a separate environment from the `semantic`/`adapted` extras:
the current MLX and Torch training stacks require different Transformers major
versions. The dependency-free core package retains its separate Python support.

```sh
python -m pip install -e '.[mlx]'
```

Obtain the pinned Gemma 3 4B MLX base separately under its upstream terms. The
exact repository, revision and file hashes are recorded in
[the base manifest](../research/v9-stabilization/gemma4-base-manifest.json).
The runtime makes no download requests. It copies only those verified files to
a temporary private snapshot before loading them; allow roughly 2.7 GB of
temporary disk space in addition to the model cache and available model memory.

For the completed pilot adapter:

```python
import json
from pathlib import Path
from roguard.interpreter import InterpreterModel

study = Path("research/v9-stabilization/interpreter-14")
verification = json.loads((study / "archive-verification.json").read_text())
model = InterpreterModel(
    study / "selected-adapter.json.gz",
    "/path/to/pinned/gemma-3-text-4b-it-4bit/snapshot",
    expected_archive_sha256=verification["archive_sha256"],
)
result = model.predict({
    "language": "ro",
    "turns": [{"role": "minor", "text": "Poți explica această temă?"}],
})
print(result)
```

Use `language="uk"` for Ukrainian. `focus=2` selects the third turn; later turns
are excluded from validation, the input digest and model inference. The selected
turn must have role `minor`. Inputs allow up to 12 selected turns, 2,000
characters per turn and 6,000 characters in total. More than 1,024 model tokens
produces an explicit context-limit abstention; the model never silently truncates
the conversation. Only the fixed policy/template prefix is shared in the KV
cache; conversation-specific cache state is not reused across requests.

The same API is available as `python -m roguard.interpreter --help`, accepting
one JSON conversation on standard input. A caller must supply the adapter path,
its expected SHA-256 and the pinned local base path.

## Verification status

Unit tests check prompt compatibility with the frozen training implementation,
future-turn isolation, malformed-output rejection, bounded archive handling and
private asset snapshots. They use test doubles and do not establish neural
accuracy. `installed-interpreter-30` is queued to replay the selected candidate's
400-case screen from an isolated installed wheel. It requires identical
interpretations, validity flags and decisions, with conditional probability
differences no larger than 0.001. That replay has not completed yet.

See [the stabilization record](V9_STABILIZATION.md) for training results,
qualification requirements and the limits of the synthetic evaluations.

## Turn interpretations into a clarification plan

The source checkout also provides a deterministic bridge to v9's review planner:

```python
from roguard.interpreter_assurance import plan_review, verify_review_plan

conversation = {
    "language": "ro",
    "turns": [{"role": "minor", "text": "Poți explica această temă?"}],
}
reports = [model.predict(conversation)]
expected_models = [model.identity]  # From the host's configured model.
plan = plan_review(conversation, reports, expected_model_identities=expected_models)
verify_review_plan(plan, conversation, reports, expected_model_identities=expected_models)
print(plan["certificate"]["status"])
```

For several configured models, supply one report and expected identity per model.
The planner retains complete joint interpretations; it does not create mixed
interpretations by combining fields from different models. If their proposed
actions differ, the certificate contains a clarification tree with Romanian or
Ukrainian questions. An invalid model output retains all 432 possible attribute
combinations. A caller can supply question costs through `question_costs`.

Reports must match the selected conversation prefix, focal turn, model identity
and interpreter contract. Missing or repeated models are rejected. Conditional
probabilities do not establish hypothesis coverage. Hosts must authenticate
inference services and reviewers themselves: hashes bind content but do not
prove who produced it. A determined action is conditional on the supplied
interpretations, not confirmation that they are true. Neither planning nor
answering a question grants notification or other external-action permission.

The bridge is also available as `python -m roguard.interpreter_assurance --help`.
Eleven focused tests pass, including binding verification, abstention handling,
correlation preservation and stale-report rejection. Testing against actual
installed neural outputs is queued after the interpreter replay; those results
are not yet available.
