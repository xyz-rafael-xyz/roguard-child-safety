# V9 local Romanian and Ukrainian review model

This optional model proposes whether a focal message needs review under the
project's declared synthetic review policy. It uses earlier conversation turns
to distinguish a current concern, an ordinary topic change, and a revision of an
earlier statement. A review proposal is not a finding that a child is unsafe.

Available in RoGuard **9.0.0**, through the default `v9` model profile.

## Install and download

The optional Gemma 4 model uses native Apple Silicon and Python 3.12 or later
(validated on Python 3.12).
The core toolkit remains available on Python 3.9 or later, including Linux and
Windows; this particular optional model has a narrower platform requirement.
Use a separate virtual environment for this extra: the older `semantic` and
`adapted` extras require a different Transformers major version.

```sh
pip install --upgrade "roguard-child-safety[gemma4]"
roguard-fetch fetch
```

The explicit fetch checks each file's size and SHA-256 against the profile
packaged with RoGuard. It downloads the unchanged base from a pinned upstream
revision and the RoGuard adapters from the project's release assets. Completed
verified files are reused if fetching is interrupted. The base is approximately
5.18 GB. Inference makes a verified private copy before loading, so allow space
for both the installed base and its temporary copy, plus package dependencies.

## Review fictional examples

```sh
roguard-review --language ro --text "Mă ajuți să înțeleg exercițiul?"
roguard-review --language uk --text "Допоможеш розібратися із завданням?"
```

For a conversation, pass a JSON file with `language` and `turns`, or supply the
same JSON on standard input. Each turn has `role` and `text`. Use `--focus 2` to
review the third turn. Later turns are excluded from the selected input.

Save this fictional conversation as `conversation.json`:

```json
{
  "language": "uk",
  "turns": [
    {"role": "minor", "text": "Сьогодні я малюю листівку."},
    {"role": "assistant", "text": "Що ти хочеш намалювати?"},
    {"role": "minor", "text": "Допоможеш намалювати сонце?"}
  ]
}
```

```sh
roguard-review conversation.json --focus 2
```

The output is a review proposal. A positive result does not verify an event or
authorize contact with anyone. A negative result does not resolve an earlier
concern. Scores are not calibrated probabilities of safety. The limit is 1,024
tokens for the full serialized model input, including its instructions. Longer
inputs produce `clarify` instead of being silently truncated; malformed or nonfinite
model outputs are rejected with an error. Neither outcome may be treated as a
negative decision by an application.

## Reuse a model in an application

Load once and reuse the object to avoid repeating model startup for each message:

```python
from roguard.entailment_bundle import load_bundle
from roguard.model_install import cache_directory, install, profile

raw, manifest_sha = profile("v9")
bundle, bundle_sha = install(raw, manifest_sha, cache_directory(), download=False)
model = load_bundle(bundle, expected_sha256=bundle_sha)

result = model.predict({
    "language": "ro",
    "turns": [{"role": "minor", "text": "Mă ajuți să înțeleg exercițiul?"}],
})
print(result)
```

Fetching is explicit; this inference example uses the existing local cache.
The response binds the proposal to the selected input and model identity and
omits the input text. Applications remain responsible for their own access,
retention, human-review and permission controls. Romanian and Ukrainian use
separate adapters; the language dispatcher retains one active model at a time.
For a queue, group messages by language and use `model.predict_many(payloads,
focuses=focus_indices)` to reuse model startup and the shared instruction prefix.
Each payload must include the conversation history relevant to that request;
the model does not remember earlier requests.

The model's registered accuracy task is the binary review proposal. Its auxiliary
interpretation heads do not establish verified facts or separately qualified
accuracy for each interpretation field. The decision-assurance planner is a
separate API for supplied interpretation hypotheses.

The older `roguard-studio --context` and `--evidence` screens retain their
documented backends. They do not automatically switch to this Gemma 4 model.
Use `roguard-review` or the Python API above for the qualified v9 model profile.

## Qualification record

The first neural evaluation of the reserved 600 synthetic cases passed the unchanged 99% recall and 5% false-review limits in both languages. The installed package also reproduced all 5,022 prior scores exactly. Those prior cases had zero missed concerns and two Romanian false alerts; all six language/group checks passed. Development selection used 6,341 cases and had zero errors across 14 groups.

| Language | Reserved cases | Missed concerns | False reviews | Recall | False-review rate |
|---|---:|---:|---:|---:|---:|
| Romanian | 300 | 0/150 | 0/150 | 100.00% | 0.00% |
| Ukrainian | 300 | 0/150 | 0/150 | 100.00% | 0.00% |

The reserved test was opened only after development, prior and installed-runtime gates passed. Abstentions count as errors. These are project-authored synthetic combinations, not independent observations or a real-child deployment study. The analyst was not blind to the test design. Repeated prior results are regression evidence. No measured real-world benefit or flawless performance is claimed.

[Selected checkpoints](../research/v9-stabilization/reported-tail-fit-196/actual/report.json) · [Prior regression](../research/v9-stabilization/reported-tail-prior-197/actual/report.json) · [Installed runtime](../research/v9-stabilization/reported-tail-installed-198/actual/report.json) · [First reserved evaluation](../research/v9-stabilization/reported-tail-reserved-199/actual/report.json) · [Release equivalence](../research/v9-stabilization/stable-release-200/wheel-equivalence.json).

The immutable profile is `v9-gemma4-20261019`; `v9` selects the same manifest. Its SHA-256 is `90c9cbacf83a78070565dccab1a8de4fb64015b4f2ed210e1425ec3e7ecad2bb`. The release wheel preserves all 156 qualified package files byte for byte, adding only the download manifest and profile catalog. [Training method and numerical checks](V9_SHARED_ATTENTION.md).

## Recompute the published benchmark

After installing the released model, run this from a checkout of the matching
release tag. It performs local inference again and evaluates all cases, including
any abstentions; it does not train or change a cutoff. The input statements are
synthetic. Replaying an already published benchmark is a regression check.

```python
import json
from pathlib import Path
from roguard.entailment_bundle import load_bundle
from roguard.model_install import cache_directory, install, profile
from roguard.review_evaluation import evaluate

raw, manifest_sha = profile("v9")
bundle, bundle_sha = install(raw, manifest_sha, cache_directory(), download=False)
model = load_bundle(bundle, expected_sha256=bundle_sha)
expected, predicted = [], []
for language in ("ro", "uk"):
    path = Path("research/v9-stabilization/paired-scope-117/data") / f"{language}-test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        results = model.predict_many(
            [{"language": language, "turns": row["turns"]} for row in batch],
            focuses=[row["focus"] for row in batch],
        )
        if len(results) != len(batch):
            raise ValueError("Incomplete model response")
        for row, result in zip(batch, results):
            expected.append({"case_id": row["case_id"], "language": language, "gold": row["review"]})
            predicted.append({"case_id": row["case_id"], **result})
print(json.dumps(evaluate(expected, predicted), indent=2))
```
