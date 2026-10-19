# Local semantic model bundles

This experimental source-checkout interface runs one explicitly selected native
encoder with six semantic heads. It extracts subject, time, assertion, source,
relation and concern distributions, then evaluates their mass under the fixed
review policy. Scores are uncalibrated model outputs, not probabilities that a
child is in danger. No candidate described here has qualified as the default.

The loader checks the bundle manifest against a trusted SHA-256, confines asset
paths to its directory, and delegates model, tokenizer and policy identity
checks to `NativeFieldsModel`. Model weights are loaded lazily on the first
prediction. Nothing is downloaded implicitly. Neither loading nor prediction
contacts a guardian or authorizes an external action.

## Bundle layout

Keep a selected checkpoint, its head artifact and descriptor, the pinned base
manifest, and the tokenizer/configuration files together:

```text
bundle/
  bundle.json
  selected-heads.json.gz
  review.json.gz
  selected-model.safetensors
  base-manifest.json
  base/
    config.json
    ... pinned tokenizer files ...
```

`bundle.json` has this exact schema:

```json
{
  "format": "roguard-native-field-bundle-v1",
  "heads_sha256": "REPLACE_WITH_TRUSTED_SELECTED_HEADS_SHA256",
  "assets": {
    "heads": "selected-heads.json.gz",
    "artifact": "review.json.gz",
    "checkpoint": "selected-model.safetensors",
    "base": "base",
    "manifest": "base-manifest.json"
  }
}
```

The full selected checkpoint replaces the original base weight file; only the
pinned base tokenizer and configuration files are needed beside it. Preserve
the applicable upstream model licenses and notices when distributing assets.
These files are not included in the ordinary package installation.

## Run an explicit candidate

Use a source checkout with its optional native runtime dependencies installed:

```sh
python -m pip install -e '.[semantic]'
python -m roguard.native_field_bundle --bundle bundle/bundle.json --sha256 TRUSTED_BUNDLE_SHA256 --input conversation.json
```

The input contains `language` (`ro` or `uk`) and `turns`, each with `role` and
`text`. Use `--focus 0` to interpret the first turn using only its prefix; by
default the final turn is focal. Input beyond the model context limit produces
an abstention rather than silent truncation. The command emits the structured
JSON proposal. Standard input is supported when `--input` is omitted.

```python
from roguard.native_field_bundle import load_bundle

model = load_bundle("bundle/bundle.json", expected_sha256=trusted_bundle_digest)
proposal = model.predict(conversation)
proposals = model.predict_many(conversations, focuses=focal_turn_indices)
```

Keep the trusted digest in application configuration. Computing it from an
untrusted download and immediately accepting that same download does not
establish trust. A correct hash establishes identity; it does not establish
accuracy. See [stabilization results](V9_STABILIZATION.md) for the accuracy and
installed-runtime evidence attached to each candidate.

## Shared interface for binary, semantic and combined models

`python -m roguard.model_bundle` accepts the same options and dispatches checked
semantic or combined bundles to their respective loaders. For a direct binary
`NativeReviewModel`, use `format: "roguard-native-review-bundle-v1"`, replace
`heads_sha256` with the trusted descriptor's `artifact_sha256`, and omit `heads`
from `assets`. The same `load_bundle` Python API returns a model supporting
`predict` and `predict_many` in these cases. This chooses the runtime declared
by the pinned manifest; it does not choose a model or certify its accuracy.

The same command also accepts a native-classifier/interpreter combination:

```json
{
  "format": "roguard-review-fusion-bundle-v1",
  "fusion": {"path": "fusion.json", "sha256": "TRUSTED_FUSION_SHA256"},
  "components": {
    "native": {
      "artifact": "native/review.json.gz",
      "checkpoint": "native/selected-model.safetensors",
      "base": "native/base",
      "manifest": "native/base-manifest.json"
    },
    "interpreter": {
      "archive": "interpreter/selected-adapter.json.gz",
      "base": "interpreter/base"
    }
  }
}
```

This combination requires Apple Silicon for the MLX interpreter and both the
`semantic` and `mlx` optional dependencies. The pinned interpreter base files
must already be present. Loading remains lazy, with no implicit download.
`ReviewFusionPipeline` runs the native classifier and interpreter and passes
their input-bound reports to the selected fusion model. It validates every
conversation before inference, preserves focal turn indices, and rejects
incomplete component batches. This interface does not qualify the combined model
or change the installed default.

## Fixed categorical policy comparison

The source checkout also provides `DiscreteFieldsModel` for an explicitly
selected native semantic model. It applies the declared policy directly to the
highest-scoring label in each field, instead of using a probability-mass cutoff:

```python
from roguard.discrete_fields import DiscreteFieldsModel
from roguard.native_field_bundle import load_bundle

model = DiscreteFieldsModel(load_bundle("bundle/bundle.json", expected_sha256=trusted_bundle_digest))
proposal = model.predict(conversation)
```

This is a separate experimental decision path with a distinct model identity.
It preserves context-limit abstentions and validates the source report's input,
model, policy and field bindings. It does not authenticate the report producer.
Study69 compares both languages against the unchanged accuracy gates and checks
the policy on actual installed neural outputs; it is not an approved default.

## Experimental three-model report combination

`roguard.three_expert.ThreeExpertReview` combines reports from the direct native
classifier, native semantic heads, and interpreter. Load a JSON tree artifact
with its independently obtained SHA-256, then call
`predict(payload, native_report, semantic_report, interpreter_report, focus=...)`.
All reports must describe the same input and pinned model identities. Any
component abstention produces a clarification request. The loader rejects
malformed, cyclic, shared, overdeep, or oversized tree structures.

This API combines reports; it does not load the neural models or authenticate
reports supplied by an untrusted caller. The output never authorizes notification
or an external action. It is not the default detector. Study80 verified the
Romanian study79 artifact against 200 cached outputs from actual installed neural
runtimes: zero changed decisions and zero score difference. Ukrainian qualification
and reserved evaluation remain incomplete; this is not a stable v9 release.

### Running the three models together

The current source adds `ThreeExpertPipeline` and the unified loader format
`roguard-three-expert-bundle-v1`. It accepts the same conversation and focus as
other model bundles and runs the direct native model, native semantic model, and
interpreter before combining their input-bound reports. A batch is fully checked
before inference; missing component results cannot silently drop cases.

The combined pipeline uses the `fusion` extra on Apple Silicon with Python 3.10 or later. The older
`semantic` extra pins Transformers 4, whereas the MLX interpreter requires
Transformers 5; install `fusion` in a separate environment, without also requesting
`semantic` or `adapted`. This combined environment is undergoing actual-model
verification in study84. The plain package remains available without these heavy
optional dependencies. No model is downloaded by loading a bundle.

Its manifest contains `format`, a `fusion` object (`path`, `sha256`), and
`components`:

- `native`: `artifact`, `checkpoint`, `base`, `manifest`.
- `semantic`: `heads`, `artifact`, `checkpoint`, `base`, `manifest`.
- `interpreter`: `archive`, `base`.

All paths are relative to and confined within the manifest directory. Obtain its
SHA-256 through trusted configuration, then use `roguard-model --bundle ...
--sha256 ... --input ...` or `roguard.model_bundle.load_bundle(...)`. The models
remain experimental until their complete qualification checks pass.
