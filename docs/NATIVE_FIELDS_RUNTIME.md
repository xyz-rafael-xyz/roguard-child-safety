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
