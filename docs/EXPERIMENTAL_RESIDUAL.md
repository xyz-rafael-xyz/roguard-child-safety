# Optional role residual runtime

This is an **unqualified experiment**, not a v10 release. The package remains
9.0.0. No residual weights are supplied, downloaded, or selected by a default
model profile. Existing v9 model loading and predictions remain unchanged.

The explicit Python entry point is `roguard.load_experimental_role_residual`.
Disabled mode delegates to a configured public parent model and returns its
original result objects without reading residual assets or importing numerical
extras:

```python
from roguard import load_experimental_role_residual

model = load_experimental_role_residual(enabled=False, parent=existing_model)
result = model.predict(payload, focus=None)
results = model.predict_many(payloads, focuses=None)
```

Enabled mode requires the existing `gemma4` extra, Apple Silicon and the pinned
runtime. It accepts only local, explicitly hash-bound research assets:

```python
model = load_experimental_role_residual(
    enabled=True,
    base=base_snapshot,
    original_v9_adapter=v9_adapter,
    original_v9_spec=v9_spec,
    parent75_weights=parent_weights,
    parent75_spec=parent_spec,
    residual_manifest=residual_manifest,
    residual_tensors=residual_safetensors,
    expected_residual_manifest_sha256=manifest_sha256,
)
```

One backbone serves both routes. Inputs of at most **613 classifier tokens** use
the frozen research parent75 head plus the aligned role residual. Inputs from
614 through 1024 tokens use the **original v9 head, policy mode, threshold,
model identity and result format**. Above 1024, the original v9 context-limit
clarification is retained. There is no truncation and no fallback on errors or
low confidence. `model.profile` exposes these routes and the unqualified status.
Token counts include the fixed classifier prompt; they are not word counts.

The residual reads the qualified final vector and three mean pools from the
existing global VALUE23 state: focal minor, prior minor and prior nonminor.
Roles come from validated message structure. Instruction text, delimiters,
special tokens and tokens straddling content boundaries are excluded from pools
without changing classifier token IDs. These features are not evidence that
the model understands speaker attribution. No second backbone forward is added.

The float32 tensor inventory is exactly:

| Tensor | Shape |
| --- | --- |
| `first.weight` | 32 × 5638 |
| `first.bias` | 32 |
| `last.weight` | 17 × 32 |
| `last.bias` | 17 |

There are 181,009 parameters. Each of the four feature blocks is independently
L2-normalized in float64 with a minimum norm of `1e-6`, scaled by the square root
of its width, and converted to float32. Six shared count/presence values follow
these blocks. The 32-unit layer uses exact-erf GELU. The residual changes only
the first 17 semantic logits; the last two legacy logits retain their bytes.
Enabled zero-initialized residuals use normal addition; only explicit disabled
mode is a bypass.

The manifest format is `roguard-role-residual-experiment-1`, with `qualified`
strictly `false`. It binds the exact tensor file, packaged runtime and alignment
source hashes, reviewed alignment lineage, parent/v9 specs and adapters, base
revision, selected checkpoint, training plan, feature manifest and selection
proof. The normalization and fallback dictionaries are fixed by the runtime.
The safetensors reader is limited to 1 MiB, manifests to 128 KiB; the existing
16 MB parent adapter limit is unchanged. Training comparators are not eligible
for automatic export or promotion.

Synthetic numerical and routing tests do not qualify a candidate. A selected
candidate still needs native numerical/decision parity, all retained full-data
quality and regression gates, and the existing independent-review requirements.
No accuracy, v9 superiority, calibration or production speed claim follows from
this integration.
