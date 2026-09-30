# Reproduce the local checks and research runs

This public source checkout omits trained adapter weights and their original Git history. The structured CLI, synthetic records, and model-free checks run here. Weight replays and historical commit-order checks described below require the private research archive; [the public-release note](PUBLIC_RELEASE.md) lists the boundary.

The public Git repository contains code, taxonomy, attested abstract batches, frozen prediction records, and metadata for selected research adapters. The adapter weights remain in a private research archive. Downloaded base weights and other training intermediates stay under ignored `checkpoints/`. MLX studies require a local Apple Silicon machine; the read-only contract CLI and model-free tests do not.

## Lightweight path

```sh
python3.12 -m venv .venv312
. .venv312/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -q
python -m roguard --jsonl examples/contracts_mixed.jsonl
roguard-contrast examples/contrast_route_before.json examples/contrast_route_after.json
```

Use the [input schema](../schema/contract-input.schema.json) for editor completion and the [report schema](../schema/contract-report.schema.json) for integrations. The CLI checks all JSONL lines before emitting output. Its symbolic facts are caller supplied; it does not interpret a child message.

## Data provenance

The accepted training and held-out batches are named explicitly. For a registered generated batch, the following command replays its generator, exact-byte attestation, label balance, and prior-surface audit:

```sh
python training/validate_generated.py --batch batch-0018
```

Batch 0014 is training, 0015 is development, and 0016–0018 are separate consumed test splits. A test batch is rejected by `load_train_dev()`. Do not use a test split to choose a prompt, threshold, checkpoint, or model family.

## Local RoMistral v6 study

The [README](../README.md#quickstart-english) gives the pinned RoMistral download and 4-bit conversion commands. Its upstream model card lists a noncommercial license. With `checkpoints/romistral-4bit` prepared, the frozen v6 commands are:

```sh
python -m pip install -e '.[mlx]'
python training/train_mlx.py --batch batch-0014 --batch batch-0015 --model checkpoints/romistral-4bit --base-revision fc097a5fdc2a5a689df084c76ffacf2cfb9f6550 --prompt-version v4 --iters 1200 --output checkpoints/ro-v6
python eval/select_v6_dev.py --output eval/runs/ro-v6-dev-selection.json
```

Its selected weight hash was committed before batch 0016 inference. The [frozen benchmark](../BENCHMARK.md#frozen-romanian-compositional-comparison-batch-0016) reports the test and all comparators. A fresh reproduction should write to new local paths and compare hashes; the scripts protect existing outputs from overwrite.

## Local Qwen3 studies

Install the MLX extra in the same virtual environment, then run one fit at a time. On a new machine, add `--download` to fetch the pinned 4-bit snapshot. Both trainers verify the preregistered model file hashes before fitting.

```sh
python -m pip install -e '.[mlx]'
python training/train_qwen_transfer.py --output checkpoints/ro-qwen-v7 --download
python eval/select_qwen_dev.py --output eval/runs/ro-qwen-v7-dev-selection.json
```

The v7 fit uses the original v4 binary exporter. Its [frozen test](../BENCHMARK.md#frozen-romanian-qwen-transfer-comparison-batch-0017) exposed an output-format failure. The [prospectively amended v8 protocol](V8_PROTOCOL.md) changes training-task exposure by giving each category 144 `da` and 144 `nu` tasks, and accepts only Qwen's empty reasoning wrapper around a single decision. Its development export is byte-identical; its [fresh test](../BENCHMARK.md#frozen-romanian-balanced-qwen-comparison-batch-0018) failed the accuracy target despite 144/144 parsed cards.

```sh
python training/train_qwen_balanced.py --output checkpoints/ro-qwen-v8 --download
python eval/select_qwen_v7_wrapper_dev.py --output eval/runs/ro-qwen-v7-wrapper-dev-selection.json
python eval/select_qwen_balanced_dev.py --output eval/runs/ro-qwen-v8-dev-selection.json
```

Each selector evaluates its fixed checkpoints on batch 0015. **Commit the selected-step record and weight hash before test inference.** The evaluation commands reject any attempt to pass a test batch to training, and comparison scripts check hashes, selected metadata, prompt, model revision, predictions, and row order. They do not establish authenticity of child-language performance.

After each selection is committed, run its fresh test exactly once:

```sh
python eval/evaluate_mlx.py --batch batch-0017 --adapter checkpoints/ro-qwen-v7/selected-adapter --output eval/runs/ro-qwen-v7-test-0017.json
python eval/evaluate_mlx.py --batch batch-0018 --adapter checkpoints/ro-qwen-v8/selected-adapter --qwen-empty-think-wrapper --output eval/runs/ro-qwen-v8-test-0018.json
python eval/evaluate_mlx.py --batch batch-0018 --adapter checkpoints/ro-qwen-v7-wrapper/selected-adapter --qwen-empty-think-wrapper --output eval/runs/ro-qwen-v7-wrapper-test-0018.json
python eval/run_binary_panel.py --batch batch-0018 --model qwen3_4b --qwen-empty-think-wrapper --output eval/runs/qwen3-base-v4-0018.json
```

Run the corresponding pinned baselines on those same batches, then `eval/compare_qwen_v7.py` or `eval/compare_qwen_balanced_v8.py`. The [protocols](V7_PROTOCOL.md) and [benchmark](../BENCHMARK.md) list the exact comparators and success rules. Run files under `eval/runs/` are ignored until explicitly added to Git; the scripts refuse to overwrite an existing result. The [model card](MODEL_CARD.md) explains the intended research domain and deployment limits.

## Private selected MLX research artifacts

The private repo also preserves the final-step [v1](../models/ro-v1-abstract/README.md), [v2](../models/ro-v2-abstract/README.md), and [v3](../models/ro-v3-abstract/README.md) RoMistral pilots. Their portable configs omit local training paths. The v2 and v3 weight hashes match their frozen comparison records; v1 has no pretest weight-hash record, so its manifest binds the local final-step checkpoint and saved run, and a local base-model replay checks a saved label. All three pilots failed to establish reliable six-category decisions.

```sh
python eval/verify_committed_pilot_adapters.py
python eval/verify_committed_pilot_adapters.py --base-model-path checkpoints/romistral-4bit
```

The first command needs no model dependency. The second loads the pinned local RoMistral base and each selected pilot adapter, then reproduces one saved abstract-card label. `training/package_pilot_mlx.py` requires the historical local final-step file and refuses to overwrite a packaged artifact.

The private repo preserves the selected [RoMistral v4](../models/ro-v4-abstract/README.md), [RoMistral v6](../models/ro-v6-abstract/README.md), [Qwen v7](../models/ro-qwen-v7-abstract/README.md), and [Qwen v8](../models/ro-qwen-v8-abstract/README.md) LoRA weight bytes. Their compact adapter configs contain only the fields MLX needs at inference; the metadata no longer depends on a training-machine path. They **all failed** their registered symbolic targets. The v7 wrapper-parser comparison reuses the same v7 weights, so there is no duplicate adapter.

```sh
python eval/verify_committed_mlx_adapters.py
python eval/verify_committed_mlx_adapters.py --romistral-base checkpoints/romistral-4bit --qwen-base /path/to/pinned/qwen-4bit-snapshot
```

The first command checks selection and artifact hashes without MLX. The second, on Apple Silicon with `.[mlx]` installed and both pinned local bases present, loads each selected adapter and reproduces one saved symbolic-card label. `MLXClassifier` accepts `base_model_path=` for these portable artifacts and refuses to load one without a local base path. The [packaging script](../training/package_selected_mlx.py) checks a historical local weight against its development selection before creating a new artifact; it never overwrites an existing one. This is research reproducibility, not a live child-message screening procedure.

## Romanian mmBERT encoder studies

The [v9 full-weight attempt](V9_PROTOCOL.md) was interrupted under local memory pressure before test inference. The [v9a amendment](V9A_AMENDMENT.md) fits a small pairwise LoRA adapter on the same approved batch 0014, selects one shared threshold on batch 0015, and reports its failed frozen result on batch 0019. The [v10 protocol](V10_PROTOCOL.md) adds cross-category negative tasks for A1 and S1 and uses a new sealed batch 0020. The base is [JHU mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) at one pinned revision; the fetch script checks SHA-256 hashes for its study files. An MLX installation is not needed for the encoder fit.

```sh
python -m pip install -e '.[train]'
MMBERT_PATH=$(python training/fetch_mmbert.py)
python training/train_mmbert_pairs.py --base-model-path "$MMBERT_PATH" --output checkpoints/reproduction-v9a
python training/train_mmbert_cross.py --base-model-path "$MMBERT_PATH" --output checkpoints/reproduction-v10
```

The trainers reject existing output directories, use only attested train/development batches, and save each epoch's adapter and full development scores. Reproduction outputs must go to new local paths. The committed [v9a development choice](../eval/runs/ro-mmbert-v9a-dev-selection.json) and frozen [comparison](../eval/runs/ro-mmbert-v9a-test-0019-comparison.json) bind the original fit and test hashes. For a new study, **commit the selected development record and adapter hash before test inference**. `eval/run_mmbert_v9a.py` and `eval/run_mmbert_v10.py` enforce that order, and their outputs are not overwritten. On macOS the trainer uses MPS when available; otherwise it uses CUDA or CPU. The 0–1 scores and selected cutoffs are research diagnostics, not calibrated child-risk probabilities.

The later [v11 surface study](V11_PROTOCOL.md) and [v12 category-threshold study](V12_PROTOCOL.md) keep their failed tests visible. V12 reuses unchanged v11 epoch-3 weights; its [committed development scores](../eval/runs/ro-mmbert-v11-epoch-3-dev.json) and [selection](../eval/runs/ro-mmbert-v12-dev-selection.json) reproduce the six cutoffs without loading a model. Run `python eval/verify_mmbert_v12.py` to check the frozen batch-0024 decisions and metrics from saved scores. To repeat inference on another machine, fetch the pinned mmBERT base, reproduce the v11 adapter into a new output directory, verify its selected weight hash, and use `eval/run_mmbert_v12.py` with a new output path. Batch 0024 is consumed; repeated inference is a reproducibility check, not a new held-out experiment.

The private repository now includes the exact [selected v11 research adapter](../models/ro-mmbert-v11-abstract/README.md), with a portable PEFT config. The base weights are fetched separately. To inspect a neutral symbolic card locally after installing `.[train]`:

```sh
MMBERT_PATH=$(python training/fetch_mmbert.py)
python eval/check_committed_adapter.py --base-model-path "$MMBERT_PATH"
```

The check loads the committed adapter and matches one saved score and label on a consumed abstract card. It prints no card text. To call the Python API directly:

The selected [v9a](../models/ro-mmbert-v9a-abstract/README.md) and [v10](../models/ro-mmbert-v10-abstract/README.md) historical comparators are also in private Git with unchanged selected weight bytes and portable configs. Both **failed** their registered synthetic targets. Verify their selections and optionally load them against the same pinned base:

```sh
python eval/verify_committed_historical_adapters.py
python eval/verify_committed_historical_adapters.py --base-model-path "$MMBERT_PATH"
```

Their packaging script, `training/package_selected_mmbert.py`, checks selected hashes before copying a historical local checkpoint to a new artifact directory. It refuses to overwrite an existing artifact. The selected records and raw failure measurements remain unchanged.

To call the v11 Python API directly:

```python
from roguard import MMBertResearchClassifier, screen

base_path = "<path printed by python training/fetch_mmbert.py>"
backend = MMBertResearchClassifier(base_path, "models/ro-mmbert-v11-abstract")
result = screen("Fișă simbolică: calitatea sursei: adult.", language="ro",
                source_kind="message", backend=backend, thresholds=backend.thresholds)
print(result.labels)
```

This snippet checks loading and inference, not accuracy. The adapter's pinned cutoff is the **failed** v11 selection; its scores are uncalibrated and it should not screen live child messages.

Run `python eval/verify_mmbert_v13.py` to recompute the later frozen D1/S1 advisory result from saved scores without downloading the base. Batch 0025 is consumed; this verifies the recorded result and must not be used to choose another model or threshold. CI also recomputes V12, V13, V14, and the negative-only probe. A passing verifier for V12 confirms the recorded **failed experiment**; it does not change that outcome.

The [v14 protocol](V14_PROTOCOL.md) combines unchanged v11 model decisions for D1/S1 with caller-declared contract checks for R1/A1/P1/G1. The separate batch-0026 sidecar gives those four categories exact synthetic facts, so this is a different input condition from the text-only comparator. Its [frozen result](../BENCHMARK.md#frozen-romanian-mixed-input-workflow-batch-0026) can be recomputed without downloading a model:

```sh
python training/validate_generated.py --batch batch-0026
python eval/verify_mmbert_v14.py
```

The verifier checks saved score thresholds, generated sidecars, checker decisions, all reported metrics, and hashes for the committed test, protocol, and adapter. Batch 0026 has already been consumed. Repeating inference with `eval/run_mmbert_v14.py` into a different path is a reproducibility check, not another independent test.

The separate [neutral-request probe](NEUTRAL_PROBE_PROTOCOL.md) checks one out-of-domain failure mode: false D1 reviews on 48 authored, harmless Romanian requests. It uses the same frozen adapter and cutoff; the negative-only result cannot establish recall. Verify its saved scores and group counts without loading the model:

```sh
python eval/verify_mmbert_neutral.py
```

The runner `eval/run_mmbert_neutral.py` refuses uncommitted inputs and existing outputs. Its original texts are consumed as a test, not available for new cutoff selection.

The later [V15 advisory prose protocol](V15_PROTOCOL.md) froze the same v11 adapter and cutoff on a new D1/S1 rendering. It **failed** at 23/48 exact pairs. Recompute the saved decisions and metrics without loading the model:

```sh
python eval/verify_mmbert_v15.py
```

The sealed batch 0027 is consumed. `eval/run_mmbert_v15.py` can replay inference into a new path after fetching the pinned base, but that is a reproducibility check, not a new held-out test.

The [V16 protocol](V16_PROTOCOL.md) continues v11 on new abstract D1/S1 prose. The selected epoch-1 adapter, shared cutoff, development scores, and [failed test result](../BENCHMARK.md#frozen-romanian-d1s1-prose-repair-batch-0030) are preserved in private Git. Check the selection and frozen comparison without downloading a base:

```sh
python eval/verify_committed_v16_selection.py
python eval/verify_mmbert_v16.py
```

With the pinned base and `.[train]` installed, `MMBertResearchClassifier(base_path, "models/ro-mmbert-v16-abstract")` loads the selected artifact. It reproduced one saved development score locally. The selected artifact's manifest records development selection; the test failure is in the immutable benchmark and run file. Batch 0030 cannot guide a new cutoff or checkpoint.

V17 kept V16 weights fixed and selected a D1 cutoff on batch 0031 before testing on batch 0032. The cutoff failed to transfer. `python eval/verify_mmbert_v17.py` recomputes both policies from the saved score vectors without downloading a model. Both batches are consumed.

V18 uses a separate [four-factor abstract D1 adapter](../models/ro-mmbert-v18-factor-abstract/README.md) selected on batch 0033, with a failed development gate. The [diagnostic test](../BENCHMARK.md#prospective-romanian-factorized-d1-study-batch-0034) on batch 0034 also failed. Recompute the selection, adapter hashes, per-field counts, and test decisions without downloading a base:

```sh
python eval/verify_committed_v18_selection.py
python eval/verify_mmbert_v18.py
```

With the pinned base and `.[train]` installed, `D1FactorResearchClassifier(base_path, "models/ro-mmbert-v18-factor-abstract")` loads the selected adapter. It reproduced one saved development factor-score vector locally within floating-point tolerance. This class is for research on invented state descriptions only; the model did not meet its registered D1 target.

V19 and V20 preserve their selected mmBERT adapters, raw test scores, and failed prospective comparisons. Their saved results can be checked without model downloads:

```sh
python eval/verify_committed_v19_selection.py
python eval/verify_mmbert_v19.py
python eval/verify_committed_v20_selection.py
python eval/verify_mmbert_v20.py
```

V21 used a different, pinned multilingual NLI base with four Romanian hypotheses. Its selected adapter failed the development gate and the sealed batch-0038 comparison. To reproduce the saved arithmetic, no model download is needed:

```sh
python eval/verify_committed_v21_selection.py
python eval/verify_nli_v21.py
```

To replay **all** 1,248 batch-0038 scores from the private adapters after installing `.[train]`, fetch both pinned bases. The NLI download is about 557 MB. This replay checks reproducibility of an already consumed failed test; it does not create a new held-out result:

```sh
MMBERT_PATH=$(python training/fetch_mmbert.py)
NLI_PATH=$(python training/fetch_nli.py)
python eval/replay_nli_v21.py --nli-base-model-path "$NLI_PATH" --mmbert-base-model-path "$MMBERT_PATH"
```

`NLIResearchClassifier(nli_base_path, "models/ro-nli-v21-abstract")` loads the selected V21 adapter for abstract-card research. Its four scores are entailment-versus-contradiction diagnostics, not calibrated child-safety probabilities. Do not use it to screen child messages.

V22 continued the selected V16 mmBERT adapter using equal exposure to seven already consumed Romanian D1 surface families. It selected an epoch on consumed development batches 0037 and 0038, failed its registered development gate, and has **no new held-out test**. Verify its selected weight, all three epoch score streams, and the failed gate without downloading a base:

```sh
python eval/verify_committed_v22_selection.py
```

With the pinned mmBERT base, `MMBertResearchClassifier(base_path, "models/ro-mmbert-v22-abstract")` loads the selected private research adapter. Its `0.5` cutoff and development failure are preserved. It is not a child-message detector.

To repeat the selected adapter's 192 development scores from the pinned base and compare every value with the committed score stream:

```sh
MMBERT_PATH=$(python training/fetch_mmbert.py)
python eval/replay_mmbert_v22.py --base-model-path "$MMBERT_PATH"
```

V23 continued the same V16 adapter with positive-class weighting and online group-loss weights. It selected epoch 2 on the already consumed 0037/0038 development surfaces and **failed** the unchanged gate. To recompute all epoch decisions, the selection, weight hashes, and the failed result without a base download:

```sh
python eval/verify_committed_v23_selection.py
```

The selected `models/ro-mmbert-v23-abstract` adapter remains a private research artifact, not a detector for child messages. With the pinned base and `.[train]` installed, repeat all 192 selected development scores:

```sh
MMBERT_PATH=$(python training/fetch_mmbert.py)
python eval/replay_mmbert_v23.py --base-model-path "$MMBERT_PATH"
```
