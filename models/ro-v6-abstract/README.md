# Selected RoMistral v6 research adapter

This private artifact preserves the selected v6 LoRA weight bytes and a minimal portable MLX config. It [failed its fresh symbolic test](../../BENCHMARK.md#frozen-romanian-compositional-comparison-batch-0016) at 9/72 complete pairs. Supply the locally prepared, pinned RoMistral 4-bit base through `MLXClassifier(..., base_model_path=...)`. The [artifact verifier](../../eval/verify_committed_mlx_adapters.py) checks the frozen selection and can replay one consumed test card. This adapter is not validated for child messages.
