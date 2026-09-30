# Selected RoMistral v4 research adapter

This private artifact preserves the selected v4 LoRA weight bytes and a minimal portable MLX config. It [failed its fresh symbolic test](../../BENCHMARK.md#frozen-romanian-binary-paired-card-comparison-batch-0012). Supply the locally prepared, pinned RoMistral 4-bit base through `MLXClassifier(..., base_model_path=...)`. The [artifact verifier](../../eval/verify_committed_mlx_adapters.py) checks the frozen selection and can replay one consumed test card. This adapter is not validated for child messages.
