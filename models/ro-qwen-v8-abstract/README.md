# Selected Qwen v8 research adapter

This private artifact preserves the selected v8 LoRA weight bytes and a minimal portable MLX config. It [failed its fresh symbolic test](../../BENCHMARK.md#frozen-romanian-balanced-qwen-comparison-batch-0018) at 13/72 complete pairs. Supply the pinned local Qwen 4-bit base and use `parse_qwen_binary` for its registered output format. The [artifact verifier](../../eval/verify_committed_mlx_adapters.py) checks the frozen selection and can replay one consumed test card. This adapter is not validated for child messages.
