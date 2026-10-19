---
library_name: mlx
license: gemma
base_model: google/gemma-4-E4B-it
tags:
- mlx
- mlx-vlm
- gemma4
- image-text-to-text
---

# mlx-community/gemma-4-e4b-it-4bit

MLX conversion of [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) for Apple silicon.

- Source revision: `fee6332c1abaafb77f6f9624236c63aa2f1d0187`
- Variant: `4bit`
- Converted with `mlx_vlm.convert` from the local `mlx-vlm` checkout.

## Usage

```bash
pip install mlx-vlm
python -m mlx_vlm.generate --model mlx-community/gemma-4-e4b-it-4bit --prompt "Describe this image." --image path/to/image.jpg
```
