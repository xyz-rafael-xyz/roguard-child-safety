"""Optional local semantic features. No network or message cache at inference."""
from __future__ import annotations
import math
from pathlib import Path
from .mmbert_study import MODEL_ID, REVISION, verify_base

MAX_TOKENS = 256

class SemanticEncoder:
    def __init__(self, path, device='cpu', quantize=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        path = Path(path)
        verify_base(path)
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                               attn_implementation='eager').eval()
        if quantize:
            if device != 'cpu':
                raise ValueError('Dynamic quantization requires CPU')
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.to(device)

    def encode(self, texts, batch_size=16):
        import numpy as np
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise ValueError('Invalid batch size')
        output = []
        torch = self.torch
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                tokens = self.tokenizer(texts[offset:offset+batch_size], padding=True,
                                        truncation=False, return_tensors='pt')
                if tokens['input_ids'].shape[1] > MAX_TOKENS:
                    raise ValueError('Semantic token limit exceeded; shorten the conversation')
                tokens = {key: value.to(self.device) for key,value in tokens.items()}
                hidden = self.model(**tokens).last_hidden_state.float()
                mask = tokens['attention_mask'].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, dim=1) * 10
                output.extend(pooled.cpu().numpy())
        return np.asarray(output)


def head_scores(features, head):
    import numpy as np
    weights = np.asarray(head['weights'], dtype=float)
    if weights.ndim != 1 or not np.isfinite(weights).all() or not math.isfinite(head['intercept']):
        raise ValueError('Invalid semantic head')
    logits = np.asarray(features) @ weights + head['intercept']
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
