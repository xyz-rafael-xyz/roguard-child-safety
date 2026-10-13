"""Optional local semantic features. No network or message cache at inference."""
from __future__ import annotations
import math
import re
import unicodedata
from pathlib import Path
from .mmbert_study import MODEL_ID, REVISION, verify_base

MAX_TOKENS = 256

def normalize_semantic_text(text):
    """Canonicalize surface forms without removing roles or turn boundaries.

    Ordinary sentence casing is retained. Fully capitalized lines use sentence
    case; this cannot reconstruct proper-name casing and is not a language fix.
    """
    lines=[]
    for line in unicodedata.normalize('NFC',text).split('\n'):
        line=re.sub(r'[^\S\n]+',' ',line)
        match=re.match(r'^(minor|assistant|adult|other|unknown): ',line)
        prefix=match.group(0) if match else ''
        body=line[len(prefix):]
        if body.isupper():
            body=body.lower()
            for index,char in enumerate(body):
                if char.isalpha():
                    body=body[:index]+char.upper()+body[index+1:]
                    break
        lines.append(prefix+body)
    return '\n'.join(lines)


class SemanticEncoder:
    def __init__(self, path, device='cpu', quantize=False, normalize_inputs=True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        path = Path(path)
        verify_base(path)
        self.torch = torch
        self.device = device
        self.normalize_inputs = normalize_inputs
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                               attn_implementation='eager').eval()
        if quantize:
            if device != 'cpu':
                raise ValueError('Dynamic quantization requires CPU')
            if torch.backends.quantized.engine == 'none' and 'qnnpack' in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = 'qnnpack'
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
                tokens = self.tokenizer([normalize_semantic_text(t) if self.normalize_inputs else t for t in texts[offset:offset+batch_size]], padding=True,
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
