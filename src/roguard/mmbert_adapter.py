"""Load the pinned, experimental v11 abstract-card adapter from Git."""

from __future__ import annotations

import json
from pathlib import Path

from .mmbert_study import (MAX_LENGTH, MODEL_ID, REVISION, verify_adapter_head, verify_base)
from .prompt_v4 import applicable_codes, make_prompt_v4
from .review import CATEGORIES, KINDS, sha256


class MMBertResearchClassifier:
    """Romanian symbolic-card scorer; scores are not calibrated risk estimates."""

    def __init__(self, base_model_path: str | Path, adapter_dir: str | Path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        base = Path(base_model_path).resolve()
        adapter = Path(adapter_dir).resolve()
        manifest = json.loads((adapter / "research.json").read_text(encoding="utf-8"))
        if (manifest.get("artifact_kind") != "experimental_romanian_abstract_card_adapter" or
                manifest.get("base_model") != MODEL_ID or
                manifest.get("base_revision") != REVISION or
                manifest.get("trained_language") != "ro" or
                manifest.get("trained_domain") != "abstract_symbolic_cards_only" or
                manifest.get("evaluation_status") not in (
                    "failed_registered_synthetic_target", "selected_development_only") or
                manifest.get("adapter_weight_sha256") != sha256(adapter / "adapter_model.safetensors") or
                manifest.get("adapter_config_sha256") != sha256(adapter / "adapter_config.json") or
                manifest.get("prompt_sha256") != sha256(Path(__file__).with_name("prompt_v4.py"))):
            raise ValueError("Research adapter or prompt differs from its pinned manifest")
        config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if (config.get("base_model_name_or_path") != MODEL_ID or
                config.get("r") != 16 or config.get("lora_alpha") != 32):
            raise ValueError("Research adapter configuration differs")
        verify_base(base)
        verify_adapter_head(adapter)
        threshold = manifest.get("shared_threshold")
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ValueError("Invalid research threshold")
        self.thresholds = {code: float(threshold) for code in CATEGORIES}
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=2, use_safetensors=False, local_files_only=True)
        self.model = PeftModel.from_pretrained(model, adapter, is_trainable=False).to(self.device)
        self.model.eval()
        self.torch = torch

    def score(self, text: str, language: str, source_kind: str) -> dict[str, float]:
        if language != "ro" or source_kind not in KINDS or not isinstance(text, str) or not text.strip():
            raise ValueError("Research adapter requires a nonempty Romanian symbolic card")
        codes = applicable_codes(source_kind)
        row = {"language": language, "source_kind": source_kind, "text": text}
        prompts = [make_prompt_v4(row, code) for code in codes]
        encoded = self.tokenizer(prompts, padding=True, truncation=True,
                                 max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(**encoded).logits.float(), dim=-1)[:, 1]
        scores = {code: 0.0 for code in CATEGORIES}
        scores.update({code: float(value) for code, value in zip(codes, probabilities.cpu().tolist())})
        return scores
