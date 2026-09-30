"""Load the frozen V21 Romanian abstract D1 NLI research adapter."""

from __future__ import annotations

import json
from pathlib import Path

from .mmbert_study import verify_adapter_head
from .nli_v21 import (FIELDS, MAX_LENGTH, MODEL_ID, REVISION,
                       field_probabilities, hypotheses_for, verify_base)
from .review import sha256


class NLIResearchClassifier:
    """Score four invented D1 state fields; outputs are not child-risk estimates."""

    def __init__(self, base_model_path: str | Path, adapter_dir: str | Path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        base, adapter = Path(base_model_path).resolve(), Path(adapter_dir).resolve()
        manifest = json.loads((adapter / "research.json").read_text(encoding="utf-8"))
        if (manifest.get("artifact_kind") != "experimental_romanian_abstract_d1_nli_adapter" or
                manifest.get("study") != "v21_nli_adapted_d1" or
                manifest.get("base_model") != MODEL_ID or
                manifest.get("base_revision") != REVISION or
                manifest.get("trained_language") != "ro" or
                manifest.get("trained_domain") != "abstract_symbolic_cards_only" or
                manifest.get("evaluation_status") != "selected_development_failed_gate" or
                manifest.get("field_cutoff") != 0.5 or
                manifest.get("adapter_weight_sha256") != sha256(adapter / "adapter_model.safetensors") or
                manifest.get("adapter_config_sha256") != sha256(adapter / "adapter_config.json") or
                manifest.get("prompt_sha256") != sha256(Path(__file__).with_name("nli_v21.py"))):
            raise ValueError("V21 NLI research adapter differs from pinned manifest")
        config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if (config.get("base_model_name_or_path") != MODEL_ID or
                config.get("r") != 16 or config.get("lora_alpha") != 32 or
                set(config.get("target_modules", [])) != {"query_proj", "value_proj"} or
                not {"classifier", "pooler"} <= set(config.get("modules_to_save", [])) or
                not set(config.get("modules_to_save", [])) <= {"classifier", "pooler", "score"}):
            raise ValueError("V21 NLI adapter configuration differs")
        verify_base(base)
        verify_adapter_head(adapter, num_labels=3, require_pooler=True)
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        foundation = AutoModelForSequenceClassification.from_pretrained(
            base, use_safetensors=True, local_files_only=True).float()
        if foundation.config.id2label != {0: "entailment", 1: "neutral", 2: "contradiction"}:
            raise ValueError("V21 NLI base label order differs")
        self.model = PeftModel.from_pretrained(foundation, adapter, is_trainable=False).to(self.device)
        self.model.eval()
        self.torch = torch

    def score(self, text: str) -> dict[str, float]:
        prompts = hypotheses_for(text)
        encoded = self.tokenizer([item[0] for item in prompts],
                                 [item[1] for item in prompts],
                                 padding=True, truncation=True,
                                 max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            values = field_probabilities(self.model(**encoded).logits)
        return dict(zip(FIELDS, values))
