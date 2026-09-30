"""Load the frozen V20 balanced four-output abstract D1 research adapter."""

from __future__ import annotations

import json
from pathlib import Path

from .mmbert_study import MAX_LENGTH, MODEL_ID, REVISION, verify_adapter_head, verify_base
from .prompt_v18 import FIELDS
from .review import sha256


class BalancedJointD1ResearchClassifier:
    """Score four invented state fields; scores are not child-risk estimates."""

    def __init__(self, base_model_path: str | Path, adapter_dir: str | Path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        base, adapter = Path(base_model_path).resolve(), Path(adapter_dir).resolve()
        manifest = json.loads((adapter / "research.json").read_text(encoding="utf-8"))
        if (manifest.get("artifact_kind") != "experimental_romanian_abstract_d1_balanced_joint_adapter" or
                manifest.get("study") != "v20_balanced_joint_d1" or
                manifest.get("base_model") != MODEL_ID or manifest.get("base_revision") != REVISION or
                manifest.get("trained_language") != "ro" or
                manifest.get("trained_domain") != "abstract_symbolic_cards_only" or
                manifest.get("evaluation_status") != "selected_development_failed_gate" or
                manifest.get("field_cutoff") != 0.5 or
                manifest.get("adapter_weight_sha256") != sha256(adapter / "adapter_model.safetensors") or
                manifest.get("adapter_config_sha256") != sha256(adapter / "adapter_config.json")):
            raise ValueError("V20 balanced joint adapter differs from its pinned manifest")
        config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if (config.get("base_model_name_or_path") != MODEL_ID or
                config.get("r") != 16 or config.get("lora_alpha") != 32):
            raise ValueError("V20 adapter configuration differs")
        verify_base(base)
        verify_adapter_head(adapter, num_labels=4)
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        foundation = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=4, problem_type="multi_label_classification",
            use_safetensors=False, local_files_only=True)
        self.model = PeftModel.from_pretrained(foundation, adapter, is_trainable=False).to(self.device)
        self.model.eval()
        self.torch = torch

    def score(self, text: str) -> dict[str, float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("V20 requires a nonempty abstract Romanian D1 card")
        encoded = self.tokenizer([text], padding=True, truncation=True,
                                 max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            values = self.torch.sigmoid(self.model(**encoded).logits.float())[0]
        return {field: float(value) for field, value in zip(FIELDS, values.cpu().tolist())}
