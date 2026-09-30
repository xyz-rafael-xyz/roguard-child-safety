"""Load the frozen V27 factorized metadata adapter for research scoring only."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .mmbert_study import MODEL_ID, REVISION, verify_base
from .review import sha256
from .v26_facts import score_rows


class V27BalancedResearchClassifier:
    def __init__(self, base_model_path: str | Path, artifact_dir: str | Path):
        import torch
        from peft import PeftModel
        from safetensors import safe_open
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        base, artifact = Path(base_model_path).resolve(), Path(artifact_dir).resolve()
        manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
        thresholds = manifest.get("thresholds")
        if (manifest.get("study") != "v27_bilingual_balanced_fact_synthetic" or
                manifest.get("status") != "selected_before_test_reveal" or
                manifest.get("scope") != "invented_abstract_metadata_only" or
                manifest.get("base_model") != MODEL_ID or
                manifest.get("base_revision") != REVISION or
                manifest.get("weight_sha256") != sha256(artifact / "adapter_model.safetensors") or
                manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
                manifest.get("prompt_sha256") != sha256(Path(__file__).with_name("v26_facts.py")) or
                manifest.get("loader_sha256") != sha256(Path(__file__)) or
                not isinstance(thresholds, dict) or
                set(thresholds) != {"ro:D1", "ro:S1", "uk:D1", "uk:S1"} or
                any(type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value <= 1 for value in thresholds.values())):
            raise ValueError("V27 selected adapter, prompt, or cutoff differs")
        verify_base(base)
        with safe_open(artifact / "adapter_model.safetensors", framework="pt", device="cpu") as stream:
            if (stream.get_slice("base_model.model.classifier.weight").get_shape() != [6, 768] or
                    stream.get_slice("base_model.model.classifier.bias").get_shape() != [6]):
                raise ValueError("V27 fitted six-fact head is missing")
        self.thresholds = thresholds
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        foundation = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=6, use_safetensors=False, local_files_only=True)
        self.model = PeftModel.from_pretrained(foundation, artifact,
                                             is_trainable=False).to(self.device)

    def score_cards(self, rows: list[dict]) -> tuple[list[float], list[list[float]]]:
        return score_rows(self.model, self.tokenizer, rows, self.device)
