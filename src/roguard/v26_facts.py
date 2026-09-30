"""Research-only latent-fact scorer for invented Romanian/Ukrainian metadata."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .mmbert_study import MAX_LENGTH, MODEL_ID, REVISION, verify_base
from .review import sha256

PROMPTS = {
    "ro": "Extrage proprietățile consemnate în această fișă fictivă. Fiecare proprietate se citește separat. Fișă:",
    "uk": "Визнач окремі властивості в цій вигаданій картці. Кожну властивість читай незалежно. Картка:",
}
FACTS = ("source_role_minor", "safety_or_support_anchor",
         "indirect_or_repeated_support_pattern", "explicit_support_request",
         "applicable", "passed")


def make_input(row: dict) -> str:
    language, kind, text = row.get("language"), row.get("source_kind"), row.get("text")
    if (language not in PROMPTS or kind not in ("message", "response") or
            not isinstance(text, str) or not text.strip()):
        raise ValueError("V26 requires a Romanian or Ukrainian abstract metadata card")
    return f"{PROMPTS[language]} {text}"


def truth_vector(row: dict) -> tuple[list[float], list[float]]:
    """Training/evaluation oracle only. Inference never receives latent facts."""
    facts = row["latent_facts"]
    if row["source_kind"] == "message":
        return ([float(facts["source_role"] == "minor"),
                 float(facts["safety_or_support_anchor"]),
                 float(facts["indirect_or_repeated_support_pattern"]),
                 float(facts["explicit_support_request"]), 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    return ([0.0, 0.0, 0.0, 0.0, float(facts["applicable"]),
             float(facts["passed"])], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0])


def decision_score(fact_probabilities: list[float], source_kind: str) -> float:
    if len(fact_probabilities) != 6 or any(
            type(value) not in (int, float) or not math.isfinite(value) or
            not 0 <= value <= 1 for value in fact_probabilities):
        raise ValueError("V26 needs six finite fact probabilities")
    role, anchor, indirect, explicit, applicable, passed = fact_probabilities
    if source_kind == "message":
        return role * anchor * (1 - (1 - indirect) * (1 - explicit))
    if source_kind == "response":
        return applicable * (1 - passed)
    raise ValueError("V26 source kind differs")


def score_rows(model, tokenizer, rows: list[dict], device: str,
               batch_size: int = 8) -> tuple[list[float], list[list[float]]]:
    import torch

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("V26 batch size must be positive")
    prompts = [make_input(row) for row in rows]
    fact_scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(prompts[start:start + batch_size], padding=True,
                                truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt").to(device)
            fact_scores.extend(torch.sigmoid(model(**encoded).logits.float()).cpu().tolist())
    scores = [decision_score(values, row["source_kind"])
              for row, values in zip(rows, fact_scores)]
    return scores, fact_scores


class V26FactResearchClassifier:
    """Load the selected six-fact adapter; never expose it as a child-screening API."""

    def __init__(self, base_model_path: str | Path, artifact_dir: str | Path):
        import torch
        from peft import PeftModel
        from safetensors import safe_open
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        base, artifact = Path(base_model_path).resolve(), Path(artifact_dir).resolve()
        manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
        thresholds = manifest.get("thresholds")
        if (manifest.get("study") != "v26_bilingual_factorized_synthetic" or
                manifest.get("status") != "selected_before_test_reveal" or
                manifest.get("scope") != "invented_abstract_metadata_only" or
                manifest.get("base_model") != MODEL_ID or
                manifest.get("base_revision") != REVISION or
                manifest.get("weight_sha256") != sha256(artifact / "adapter_model.safetensors") or
                manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
                manifest.get("prompt_sha256") != sha256(Path(__file__)) or
                not isinstance(thresholds, dict) or
                set(thresholds) != {"ro:D1", "ro:S1", "uk:D1", "uk:S1"} or
                any(type(v) not in (int, float) or not math.isfinite(v) or
                    not 0 <= v <= 1 for v in thresholds.values())):
            raise ValueError("V26 selected fact adapter or prompt differs")
        verify_base(base)
        with safe_open(artifact / "adapter_model.safetensors", framework="pt", device="cpu") as stream:
            if (stream.get_slice("base_model.model.classifier.weight").get_shape() != [6, 768] or
                    stream.get_slice("base_model.model.classifier.bias").get_shape() != [6]):
                raise ValueError("V26 fitted six-fact head is missing")
        self.thresholds = thresholds
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        foundation = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=6, use_safetensors=False, local_files_only=True)
        self.model = PeftModel.from_pretrained(foundation, artifact,
                                             is_trainable=False).to(self.device)

    def score_cards(self, rows: list[dict]) -> tuple[list[float], list[list[float]]]:
        return score_rows(self.model, self.tokenizer, rows, self.device)
