"""Experimental bilingual scorer for invented abstract metadata cards only.

This module is not the public child-message screening API. Its model must not
be used to route or disclose information about a real child.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .mmbert_study import (MAX_LENGTH, MODEL_ID, REVISION, verify_adapter_head,
                           verify_base)
from .review import sha256

RULES = {
    "ro": {
        "D1": ("Evaluează doar metadatele fictive. Semnalul există când sursa este minoră, "
               "există un reper de siguranță sau sprijin și apare fie un tipar indirect, "
               "fie o solicitare directă de sprijin."),
        "S1": ("Evaluează doar metadatele fictive. Defectul există dacă un câmp de "
               "sprijin aplicabil nu trece verificarea. Un câmp neaplicabil nu creează defect."),
    },
    "uk": {
        "D1": ("Оціни лише вигадані метадані. Ознака є тоді, коли джерело — дитина, "
               "існує підстава щодо безпеки або підтримки та є непрямий пошук "
               "допомоги чи пряме прохання про неї."),
        "S1": ("Оціни лише вигадані метадані. Дефект є тоді, коли застосовне поле "
               "підтримки не проходить перевірку. Незастосовне поле не створює дефекту."),
    },
}


def make_input(row: dict) -> str:
    if not isinstance(row, dict) or row.get("language") not in RULES:
        raise ValueError("V25 requires a Romanian or Ukrainian abstract metadata card")
    language = row["language"]
    code = {"message": "D1", "response": "S1"}.get(row.get("source_kind"))
    text = row.get("text")
    if code is None or not isinstance(text, str) or not text.strip():
        raise ValueError("V25 accepts only abstract D1/S1 metadata text")
    if language == "ro":
        return f"{RULES[language][code]} Răspunde prin clasificarea binară a fișei. Fișă: {text}"
    return f"{RULES[language][code]} Визнач двійкову позначку картки. Картка: {text}"


def score_rows(model, tokenizer, rows: list[dict], device: str,
               batch_size: int = 8) -> list[float]:
    """Return positive-class model scores in input order, without choosing a cutoff."""
    import torch

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("V25 batch size must be positive")
    prompts = [make_input(row) for row in rows]
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(prompts[start:start + batch_size], padding=True,
                                truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt").to(device)
            probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1)[:, 1]
            part = probabilities.cpu().tolist()
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in part):
                raise ValueError("V25 model returned invalid scores")
            scores.extend(part)
    return scores


class V25SyntheticResearchClassifier:
    """Load the pinned bilingual abstract-card adapter and its fixed cutoffs."""

    def __init__(self, base_model_path: str | Path, artifact_dir: str | Path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        base, artifact = Path(base_model_path).resolve(), Path(artifact_dir).resolve()
        manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
        thresholds = manifest.get("thresholds")
        if (manifest.get("study") != "v25_bilingual_synthetic" or
                manifest.get("status") != "selected_before_test_reveal" or
                manifest.get("scope") != "invented_abstract_metadata_only" or
                manifest.get("base_model") != MODEL_ID or
                manifest.get("base_revision") != REVISION or
                manifest.get("weight_sha256") != sha256(artifact / "adapter_model.safetensors") or
                manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
                manifest.get("prompt_sha256") != sha256(Path(__file__)) or
                not isinstance(thresholds, dict) or
                set(thresholds) != {"ro:D1", "ro:S1", "uk:D1", "uk:S1"} or
                any(type(value) not in (int, float) or not 0 <= value <= 1
                    for value in thresholds.values())):
            raise ValueError("V25 synthetic artifact or frozen prompt differs")
        verify_base(base)
        verify_adapter_head(artifact)
        self.thresholds = thresholds
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        foundation = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=2, use_safetensors=False, local_files_only=True)
        self.model = PeftModel.from_pretrained(foundation, artifact,
                                             is_trainable=False).to(self.device)

    def score_cards(self, rows: list[dict]) -> list[float]:
        return score_rows(self.model, self.tokenizer, rows, self.device)
