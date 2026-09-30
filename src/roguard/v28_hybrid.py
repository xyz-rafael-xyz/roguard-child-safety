"""Research-only evidence reader for invented bilingual metadata cards."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .review import sha256
from .v26_facts import V26FactResearchClassifier, decision_score

NLI_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
NLI_HASHES = {
    "config.json": "bc7b85f164a17c1b007d87fb99d3676f4a4d2e6511d2288dd3f5334dc0f34e6b",
    "model.safetensors": "7c8e29f1115986d032e92b0fbaa0bdef1062a46f658b08705f237c05014a8541",
    "tokenizer.json": "e23095eb61ba944c7be3a5d3e8ec19e37ce7ced0daa03550bde03e83c21b3f8a",
    "tokenizer_config.json": "25e9f84c6b43da7bff43a0aee8eeee2c8a39d246823d8464b9275ef704eab42f",
}
HYPOTHESES = {
    "ro": (
        "Sursa este minoră.",
        "Există un reper privind siguranța sau sprijinul.",
        "Este prezentă căutarea indirectă sau repetată a sprijinului.",
        "Există o cerere directă de sprijin.",
        "Criteriul analizat este aplicabil.",
    ),
    "uk": (
        "Джерелом є неповнолітня особа.",
        "Є підстава щодо безпеки або підтримки.",
        "Є непрямий або повторний пошук підтримки.",
        "Є пряме прохання про допомогу.",
        "Оцінюваний критерій є застосовним.",
    ),
}
CELLS = ("ro:D1", "ro:S1", "uk:D1", "uk:S1")


def abstract_body(row: dict) -> str:
    if (row.get("language") not in HYPOTHESES or
            row.get("source_kind") not in ("message", "response") or
            not isinstance(row.get("text"), str)):
        raise ValueError("V28 accepts only Romanian/Ukrainian abstract metadata cards")
    parts = row["text"].split(": ", 1)
    if len(parts) != 2 or ". " not in parts[1]:
        raise ValueError("V28 abstract card needs a labeled body and closing sentence")
    body, closing = parts[1].rsplit(". ", 1)
    if not body or not closing:
        raise ValueError("V28 abstract card is incomplete")
    return body


def required_hypotheses(row: dict) -> tuple[int, ...]:
    if row.get("language") not in HYPOTHESES or row.get("source_kind") not in ("message", "response"):
        raise ValueError("V28 language or source kind differs")
    if row["language"] == "ro" and row["source_kind"] == "message":
        return (0, 1, 2, 3)
    return (0,) if row["source_kind"] == "message" else (4,)


def compose(row: dict, encoder_facts: list[float], nli_facts: dict[int, float]) -> tuple[float, list[float]]:
    if len(encoder_facts) != 6 or set(nli_facts) != set(required_hypotheses(row)):
        raise ValueError("V28 required fact evidence is missing")
    facts = list(encoder_facts)
    for index, value in nli_facts.items():
        facts[index] = value
    return decision_score(facts, row["source_kind"]), facts


class HybridComponents:
    """Frozen component models; no training or routing action occurs here."""

    def __init__(self, mmbert_base: str | Path, nli_base: str | Path,
                 prior_artifact: str | Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.prior_artifact = Path(prior_artifact).resolve()
        self.nli_base = Path(nli_base).resolve()
        if any(sha256(self.nli_base / name) != expected
               for name, expected in NLI_HASHES.items()):
            raise ValueError("V28 pinned multilingual entailment model differs")
        self.fact_model = V26FactResearchClassifier(mmbert_base, self.prior_artifact)
        self.device = self.fact_model.device
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_base,
                                                           local_files_only=True)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            self.nli_base, local_files_only=True).to(self.device).eval()
        if self.nli_model.config.id2label != {
                0: "entailment", 1: "neutral", 2: "contradiction"}:
            raise ValueError("V28 entailment label order differs")

    def score_cards(self, rows: list[dict]) -> tuple[list[float], list[dict]]:
        import torch

        _, encoder_vectors = self.fact_model.score_cards(rows)
        tasks = [(row_index, fact_index, abstract_body(row),
                  HYPOTHESES[row["language"]][fact_index])
                 for row_index, row in enumerate(rows)
                 for fact_index in required_hypotheses(row)]
        nli = [{} for _ in rows]
        with torch.inference_mode():
            for start in range(0, len(tasks), 16):
                part = tasks[start:start + 16]
                encoded = self.nli_tokenizer(
                    [task[2] for task in part], [task[3] for task in part],
                    padding=True, truncation=True, max_length=256,
                    return_tensors="pt").to(self.device)
                logits = self.nli_model(**encoded).logits.float()
                probabilities = torch.softmax(logits[:, [0, 2]], dim=-1)[:, 0].cpu().tolist()
                for (row_index, fact_index, _, _), value in zip(part, probabilities):
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("V28 entailment score is invalid")
                    nli[row_index][fact_index] = value
        scores, evidence = [], []
        for row, encoder, nli_facts in zip(rows, encoder_vectors, nli):
            score, composed = compose(row, encoder, nli_facts)
            scores.append(score)
            evidence.append({
                "rule": "ro_d1_nli_all" if row["language"] == "ro" and
                        row["source_kind"] == "message" else
                        "uk_d1_nli_role" if row["source_kind"] == "message" else
                        "s1_nli_applicability",
                "nli_fact_scores": {str(key): value for key, value in nli_facts.items()},
                "composed_fact_scores": composed,
            })
        return scores, evidence


class V28HybridResearchClassifier(HybridComponents):
    """Load cutoffs frozen on V28 development before opening its test."""

    def __init__(self, mmbert_base: str | Path, nli_base: str | Path,
                 prior_artifact: str | Path, artifact_dir: str | Path):
        artifact = Path(artifact_dir).resolve()
        manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
        thresholds = manifest.get("thresholds")
        if (manifest.get("study") != "v28_bilingual_evidence_hybrid_synthetic" or
                manifest.get("status") != "selected_before_test_reveal" or
                manifest.get("scope") != "invented_abstract_metadata_only" or
                manifest.get("hybrid_source_sha256") != sha256(Path(__file__)) or
                manifest.get("nli_model_id") != NLI_MODEL_ID or
                manifest.get("nli_revision") != NLI_REVISION or
                manifest.get("nli_files_sha256") != NLI_HASHES or
                manifest.get("prior_artifact_manifest_sha256") != sha256(
                    Path(prior_artifact) / "research.json") or
                not isinstance(thresholds, dict) or set(thresholds) != set(CELLS) or
                any(type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value <= 1 for value in thresholds.values())):
            raise ValueError("V28 selected hybrid artifact or cutoffs differ")
        self.thresholds = thresholds
        super().__init__(mmbert_base, nli_base, prior_artifact)
