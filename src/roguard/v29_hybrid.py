"""Research-only hybrid with Romanian D1 anchor read by the frozen fact head."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .review import sha256
from .v26_facts import decision_score
from .v28_hybrid import (CELLS, HYPOTHESES, NLI_HASHES, NLI_MODEL_ID,
                         NLI_REVISION, HybridComponents, abstract_body)


def required_hypotheses(row: dict) -> tuple[int, ...]:
    if row.get("language") not in HYPOTHESES or row.get("source_kind") not in ("message", "response"):
        raise ValueError("V29 language or source kind differs")
    if row["language"] == "ro" and row["source_kind"] == "message":
        return (0, 2, 3)
    return (0,) if row["source_kind"] == "message" else (4,)


def compose(row: dict, encoder_facts: list[float], nli_facts: dict[int, float]) -> tuple[float, list[float]]:
    if len(encoder_facts) != 6 or set(nli_facts) != set(required_hypotheses(row)):
        raise ValueError("V29 required fact evidence is missing")
    facts = list(encoder_facts)
    for index, value in nli_facts.items():
        facts[index] = value
    return decision_score(facts, row["source_kind"]), facts


class V29HybridComponents(HybridComponents):
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
                    [item[2] for item in part], [item[3] for item in part],
                    padding=True, truncation=True, max_length=256,
                    return_tensors="pt").to(self.device)
                logits = self.nli_model(**encoded).logits.float()
                probabilities = torch.softmax(logits[:, [0, 2]], dim=-1)[:, 0].cpu().tolist()
                for (row_index, fact_index, _, _), value in zip(part, probabilities):
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("V29 entailment score is invalid")
                    nli[row_index][fact_index] = value
        scores, evidence = [], []
        for row, encoder, nli_facts in zip(rows, encoder_vectors, nli):
            score, composed = compose(row, encoder, nli_facts)
            scores.append(score)
            evidence.append({
                "rule": "ro_d1_nli_role_patterns_encoder_anchor"
                        if row["language"] == "ro" and row["source_kind"] == "message"
                        else "uk_d1_nli_role" if row["source_kind"] == "message"
                        else "s1_nli_applicability",
                "nli_fact_scores": {str(key): value for key, value in nli_facts.items()},
                "composed_fact_scores": composed,
            })
        return scores, evidence


class V29HybridResearchClassifier(V29HybridComponents):
    def __init__(self, mmbert_base: str | Path, nli_base: str | Path,
                 prior_artifact: str | Path, artifact_dir: str | Path):
        artifact = Path(artifact_dir).resolve()
        manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
        thresholds = manifest.get("thresholds")
        if (manifest.get("study") != "v29_bilingual_anchor_routed_synthetic" or
                manifest.get("status") != "selected_before_test_reveal" or
                manifest.get("scope") != "invented_abstract_metadata_only" or
                manifest.get("hybrid_source_sha256") != sha256(Path(__file__)) or
                manifest.get("parent_hybrid_source_sha256") != sha256(
                    Path(__file__).with_name("v28_hybrid.py")) or
                manifest.get("nli_model_id") != NLI_MODEL_ID or
                manifest.get("nli_revision") != NLI_REVISION or
                manifest.get("nli_files_sha256") != NLI_HASHES or
                manifest.get("prior_artifact_manifest_sha256") != sha256(
                    Path(prior_artifact) / "research.json") or
                not isinstance(thresholds, dict) or set(thresholds) != set(CELLS) or
                any(type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value <= 1 for value in thresholds.values())):
            raise ValueError("V29 selected hybrid artifact or cutoffs differ")
        self.thresholds = thresholds
        super().__init__(mmbert_base, nli_base, prior_artifact)
