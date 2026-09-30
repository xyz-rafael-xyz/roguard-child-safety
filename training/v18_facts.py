"""Reconstruct exact D1 factor targets from registered abstract generators."""

from __future__ import annotations

import random
import runpy
from pathlib import Path

from roguard.prompt_v18 import FIELDS, decision
from roguard.review import load_approved

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILD_D1 = REFERENCE["BUILDERS"]["D1"]


def _factor_values(facts: dict) -> dict[str, bool]:
    result = {
        "minor_source": facts["role"] == "minor",
        "support_anchor": facts["anchor"],
        "indirect_pattern": facts["pattern"],
        "explicit_request": facts["explicit"],
    }
    if set(result) != set(FIELDS):
        raise AssertionError("Factor vocabulary differs")
    return result


def facts_for_batch(batch_id: str) -> list[dict[str, bool]]:
    if batch_id == "batch-0028":
        offset, pairs, surfaces, stride = 804, 24, 4, 17
    elif batch_id == "batch-0031":
        offset, pairs, surfaces, stride = 1201, 24, 1, 19
    elif batch_id == "batch-0032":
        offset, pairs, surfaces, stride = 1409, 24, 1, 19
    elif batch_id in ("batch-0033", "batch-0034"):
        module = runpy.run_path(str(Path(__file__).with_name("generate_factor_holdout.py")))
        return [_factor_values(item) for item in module["build_d1_facts"](batch_id)]
    else:
        raise ValueError("No registered D1 factor reconstruction for batch")
    values = []
    for index in range(pairs):
        for _ in range(surfaces):
            state_rng = random.Random(offset + index * stride)
            _, negative, positive, judge = BUILD_D1(index + offset, state_rng, "unused-symbol")
            if judge(negative) or not judge(positive):
                raise AssertionError("Invalid reconstructed D1 pair")
            values.extend((_factor_values(negative), _factor_values(positive)))
    return values


def load_d1_rows_with_facts(root: Path, batch_id: str) -> list[tuple[dict, dict[str, bool]]]:
    rows = [row for row in load_approved(root, [batch_id]) if row["source_kind"] == "message"]
    facts = facts_for_batch(batch_id)
    if len(rows) != len(facts) or any(
        row["labels"] != (["D1"] if decision(value) else [])
        for row, value in zip(rows, facts)
    ):
        raise ValueError("D1 factor targets differ from attested reference labels")
    return list(zip(rows, facts))
