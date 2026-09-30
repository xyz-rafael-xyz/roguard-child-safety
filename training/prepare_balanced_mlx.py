"""Export equal numbers of binary training tasks per taxonomy category."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from roguard.review import CATEGORIES, sha256
try:
    from .prepare_mlx import prepare
except ImportError:
    from prepare_mlx import prepare

CODE = re.compile(r"Regula pentru (D1|R1|A1|P1|G1|S1):")
REPEAT = {"D1": 3, "R1": 3, "A1": 1, "P1": 3, "G1": 3, "S1": 1}


def prepare_balanced(root: Path, output: Path) -> dict:
    """Keep the v4 prompt and dev export, changing only train task exposure."""
    output = output.resolve()
    manifest = prepare(root, ["batch-0014", "batch-0015"], "ro", output, "v4")
    train_path = output / "train.jsonl"
    tasks = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    expanded = []
    for task in tasks:
        match = CODE.search(task["prompt"])
        if match is None:
            raise ValueError("Training prompt lacks its declared category")
        expanded.extend([task] * REPEAT[match.group(1)])
    counts = Counter((CODE.search(task["prompt"]).group(1), task["completion"]) for task in expanded)
    if any(counts[(code, answer)] != 144 for code in CATEGORIES for answer in ("da", "nu")):
        raise ValueError("Equal-category binary export failed")
    train_path.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in expanded),
                          encoding="utf-8")
    manifest.update(train_tasks=len(expanded), train_sha256=sha256(train_path),
                    format="mlx_lm_binary_equal_category_v1",
                    balance_rule="three_copies_of_D1_R1_P1_G1_tasks_train_only",
                    train_tasks_per_category={code: {"da": counts[(code, "da")], "nu": counts[(code, "nu")]}
                                              for code in CATEGORIES})
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return manifest
