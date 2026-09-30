"""Fine-tune only explicitly selected and approved synthetic batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.calibrate import calibrate
from roguard.model import model_input
from roguard.review import CATEGORIES, load_train_dev, taxonomy_sha256


def prepare(root: Path, batch_ids: list[str], language: str) -> tuple[list[dict], list[dict]]:
    return load_train_dev(root, batch_ids, language)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", action="append", required=True, help="Explicit reviewed batch ID; repeat for each batch")
    parser.add_argument("--language", choices=("ro", "uk"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="OpenLLM-Ro/RoMistral-7b-Instruct")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.language == "uk" and args.model_id == "OpenLLM-Ro/RoMistral-7b-Instruct":
        raise ValueError("Select and review a Ukrainian base model explicitly")
    train, dev = prepare(args.root, args.batch, args.language)

    # Import heavy libraries only after the human-review gate has passed.
    import numpy as np
    import torch
    import yaml
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
    config = yaml.safe_load((args.root / "training" / "lora_config.yaml").read_text(encoding="utf-8"))
    max_length = int(config["max_length"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=len(CATEGORIES), problem_type="multi_label_classification"
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=int(config["rank"]), lora_alpha=int(config["alpha"]),
        lora_dropout=float(config["dropout"]), target_modules=list(config["target_modules"]),
        modules_to_save=["score"]
    ))

    def encode(rows: list[dict]) -> list[dict]:
        data = []
        for row in rows:
            item = tokenizer(model_input(row["text"], row["language"], row["source_kind"]), truncation=True, max_length=max_length)
            item["labels"] = [float(code in row["labels"]) for code in CATEGORIES]
            data.append(item)
        return data

    class FloatLabelCollator:
        def __init__(self):
            self.padding = DataCollatorWithPadding(tokenizer)

        def __call__(self, features):
            labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.float32)
            inputs = [{k: v for k, v in feature.items() if k != "labels"} for feature in features]
            batch = self.padding(inputs)
            batch["labels"] = labels
            return batch

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "trainer"), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
            learning_rate=2e-4, eval_strategy="epoch", save_strategy="epoch",
            report_to="none", remove_unused_columns=False, seed=20260929,
        ),
        train_dataset=encode(train), eval_dataset=encode(dev), data_collator=FloatLabelCollator(),
    )
    trainer.train()
    prediction = trainer.predict(encode(dev)).predictions
    probabilities = 1 / (1 + np.exp(-prediction))
    thresholds = {args.language: calibrate(
        [(dict(zip(CATEGORIES, probabilities[i].tolist())), set(row["labels"]), row["source_kind"]) for i, row in enumerate(dev)]
    )}
    adapter = output / "adapter"
    trainer.save_model(str(adapter))
    (adapter / "roguard_metadata.json").write_text(json.dumps({
        "base_model": args.model_id, "language": args.language, "categories": CATEGORIES, "batches": args.batch,
        "taxonomy_sha256": taxonomy_sha256(args.root, args.language), "thresholds": thresholds,
        "lora_config": config,
        "calibration_split": "dev", "synthetic_only": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"adapter": str(adapter), "train_rows": len(train), "dev_rows": len(dev)}))


if __name__ == "__main__":
    main()
