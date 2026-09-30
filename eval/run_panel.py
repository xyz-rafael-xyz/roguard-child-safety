"""Run the prior four-model panel plus RoMistral on an attested held-out set."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from roguard.prompt import make_prompt, make_prompt_v2, make_prompt_v3, parse_codes
from roguard.review import ReviewError, load_approved, taxonomy_sha256


def load_generator(entry: dict, revision: str, max_tokens: int = 32):
    if entry["engine"] == "mlx":
        from mlx_lm import generate, load
        model, tokenizer = load(entry["model"], revision=revision)

        def run(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            return generate(model, tokenizer, tokens, max_tokens=max_tokens, verbose=False)

        return run

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(entry["model"], revision=revision)
    model = AutoModelForCausalLM.from_pretrained(entry["model"], revision=revision)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def run(prompt: str) -> str:
        try:
            input_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt")
        except TypeError:
            input_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt", system_message="")
        input_ids = input_ids.to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", action="append", required=True)
    parser.add_argument("--model", required=True, choices=("qwen3_4b", "phi4_mini", "gemma3_4b", "mistral7b", "romistral7b"))
    parser.add_argument("--revision", help="Required for RoMistral; use an exact HF commit")
    parser.add_argument("--prompt-version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_approved(args.root, args.batch)
    if not rows or any(row["split"] != "test" for row in rows):
        raise ReviewError("Panel runner accepts held-out test batches only")
    registry = json.loads((Path(__file__).parent / "models.json").read_text(encoding="utf-8"))
    entry = registry[args.model]
    revision = args.revision or entry["revision"]
    if not revision:
        raise ValueError("An exact model revision is required")
    if args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    run = load_generator(entry, revision, max_tokens=args.max_tokens)
    prompt_fn = {"v1": make_prompt, "v2": make_prompt_v2, "v3": make_prompt_v3}[args.prompt_version]
    outputs = []
    for row in rows:
        prompt = prompt_fn(row)
        started = time.monotonic()
        raw = run(prompt)
        parsed = parse_codes(raw)
        outputs.append({"id": row["id"], "language": row["language"], "expected": row["labels"],
                        "raw": raw, "parsed": parsed, "strict_parse": parsed is not None,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "seconds": round(time.monotonic() - started, 3)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Do not overwrite a panel run")
    args.output.write_text(json.dumps({"model": entry["model"], "source_model": entry.get("source_model", entry["model"]), "engine": entry["engine"],
                                       "prompt_version": args.prompt_version,
                                       "max_tokens": args.max_tokens,
                                       "revision": revision, "batches": args.batch,
                                       "taxonomy_sha256": {language: taxonomy_sha256(args.root, language) for language in {row["language"] for row in rows}}, "outputs": outputs},
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
