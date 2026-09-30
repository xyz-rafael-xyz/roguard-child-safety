"""Optional Hugging Face and MLX classifier backends."""

from __future__ import annotations

import json
from pathlib import Path

from .review import CATEGORIES, sha256
from .prompt import make_prompt, make_prompt_v2, make_prompt_v3, parse_codes
from .prompt_v4 import applicable_codes, make_prompt_v4, parse_binary
from .review import APPLIES_TO


def model_input(text: str, language: str, source_kind: str) -> str:
    return f"language={language}\nsource_kind={source_kind}\ntext={text}"


class HFClassifier:
    def __init__(self, adapter_dir: str | Path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        adapter_dir = Path(adapter_dir)
        metadata = json.loads((adapter_dir / "roguard_metadata.json").read_text(encoding="utf-8"))
        if tuple(metadata["categories"]) != CATEGORIES:
            raise ValueError("Adapter category order differs from package taxonomy")
        base_id = metadata["base_model"]
        self.language = metadata["language"]
        self.tokenizer = AutoTokenizer.from_pretrained(base_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForSequenceClassification.from_pretrained(
            base_id, num_labels=len(CATEGORIES), problem_type="multi_label_classification"
        )
        base.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.torch = torch

    def score(self, text: str, language: str, source_kind: str) -> dict[str, float]:
        if language != self.language:
            raise ValueError("Adapter was trained for another language")
        inputs = self.tokenizer(model_input(text, language, source_kind), return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probabilities = self.torch.sigmoid(logits).cpu().tolist()
        return dict(zip(CATEGORIES, probabilities))


class MLXClassifier:
    """Code-only QLoRA backend; scores are hard 0/1 outputs, not probabilities."""

    def __init__(self, adapter_dir: str | Path, *, binary_parser=None,
                 base_model_path: str | Path | None = None):
        adapter_dir = Path(adapter_dir)
        metadata = json.loads((adapter_dir / "roguard_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("backend") != "mlx_lm_codes" or tuple(metadata["categories"]) != CATEGORIES:
            raise ValueError("Incompatible MLX RoGuard adapter")
        prompt_version = metadata.get("prompt_version", "v1")
        if prompt_version not in ("v1", "v2", "v3", "v4"):
            raise ValueError("Unknown RoGuard prompt version")
        manifest_path = adapter_dir / "research.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prompt_file = "prompt_v4.py" if prompt_version == "v4" else "prompt.py"
            if (manifest.get("artifact_kind") != "experimental_romanian_abstract_card_mlx_adapter" or
                    manifest.get("adapter_weight_sha256") != sha256(adapter_dir / "adapters.safetensors") or
                    manifest.get("adapter_config_sha256") != sha256(adapter_dir / "adapter_config.json") or
                    manifest.get("metadata_sha256") != sha256(adapter_dir / "roguard_metadata.json") or
                    manifest.get("prompt_sha256") != sha256(Path(__file__).with_name(prompt_file)) or
                    manifest.get("prompt_version", prompt_version) != prompt_version or
                    manifest.get("base_model") != metadata.get("base_model_id") or
                    manifest.get("base_revision") != metadata.get("base_revision") or
                    metadata.get("language") != "ro"):
                raise ValueError("Research MLX adapter differs from its pinned manifest")
        chosen_base = base_model_path if base_model_path is not None else metadata.get("base_model_path")
        if not chosen_base:
            raise ValueError("Supply a local pinned base_model_path for this portable research adapter")
        if manifest_path.exists() and not Path(chosen_base).is_dir():
            raise ValueError("Portable research adapter requires an existing local base directory")
        from mlx_lm import generate, load

        self.language = metadata["language"]
        self.prompt_version = prompt_version
        self.binary_parser = binary_parser or parse_binary
        self.prompt_fn = {"v1": make_prompt, "v2": make_prompt_v2, "v3": make_prompt_v3}.get(prompt_version)
        self.model, self.tokenizer = load(str(chosen_base), adapter_path=str(adapter_dir))
        self.generate = generate

    def score(self, text: str, language: str, source_kind: str) -> dict[str, float]:
        if language != self.language:
            raise ValueError("Adapter was trained for another language")
        row = {"language": language, "source_kind": source_kind, "text": text}
        if getattr(self, "prompt_version", "v1") == "v4":
            scores = {code: 0.0 for code in CATEGORIES}
            for code in applicable_codes(source_kind):
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": make_prompt_v4(row, code)}],
                    tokenize=False, add_generation_prompt=True,
                )
                raw = self.generate(self.model, self.tokenizer, prompt=prompt, max_tokens=8, verbose=False)
                answer = getattr(self, "binary_parser", parse_binary)(raw)
                if answer is None:
                    raise ValueError("Model output could not be parsed as a binary decision")
                scores[code] = float(answer)
            return scores
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": self.prompt_fn(row)}],
            tokenize=False, add_generation_prompt=True
        )
        raw = self.generate(self.model, self.tokenizer, prompt=prompt, max_tokens=24, verbose=False)
        codes = parse_codes(raw)
        if codes is None or any(source_kind not in APPLIES_TO[code] for code in codes):
            raise ValueError("Model output could not be parsed as applicable category codes")
        return {code: float(code in codes) for code in CATEGORIES}
