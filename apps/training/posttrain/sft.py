"""SFT with Unsloth QLoRA on the verified examples, then GGUF export.

Everything heavy is imported inside `train` so `--dry-run` works without a GPU or the
`train` extra: it validates the config and the dataset and reports what a run would do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from training.core.types import TrainingExample


class SftError(RuntimeError):
    pass


@dataclass(frozen=True)
class Plan:
    base_model: str
    dataset: Path
    output_dir: Path
    examples: int
    with_tool_calls: int
    max_seq_length: int
    epochs: int
    gguf_quant: str


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    for key in ("base_model", "dataset", "output_dir", "max_seq_length", "lora", "train", "export"):
        if key not in cfg:
            raise SftError(f"{path}: missing `{key}`")
    return cfg


def load_examples(path: Path) -> list[TrainingExample]:
    if not path.exists():
        raise SftError(f"dataset {path} not found; run `data export` then `data verify`")
    rows = [
        TrainingExample.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SftError(f"dataset {path} is empty")
    return rows


def to_conversation(ex: TrainingExample) -> list[dict[str, Any]]:
    """Prompt messages plus the reply, in the chat-template shape Unsloth expects."""
    turns = [m.model_dump(exclude_none=True, exclude_defaults=True) for m in ex.messages]
    turns.append(ex.response.model_dump(exclude_none=True, exclude_defaults=True))
    return turns


def plan(config_path: Path) -> Plan:
    cfg = load_config(config_path)
    examples = load_examples(Path(cfg["dataset"]))
    return Plan(
        base_model=cfg["base_model"],
        dataset=Path(cfg["dataset"]),
        output_dir=Path(cfg["output_dir"]),
        examples=len(examples),
        with_tool_calls=sum(1 for e in examples if e.response.tool_calls),
        max_seq_length=int(cfg["max_seq_length"]),
        epochs=int(cfg["train"]["epochs"]),
        gguf_quant=str(cfg["export"]["gguf_quant"]),
    )


def train(config_path: Path) -> Path:
    """Runs SFT and returns the exported GGUF path. Needs the `train` extra and a GPU."""
    cfg = load_config(config_path)
    examples = load_examples(Path(cfg["dataset"]))
    try:
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel

        from datasets import Dataset
    except ImportError as e:
        raise SftError("install the `train` extra: uv sync --extra train") from e

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=int(cfg["max_seq_length"]),
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
    )
    lora = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        random_state=int(cfg["train"].get("seed", 3407)),
    )
    texts = [tokenizer.apply_chat_template(to_conversation(ex), tokenize=False) for ex in examples]
    dataset = Dataset.from_dict({"text": texts})
    tr = cfg["train"]
    out = Path(cfg["output_dir"])
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(out),
            num_train_epochs=float(tr["epochs"]),
            per_device_train_batch_size=int(tr["per_device_batch_size"]),
            gradient_accumulation_steps=int(tr["gradient_accumulation"]),
            learning_rate=float(tr["learning_rate"]),
            warmup_ratio=float(tr["warmup_ratio"]),
            logging_steps=int(tr.get("logging_steps", 5)),
            seed=int(tr.get("seed", 3407)),
            report_to="tensorboard",
            dataset_text_field="text",
            max_length=int(cfg["max_seq_length"]),
        ),
    )
    trainer.train()
    gguf_dir = out / "gguf"
    model.save_pretrained_gguf(
        str(gguf_dir), tokenizer, quantization_method=cfg["export"]["gguf_quant"]
    )
    ggufs = sorted(gguf_dir.glob("*.gguf"))
    if not ggufs:
        raise SftError(f"no GGUF written under {gguf_dir}")
    return ggufs[-1]
