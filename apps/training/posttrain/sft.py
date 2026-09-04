"""SFT with Unsloth QLoRA on the verified examples, then GGUF export.

Everything heavy is imported inside `train` so `--dry-run` works without a GPU or the
`train` extra: it validates the config and the dataset and reports what a run would do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from training.core.types import SftConfig, SftError, SftPlan, TrainingExample


def load_config(path: Path) -> SftConfig:
    if not path.exists():
        raise SftError(f"config {path} not found")
    try:
        return SftConfig.model_validate(yaml.safe_load(path.read_text()))
    except ValidationError as e:
        raise SftError(f"{path}: {e}") from e


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


def plan(config_path: Path) -> SftPlan:
    cfg = load_config(config_path)
    examples = load_examples(cfg.dataset)
    return SftPlan(
        base_model=cfg.base_model,
        dataset=cfg.dataset,
        output_dir=cfg.output_dir,
        examples=len(examples),
        with_tool_calls=sum(1 for e in examples if e.response.tool_calls),
        max_seq_length=cfg.max_seq_length,
        epochs=cfg.train.epochs,
        gguf_quant=cfg.export.gguf_quant,
    )


def train(config_path: Path) -> Path:
    """Runs SFT and returns the exported GGUF path. Needs the `train` extra and a GPU."""
    cfg = load_config(config_path)
    examples = load_examples(cfg.dataset)
    try:
        from trl import SFTConfig as TrlConfig
        from trl import SFTTrainer
        from unsloth import FastLanguageModel

        from datasets import Dataset
    except ImportError as e:
        raise SftError("install the `train` extra: uv sync --extra train") from e

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        random_state=cfg.train.seed,
    )
    texts = [tokenizer.apply_chat_template(to_conversation(ex), tokenize=False) for ex in examples]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_dict({"text": texts}),
        args=TrlConfig(
            output_dir=str(cfg.output_dir),
            num_train_epochs=float(cfg.train.epochs),
            per_device_train_batch_size=cfg.train.per_device_batch_size,
            gradient_accumulation_steps=cfg.train.gradient_accumulation,
            learning_rate=cfg.train.learning_rate,
            warmup_ratio=cfg.train.warmup_ratio,
            logging_steps=cfg.train.logging_steps,
            seed=cfg.train.seed,
            report_to="tensorboard",
            dataset_text_field="text",
            max_length=cfg.max_seq_length,
        ),
    )
    trainer.train()
    gguf_dir = cfg.output_dir / "gguf"
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=cfg.export.gguf_quant)
    ggufs = sorted(gguf_dir.glob("*.gguf"))
    if not ggufs:
        raise SftError(f"no GGUF written under {gguf_dir}")
    return ggufs[-1]
