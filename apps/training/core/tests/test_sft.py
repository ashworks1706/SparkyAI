from pathlib import Path

import pytest
from training.core.types import Message, SftError, TrainingExample
from training.posttrain import sft


def test_plan_reads_config_and_counts_examples(tmp_path: Path):
    ds = tmp_path / "sft.jsonl"
    ex = TrainingExample(
        id="1",
        messages=[Message(role="system", content="s"), Message(role="user", content="q")],
        response=Message(
            role="assistant", content="", tool_calls=[{"id": "c", "name": "t", "arguments": {}}]
        ),
    )
    ds.write_text(ex.model_dump_json() + "\n")
    cfg = tmp_path / "sft.yaml"
    cfg.write_text(
        f"base_model: unsloth/Qwen3-4B\ndataset: {ds}\noutput_dir: {tmp_path / 'out'}\n"
        "max_seq_length: 512\nload_in_4bit: true\n"
        "lora: {r: 8, alpha: 16, dropout: 0.0, target_modules: [q_proj]}\n"
        "train: {epochs: 1, per_device_batch_size: 1, gradient_accumulation: 1, "
        "learning_rate: 1e-4, warmup_ratio: 0.0, logging_steps: 5, seed: 1}\n"
        "export: {gguf_quant: q4_k_m}\n"
    )
    p = sft.plan(cfg)
    assert p.examples == 1 and p.with_tool_calls == 1 and p.gguf_quant == "q4_k_m"
    assert sft.to_conversation(ex)[-1]["role"] == "assistant"


def test_incomplete_or_unknown_config_keys_fail(tmp_path: Path):
    cfg = tmp_path / "sft.yaml"
    cfg.write_text("base_model: m\ndataset: missing.jsonl\noutput_dir: o\nmax_seq_length: 1\n")
    with pytest.raises(SftError):
        sft.plan(cfg)
    cfg.write_text(
        "base_model: m\ndataset: d\noutput_dir: o\nmax_seq_length: 1\nload_in_4bit: true\n"
        "lora: {r: 8, alpha: 16, dropout: 0.0, target_modules: [q_proj]}\n"
        "train: {epochs: 1, per_device_batch_size: 1, gradient_accumulation: 1, "
        "learning_rate: 1e-4, warmup_ratio: 0.0, logging_steps: 5, seed: 1}\n"
        "export: {gguf_quant: q4_k_m}\nwandb: true\n"
    )
    with pytest.raises(SftError):
        sft.plan(cfg)
