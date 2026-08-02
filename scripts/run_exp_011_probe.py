#!/usr/bin/env python3
"""Run a two-update, train-only Experiment 011 upper-layer probe."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-011-deberta-upper-layer-refinement.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "data" / "manifest.json"
PARENT_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-010" / "result.json"
PARENT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-010" / "selected-model"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011-probe"
RESULT_PATH = RESULT_DIR / "result.json"
PROBE_ROWS = 64


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stop(message: str) -> None:
    raise SystemExit(f"Experiment 011 probe stopped: {message}")


def freeze_for_upper_layer_refinement(model, config: dict[str, object]) -> tuple[int, int, list[str]]:
    freeze_config = config["freeze"]
    start = int(freeze_config["encoder_layer_start"])
    end = int(freeze_config["encoder_layer_end"])
    trainable_prefixes = [f"deberta.encoder.layer.{index}." for index in range(start, end + 1)]
    if bool(freeze_config["train_pooler"]):
        # DebertaV2ForSequenceClassification keeps ContextPooler at the model
        # root (`pooler.*`), unlike the encoder layers under `deberta.*`.
        trainable_prefixes.append("pooler.")
    if bool(freeze_config["train_classifier"]):
        trainable_prefixes.append("classifier.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    missing = [prefix for prefix in trainable_prefixes if not any(name.startswith(prefix) for name, _ in model.named_parameters())]
    return trainable, frozen, missing


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    stop("use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    stop("PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    stop("result already exists; refusing to overwrite.")
if not PARENT_MODEL_DIR.is_dir() or not PARENT_RESULT_PATH.is_file():
    stop("finalized Experiment 010 model is missing.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
parent_result = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
if parent_result.get("test_rows_loaded") != 0 or parent_result.get("test_evaluated") is not False:
    stop("parent result reports test access.")
if data_manifest.get("test_rows_loaded") != 0:
    stop("train manifest reports test access.")
train_path = Path(data_manifest["train_path"])
if sha256(train_path) != data_manifest["train_sha256"]:
    stop("expanded train checksum mismatch.")
with train_path.open(encoding="utf-8") as handle:
    source_rows = [json.loads(line) for line in handle]
if len(source_rows) != data_manifest["train_rows"]:
    stop("expanded train row count mismatch.")

tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL_DIR, local_files_only=True, use_fast=False)
unique_rows = {str(row["text"]): row for row in source_rows}.values()
probe_source = sorted(
    unique_rows,
    key=lambda row: len(tokenizer(str(row["text"]), add_special_tokens=True)["input_ids"]),
    reverse=True,
)[:PROBE_ROWS]
encoded_records: list[dict[str, object]] = []
observed_lengths: list[int] = []
for record in probe_source:
    encoded = tokenizer(
        str(record["text"]),
        max_length=config["tokenization"]["max_length"],
        truncation=config["tokenization"]["truncation"],
    )
    observed_lengths.append(len(encoded["input_ids"]))
    encoded["labels"] = int(record["label"])
    encoded_records.append(encoded)

print("Open Model Training Lab — Experiment 011 upper-layer refinement probe")
print(f"parent_model: {PARENT_MODEL_DIR}")
print(f"parent_weight_sha256: {parent_result['weight_sha256']}")
print(f"train_rows: {len(source_rows)}")
print(f"probe_rows: {len(encoded_records)}")
print(f"probe_steps: {config['optimization']['probe_steps']}")
print(f"batch_size: {config['batching']['train_batch_size']}")
print(f"gradient_accumulation_steps: {config['batching']['gradient_accumulation_steps']}")
print(f"learning_rate: {config['optimization']['learning_rate']}")
print(f"trainable_encoder_layers: {config['freeze']['encoder_layer_start']}-{config['freeze']['encoder_layer_end']}")
print(f"max_observed_probe_tokens: {max(observed_lengths)}")
print("validation_rows_loaded: 0")
print("test_rows_loaded: 0")
print("loading_parent_model_and_training: True")

model = AutoModelForSequenceClassification.from_pretrained(PARENT_MODEL_DIR, local_files_only=True)
if config["force_float32"]:
    model.float()
trainable_parameters, frozen_parameters, missing_prefixes = freeze_for_upper_layer_refinement(model, config)
if missing_prefixes:
    stop(f"expected trainable parameter prefixes are missing: {missing_prefixes}")
if trainable_parameters <= 0 or frozen_parameters <= 0:
    stop("upper-layer freeze configuration did not produce both trainable and frozen parameters.")
print("model_parameter_dtype: float32")
print(f"trainable_parameters: {trainable_parameters}")
print(f"frozen_parameters: {frozen_parameters}")

arguments = TrainingArguments(
    output_dir=str(RESULT_DIR / "trainer-output"),
    per_device_train_batch_size=config["batching"]["train_batch_size"],
    gradient_accumulation_steps=config["batching"]["gradient_accumulation_steps"],
    max_steps=config["optimization"]["probe_steps"],
    learning_rate=config["optimization"]["learning_rate"],
    max_grad_norm=config["optimization"]["max_grad_norm"],
    weight_decay=config["optimization"]["weight_decay"],
    optim=config["optimization"]["optimizer"],
    logging_strategy="steps",
    logging_steps=1,
    logging_nan_inf_filter=False,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    seed=config["optimization"]["seed"],
    data_seed=config["optimization"]["seed"],
    dataloader_pin_memory=False,
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=ListDataset(encoded_records),
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
    processing_class=tokenizer,
)
trainer.model_accepts_loss_kwargs = False
started = time.monotonic()
train_output = trainer.train()
torch.mps.synchronize()
training_loss = float(train_output.training_loss)
elapsed_seconds = time.monotonic() - started
peak_memory_gib = torch.mps.driver_allocated_memory() / (1024**3)
if not math.isfinite(training_loss):
    stop("training loss is non-finite.")

result = {
    "experiment": config["experiment"] + "-probe",
    "parent_experiment": config["model"]["parent_experiment"],
    "parent_weight_sha256": parent_result["weight_sha256"],
    "train_rows": len(source_rows),
    "probe_rows": len(encoded_records),
    "probe_steps": config["optimization"]["probe_steps"],
    "batch_size": config["batching"]["train_batch_size"],
    "gradient_accumulation_steps": config["batching"]["gradient_accumulation_steps"],
    "learning_rate": config["optimization"]["learning_rate"],
    "trainable_encoder_layers": [
        config["freeze"]["encoder_layer_start"],
        config["freeze"]["encoder_layer_end"],
    ],
    "trainable_parameters": trainable_parameters,
    "frozen_parameters": frozen_parameters,
    "training_loss": training_loss,
    "max_observed_probe_tokens": max(observed_lengths),
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
    "weights_saved": False,
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": peak_memory_gib,
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"training_loss: {training_loss:.6f}")
print(f"elapsed_seconds: {elapsed_seconds:.3f}")
print(f"peak_driver_memory_gib: {peak_memory_gib:.3f}")
print("weights_saved: False")
print(f"result: {RESULT_PATH}")
print("exp_011_probe_ok: True")
