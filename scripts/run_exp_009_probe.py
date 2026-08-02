#!/usr/bin/env python3
"""Run a two-update DeBERTa-v3-large MPS training pipeline probe."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = Path(
    os.environ.get(
        "OPEN_MODEL_TRAINING_CONFIG",
        str(PROJECT_ROOT / "configs" / "exp-009-deberta-v3-large-classifier.json"),
    )
)
MODEL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009" / "model_manifest.json"
DATA_MANIFEST_PATH = Path(
    os.environ.get(
        "OPEN_MODEL_TRAINING_DATA_MANIFEST",
        str(PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"),
    )
)
RESULT_DIR = Path(
    os.environ.get(
        "OPEN_MODEL_TRAINING_RESULT_DIR",
        str(PROJECT_ROOT / "artifacts" / "encoder" / "exp-009-probe"),
    )
)
RESULT_PATH = RESULT_DIR / "result.json"
PROBE_ROWS = 64
PROBE_STEPS = 2


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


def read_jsonl(path: Path, expected_hash: str) -> list[dict[str, object]]:
    if sha256(path) != expected_hash:
        raise SystemExit(f"Experiment 009 probe failed: checksum mismatch for {path.name}.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Experiment 009 probe failed: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 009 probe failed: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 009 probe stopped: result already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
model_path = Path(model_manifest["snapshot_path"])
weight_path = model_path / model_manifest["weight_file"]
if sha256(weight_path) != model_manifest["weight_sha256"]:
    raise SystemExit("Experiment 009 probe failed: model checksum mismatch.")
labels = data_manifest["label_order"]
id2label = {index: label for index, label in enumerate(labels)}
label2id = {label: index for index, label in id2label.items()}
train_file = data_manifest["files"]["train"]

torch.manual_seed(config["optimization"]["seed"])
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    use_fast=False,
)
source_records = read_jsonl(Path(train_file["path"]), train_file["sha256"])
ranked_records = sorted(
    source_records,
    key=lambda record: len(
        tokenizer(str(record["text"]), add_special_tokens=True)["input_ids"]
    ),
    reverse=True,
)
probe_source = ranked_records[:PROBE_ROWS]
encoded_records: list[dict[str, object]] = []
observed_lengths: list[int] = []
for record in probe_source:
    encoded = tokenizer(
        str(record["text"]),
        max_length=config["tokenization"]["max_length"],
        truncation=True,
    )
    observed_lengths.append(len(encoded["input_ids"]))
    encoded["labels"] = int(record["label"])
    encoded_records.append(encoded)

print("Open Model Training Lab — Experiment 009 MPS training probe")
print(f"model_id: {model_manifest['model_id']}")
print(f"model_revision: {model_manifest['resolved_revision']}")
print("device: mps")
print(f"probe_rows: {PROBE_ROWS}")
print(f"probe_steps: {PROBE_STEPS}")
print(f"batch_size: {config['batching']['train_batch_size']}")
print(f"gradient_accumulation_steps: {config['batching']['gradient_accumulation_steps']}")
print("trainer_handles_gradient_accumulation_normalization: True")
print(f"max_observed_probe_tokens: {max(observed_lengths)}")
print(f"max_length: {config['tokenization']['max_length']}")
print("validation_rows_loaded: 0")
print("test_rows_loaded: 0")
print("loading_model_and_training: True")

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
    num_labels=len(labels),
)
if config.get("force_float32", False):
    model.float()
    print("model_parameter_dtype: float32")
frozen_parameter_prefixes = config.get("frozen_parameter_prefixes", [])
frozen_parameter_names: list[str] = []
for name, parameter in model.named_parameters():
    if any(name.startswith(prefix) for prefix in frozen_parameter_prefixes):
        parameter.requires_grad = False
        frozen_parameter_names.append(name)
print(f"frozen_parameter_count: {len(frozen_parameter_names)}")
print(f"frozen_parameter_prefixes: {frozen_parameter_prefixes}")
arguments = TrainingArguments(
    output_dir=str(RESULT_DIR / "trainer-output"),
    per_device_train_batch_size=config["batching"]["train_batch_size"],
    gradient_accumulation_steps=config["batching"]["gradient_accumulation_steps"],
    max_steps=PROBE_STEPS,
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
elapsed_seconds = time.monotonic() - started
training_loss = float(train_output.training_loss)
peak_memory_gib = torch.mps.driver_allocated_memory() / (1024**3)
if not math.isfinite(training_loss):
    raise SystemExit("Experiment 009 probe failed: training loss is non-finite.")

result = {
    "experiment": config["experiment"] + "-probe",
    "model_id": model_manifest["model_id"],
    "model_revision": model_manifest["resolved_revision"],
    "device": "mps",
    "probe_rows": PROBE_ROWS,
    "probe_steps": PROBE_STEPS,
    "batch_size": config["batching"]["train_batch_size"],
    "gradient_accumulation_steps": config["batching"]["gradient_accumulation_steps"],
    "trainer_handles_gradient_accumulation_normalization": True,
    "max_observed_probe_tokens": max(observed_lengths),
    "training_loss": training_loss,
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": peak_memory_gib,
    "weights_saved": False,
    "frozen_parameter_prefixes": frozen_parameter_prefixes,
    "frozen_parameter_count": len(frozen_parameter_names),
    "force_float32": bool(config.get("force_float32", False)),
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"training_loss: {training_loss:.6f}")
print(f"elapsed_seconds: {elapsed_seconds:.3f}")
print(f"peak_driver_memory_gib: {peak_memory_gib:.3f}")
print("weights_saved: False")
print(f"result: {RESULT_PATH}")
print(f"{config.get('success_marker', 'exp_009_probe_ok')}: True")
