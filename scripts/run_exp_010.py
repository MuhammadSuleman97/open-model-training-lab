#!/usr/bin/env python3
"""Train Experiment 010 on targeted training rows with validation-only selection."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-010-deberta-targeted-oversampling-full.json"
CANONICAL_DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
EXP010_DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-010" / "data" / "manifest.json"
PARENT_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "result.json"
PARENT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "selected-model"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-010"
TRAINER_OUTPUT_DIR = RESULT_DIR / "trainer-output"
RESULT_PATH = RESULT_DIR / "training_result.json"


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


class StopOnNonFiniteLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, state, control, kwargs
        for name, value in (logs or {}).items():
            if "loss" in name and isinstance(value, (float, int)) and not math.isfinite(float(value)):
                raise RuntimeError(f"Non-finite training metric: {name}={value}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, expected_hash: str) -> list[dict[str, object]]:
    if sha256(path) != expected_hash:
        raise SystemExit(f"Experiment 010 stopped: checksum mismatch for {path.name}.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def encode_records(records: list[dict[str, object]], tokenizer, max_length: int) -> ListDataset:
    encoded_records: list[dict[str, object]] = []
    for record in records:
        encoded = tokenizer(
            str(record["text"]),
            max_length=max_length,
            truncation=True,
        )
        encoded["labels"] = int(record["label"])
        encoded_records.append(encoded)
    return ListDataset(encoded_records)


def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    predicted_ids = np.asarray(prediction.predictions).argmax(axis=-1)
    expected_ids = np.asarray(prediction.label_ids)
    return {
        "accuracy": float(accuracy_score(expected_ids, predicted_ids)),
        "macro_f1": float(f1_score(expected_ids, predicted_ids, average="macro", zero_division=0)),
    }


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Experiment 010 stopped: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 010 stopped: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 010 stopped: training result already exists.")
if TRAINER_OUTPUT_DIR.exists() and any(TRAINER_OUTPUT_DIR.iterdir()):
    raise SystemExit("Experiment 010 stopped: trainer output already exists; refusing to overwrite.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
canonical_manifest = json.loads(CANONICAL_DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
exp_manifest = json.loads(EXP010_DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
parent_result = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
if exp_manifest.get("test_rows_loaded") != 0:
    raise SystemExit("Experiment 010 stopped: targeted manifest reports test access.")
labels = canonical_manifest["label_order"]
id2label = {index: label for index, label in enumerate(labels)}
label2id = {label: index for index, label in id2label.items()}
train_records = read_jsonl(Path(exp_manifest["train_path"]), exp_manifest["train_sha256"])
valid_file = canonical_manifest["files"]["valid"]
valid_records = read_jsonl(Path(valid_file["path"]), valid_file["sha256"])
if len(train_records) != config["data"]["train_rows"]:
    raise SystemExit("Experiment 010 stopped: targeted train row count mismatch.")
if len(valid_records) != config["data"]["validation_rows"]:
    raise SystemExit("Experiment 010 stopped: validation row count mismatch.")

tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL_DIR, local_files_only=True, use_fast=False)
train_dataset = encode_records(train_records, tokenizer, config["tokenization"]["max_length"])
validation_dataset = encode_records(valid_records, tokenizer, config["tokenization"]["max_length"])

print("Open Model Training Lab — Experiment 010 targeted DeBERTa continuation")
print(f"parent_model: {PARENT_MODEL_DIR}")
print(f"parent_weight_sha256: {parent_result['weight_sha256']}")
print(f"train_rows: {len(train_dataset)}")
print(f"validation_rows: {len(validation_dataset)}")
print("test_rows_loaded: 0")
print(f"labels: {len(labels)}")
print(f"epochs: {config['optimization']['epochs']}")
print(f"learning_rate: {config['optimization']['learning_rate']}")
print(f"train_batch_size: {config['batching']['train_batch_size']}")
print(f"gradient_accumulation_steps: {config['batching']['gradient_accumulation_steps']}")
print("selection_data: validation_only")
print("loading_parent_model_and_training: True")

model = AutoModelForSequenceClassification.from_pretrained(
    PARENT_MODEL_DIR,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
    num_labels=len(labels),
)
model.float()
print("model_parameter_dtype: float32")
arguments = TrainingArguments(
    output_dir=str(TRAINER_OUTPUT_DIR),
    per_device_train_batch_size=config["batching"]["train_batch_size"],
    per_device_eval_batch_size=config["batching"]["eval_batch_size"],
    gradient_accumulation_steps=config["batching"]["gradient_accumulation_steps"],
    num_train_epochs=config["optimization"]["epochs"],
    learning_rate=config["optimization"]["learning_rate"],
    warmup_steps=config["optimization"]["warmup_steps"],
    weight_decay=config["optimization"]["weight_decay"],
    max_grad_norm=config["optimization"]["max_grad_norm"],
    optim=config["optimization"]["optimizer"],
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_strategy="steps",
    logging_steps=25,
    logging_first_step=True,
    logging_nan_inf_filter=False,
    report_to="none",
    seed=config["optimization"]["seed"],
    data_seed=config["optimization"]["seed"],
    dataloader_pin_memory=False,
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[StopOnNonFiniteLoss()],
)
trainer.model_accepts_loss_kwargs = False

started = time.monotonic()
train_output = trainer.train()
torch.mps.synchronize()
elapsed_seconds = time.monotonic() - started
validation_history = [
    {
        "epoch": entry["epoch"],
        "step": int(entry.get("step", trainer.state.global_step)),
        "eval_loss": entry["eval_loss"],
        "eval_accuracy": entry["eval_accuracy"],
        "eval_macro_f1": entry["eval_macro_f1"],
    }
    for entry in trainer.state.log_history
    if "eval_accuracy" in entry
]
if len(validation_history) != config["optimization"]["epochs"]:
    raise SystemExit("Experiment 010 stopped: expected one validation measurement.")
if not math.isfinite(float(train_output.training_loss)):
    raise SystemExit("Experiment 010 stopped: final train loss is non-finite.")
if not all(
    math.isfinite(float(entry[key]))
    for entry in validation_history
    for key in ("eval_loss", "eval_accuracy", "eval_macro_f1")
):
    raise SystemExit("Experiment 010 stopped: validation metric is non-finite.")

best_child = max(validation_history, key=lambda entry: (entry["eval_accuracy"], -entry["epoch"]))
peak_memory_gib = torch.mps.driver_allocated_memory() / (1024**3)
result = {
    "experiment": config["experiment"],
    "parent_experiment": config["model"]["parent_experiment"],
    "parent_weight_sha256": parent_result["weight_sha256"],
    "train_rows": len(train_dataset),
    "validation_rows": len(validation_dataset),
    "test_rows_loaded": 0,
    "test_evaluated": False,
    "epochs": config["optimization"]["epochs"],
    "learning_rate": config["optimization"]["learning_rate"],
    "train_loss": float(train_output.training_loss),
    "validation_history": validation_history,
    "best_child_epoch": float(best_child["epoch"]),
    "best_child_validation_accuracy": float(best_child["eval_accuracy"]),
    "best_child_validation_macro_f1": float(best_child["eval_macro_f1"]),
    "best_child_checkpoint": str(TRAINER_OUTPUT_DIR / f"checkpoint-{int(best_child['step'])}"),
    "parent_validation_accuracy": exp_manifest["parent_validation_accuracy"],
    "strict_validation_improvement": float(best_child["eval_accuracy"] - exp_manifest["parent_validation_accuracy"]),
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": peak_memory_gib,
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"train_loss: {result['train_loss']:.6f}")
print(f"validation_accuracy: {result['best_child_validation_accuracy']:.6f}")
print(f"validation_macro_f1: {result['best_child_validation_macro_f1']:.6f}")
print(f"parent_validation_accuracy: {result['parent_validation_accuracy']:.6f}")
print(f"strict_validation_improvement: {result['strict_validation_improvement']:+.6f}")
print(f"peak_driver_memory_gib: {peak_memory_gib:.3f}")
print(f"elapsed_seconds: {elapsed_seconds:.3f}")
print("test_rows_loaded: 0")
print("test_evaluated: False")
print(f"result: {RESULT_PATH}")
print("exp_010_training_ok: True")
