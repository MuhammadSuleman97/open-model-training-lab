#!/usr/bin/env python3
"""Train Exp014 with broader train-only hard-negative coverage."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from hard_negative_training_common import (
    HardNegativeCollator,
    HardNegativeTrainer,
    StopOnNonFiniteLoss,
    compute_metrics,
    configure_trainable_parameters,
    encode_records,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-014-deberta-broader-hard-negative.json"
CANONICAL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-014" / "data" / "manifest.json"
PARENT_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "result.json"
PARENT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "selected-model"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-014"
TRAINER_OUTPUT_DIR = RESULT_DIR / "trainer-output"
RESULT_PATH = RESULT_DIR / "training_result.json"


def stop(message: str) -> None:
    raise SystemExit(f"Experiment 014 stopped: {message}")


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    stop("use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    stop("PyTorch MPS is unavailable; run this from the user's normal M2 Max Terminal.")
if RESULT_PATH.exists():
    stop("training result already exists; refusing to overwrite.")
if TRAINER_OUTPUT_DIR.exists() and any(TRAINER_OUTPUT_DIR.iterdir()):
    stop("trainer output already exists; refusing to overwrite.")
if not PARENT_MODEL_DIR.is_dir() or not PARENT_RESULT_PATH.is_file():
    stop("Exp011 selected model is missing.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
canonical_manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
parent_result = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
if data_manifest.get("validation_rows_loaded") != 0 or data_manifest.get("test_rows_loaded") != 0:
    stop("Exp014 data manifest reports held-out split access.")
if data_manifest.get("labels_rewritten") is not False:
    stop("Exp014 data manifest does not preserve original labels.")
if parent_result.get("test_rows_loaded") != 0 or parent_result.get("test_evaluated") is not False:
    stop("parent result reports test access.")

train_file = Path(data_manifest["files"]["train.jsonl"]["path"])
train_records = read_jsonl(train_file, data_manifest["files"]["train.jsonl"]["sha256"])
valid_file = canonical_manifest["files"]["valid"]
valid_records = read_jsonl(Path(valid_file["path"]), valid_file["sha256"])
if len(train_records) != int(config["data"]["source_train_rows"]):
    stop("train row count mismatch.")
if len(valid_records) != int(config["data"]["validation_rows"]):
    stop("validation row count mismatch.")
active_rows = sum(bool(record["hard_negative_active"]) for record in train_records)
if active_rows != int(data_manifest["hard_negative_rows"]):
    stop("hard-negative row count mismatch.")

labels = canonical_manifest["label_order"]
id2label = {index: label for index, label in enumerate(labels)}
label2id = {label: index for index, label in id2label.items()}
tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL_DIR, local_files_only=True, use_fast=False)
train_dataset = encode_records(
    train_records,
    tokenizer,
    int(config["tokenization"]["max_length"]),
    label2id,
    include_rivals=True,
)
validation_dataset = encode_records(
    valid_records,
    tokenizer,
    int(config["tokenization"]["max_length"]),
    label2id,
    include_rivals=False,
)

print("Open Model Training Lab — Experiment 014 broader hard-negative refinement")
print(f"parent_model: {PARENT_MODEL_DIR}")
print(f"parent_weight_sha256: {parent_result['weight_sha256']}")
print(f"train_rows: {len(train_dataset)}")
print(f"hard_negative_rows: {active_rows}")
print(f"validation_rows: {len(validation_dataset)}")
print("test_rows_loaded: 0")
print(f"labels: {len(labels)}")
print("epochs: 1")
print(f"learning_rate: {config['optimization']['learning_rate']}")
print(f"hard_negative_margin: {config['hard_negative_loss']['margin']}")
print(f"hard_negative_weight: {config['hard_negative_loss']['weight']}")
print(f"train_batch_size: {config['batching']['train_batch_size']}")
print(f"gradient_accumulation_steps: {config['batching']['gradient_accumulation_steps']}")
print(
    f"trainable_encoder_layers: {config['freeze']['encoder_layer_start']}-"
    f"{config['freeze']['encoder_layer_end']}"
)
print("selection_data: validation_only")
print("loading_parent_model_and_training: True")

model = AutoModelForSequenceClassification.from_pretrained(
    PARENT_MODEL_DIR,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
    num_labels=len(labels),
)
if config["force_float32"]:
    model.float()
trainable_parameters, frozen_parameters, missing_prefixes = configure_trainable_parameters(model, config)
if missing_prefixes:
    stop(f"expected trainable parameter prefixes are missing: {missing_prefixes}")
if trainable_parameters <= 0 or frozen_parameters <= 0:
    stop("freeze configuration did not produce both trainable and frozen parameters.")
print("model_parameter_dtype: float32")
print(f"trainable_parameters: {trainable_parameters}")
print(f"frozen_parameters: {frozen_parameters}")

arguments = TrainingArguments(
    output_dir=str(TRAINER_OUTPUT_DIR),
    per_device_train_batch_size=int(config["batching"]["train_batch_size"]),
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=int(config["batching"]["gradient_accumulation_steps"]),
    num_train_epochs=1,
    learning_rate=float(config["optimization"]["learning_rate"]),
    warmup_steps=int(config["optimization"]["warmup_steps"]),
    weight_decay=float(config["optimization"]["weight_decay"]),
    max_grad_norm=float(config["optimization"]["max_grad_norm"]),
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
    seed=int(config["optimization"]["seed"]),
    data_seed=int(config["optimization"]["seed"]),
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)
trainer = HardNegativeTrainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=HardNegativeCollator(tokenizer),
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[StopOnNonFiniteLoss()],
    hard_negative_margin=float(config["hard_negative_loss"]["margin"]),
    hard_negative_weight=float(config["hard_negative_loss"]["weight"]),
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
if len(validation_history) != 1:
    stop("expected one validation measurement in the completed training log.")
if not math.isfinite(float(train_output.training_loss)):
    stop("final train loss is non-finite.")
if not all(
    math.isfinite(float(entry[key]))
    for entry in validation_history
    for key in ("eval_loss", "eval_accuracy", "eval_macro_f1")
):
    stop("validation metric is non-finite.")
if not trainer.training_components:
    stop("no training loss components were recorded.")

best_child = validation_history[0]
parent_validation_accuracy = float(parent_result["selected_validation"]["accuracy"])
component_means = {
    key: float(np.mean([float(item[key]) for item in trainer.training_components]))
    for key in ("loss", "cross_entropy", "pairwise_loss")
}
active_rows_seen = sum(int(item["active_rows"]) for item in trainer.training_components)
result = {
    "experiment": config["experiment"],
    "parent_experiment": config["model"]["parent_experiment"],
    "parent_weight_sha256": parent_result["weight_sha256"],
    "train_rows": len(train_dataset),
    "hard_negative_rows": active_rows,
    "hard_negative_rows_seen_in_batches": active_rows_seen,
    "hard_negative_margin": float(config["hard_negative_loss"]["margin"]),
    "hard_negative_weight": float(config["hard_negative_loss"]["weight"]),
    "validation_rows": len(validation_dataset),
    "test_rows_loaded": 0,
    "test_evaluated": False,
    "epochs": 1,
    "learning_rate": float(config["optimization"]["learning_rate"]),
    "trainable_encoder_layers": [config["freeze"]["encoder_layer_start"], config["freeze"]["encoder_layer_end"]],
    "trainable_parameters": trainable_parameters,
    "frozen_parameters": frozen_parameters,
    "train_loss": float(train_output.training_loss),
    "training_component_means": component_means,
    "validation_history": validation_history,
    "best_child_epoch": float(best_child["epoch"]),
    "best_child_validation_accuracy": float(best_child["eval_accuracy"]),
    "best_child_validation_macro_f1": float(best_child["eval_macro_f1"]),
    "best_child_checkpoint": str(TRAINER_OUTPUT_DIR / f"checkpoint-{int(best_child['step'])}"),
    "parent_validation_accuracy": parent_validation_accuracy,
    "strict_validation_improvement": float(best_child["eval_accuracy"] - parent_validation_accuracy),
    "train_data_sha256": data_manifest["files"]["train.jsonl"]["sha256"],
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": torch.mps.driver_allocated_memory() / (1024**3),
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"train_loss: {result['train_loss']:.6f}")
print(f"training_cross_entropy_mean: {component_means['cross_entropy']:.6f}")
print(f"training_pairwise_loss_mean: {component_means['pairwise_loss']:.6f}")
print(f"hard_negative_rows_seen_in_batches: {active_rows_seen}")
print(f"validation_loss: {best_child['eval_loss']:.6f}")
print(f"validation_accuracy: {best_child['eval_accuracy']:.6f}")
print(f"validation_macro_f1: {best_child['eval_macro_f1']:.6f}")
print(f"parent_validation_accuracy: {parent_validation_accuracy:.6f}")
print(f"strict_validation_improvement: {result['strict_validation_improvement']:+.6f}")
print(f"peak_driver_memory_gib: {result['peak_driver_memory_gib']:.3f}")
print(f"elapsed_seconds: {elapsed_seconds:.3f}")
print("test_rows_loaded: 0")
print("test_evaluated: False")
print(f"result: {RESULT_PATH}")
print("exp_014_training_ok: True")
