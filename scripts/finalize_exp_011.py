#!/usr/bin/env python3
"""Validate, package and round-trip Experiment 011 without test access."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-011-deberta-upper-layer-refinement.json"
CANONICAL_DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
EXP011_DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "data" / "manifest.json"
TRAINING_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "training_result.json"
RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "result.json"
SELECTED_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "selected-model"
PARENT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-010" / "selected-model"


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
    raise SystemExit(f"Experiment 011 finalization stopped: {message}")


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    stop("use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    stop("PyTorch MPS is unavailable.")
if not TRAINING_RESULT_PATH.is_file():
    stop("training result is missing.")
if RESULT_PATH.exists() or SELECTED_MODEL_DIR.exists():
    stop("finalization artifacts already exist; refusing to overwrite.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
canonical_manifest = json.loads(CANONICAL_DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
exp_manifest = json.loads(EXP011_DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
training_result = json.loads(TRAINING_RESULT_PATH.read_text(encoding="utf-8"))
if training_result.get("test_rows_loaded") != 0 or training_result.get("test_evaluated") is not False:
    stop("training result reports test access.")
best_checkpoint = Path(training_result["best_child_checkpoint"])
if not (best_checkpoint / "model.safetensors").is_file():
    stop("best checkpoint weights are missing.")

valid_file = canonical_manifest["files"]["valid"]
valid_path = Path(valid_file["path"])
if sha256(valid_path) != valid_file["sha256"]:
    stop("validation checksum mismatch.")
with valid_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]
if len(source_records) != config["data"]["validation_rows"]:
    stop("validation row count mismatch.")

# The checkpoint and parent use the same tokenizer. Loading the parent copy
# keeps finalization robust if Trainer did not copy tokenizer files into the
# checkpoint directory.
tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL_DIR, local_files_only=True, use_fast=False)
encoded_records: list[dict[str, object]] = []
for record in source_records:
    encoded = tokenizer(str(record["text"]), max_length=config["tokenization"]["max_length"], truncation=True)
    encoded["labels"] = int(record["label"])
    encoded_records.append(encoded)
loader = DataLoader(
    ListDataset(encoded_records),
    batch_size=32,
    shuffle=False,
    num_workers=0,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
)

print("Open Model Training Lab — Experiment 011 validation-only finalization")
print(f"best_checkpoint: {best_checkpoint}")
print(f"recorded_validation_accuracy: {training_result['best_child_validation_accuracy']:.6f}")
print("selection_metric: validation_accuracy")
print(f"validation_rows: {len(source_records)}")
print("test_rows_loaded: 0")
print("loading_child_checkpoint: True")

started = time.monotonic()
model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint, local_files_only=True)
model.to(torch.device("mps"))
model.float()
model.eval()
predictions: list[int] = []
expected: list[int] = []
loss_sum = 0.0
rows = 0
with torch.no_grad():
    for batch in loader:
        batch = {name: tensor.to("mps") for name, tensor in batch.items()}
        output = model(**batch)
        batch_size = int(batch["labels"].shape[0])
        loss_sum += float(output.loss.detach().cpu()) * batch_size
        rows += batch_size
        predictions.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
        expected.extend(batch["labels"].detach().cpu().tolist())
torch.mps.synchronize()
selected_metrics = {
    "loss": loss_sum / rows,
    "accuracy": float(accuracy_score(expected, predictions)),
    "macro_f1": float(f1_score(expected, predictions, average="macro", zero_division=0)),
}
if not all(math.isfinite(value) for value in selected_metrics.values()):
    stop("validation metric is non-finite.")
if not math.isclose(
    selected_metrics["accuracy"],
    float(training_result["best_child_validation_accuracy"]),
    rel_tol=0.0,
    abs_tol=1e-12,
):
    stop("checkpoint validation accuracy does not match the training result.")

model.save_pretrained(SELECTED_MODEL_DIR)
tokenizer.save_pretrained(SELECTED_MODEL_DIR)
del model
torch.mps.empty_cache()
print("roundtrip_reload_check: True")
roundtrip_model = AutoModelForSequenceClassification.from_pretrained(SELECTED_MODEL_DIR, local_files_only=True)
roundtrip_model.to(torch.device("mps"))
roundtrip_model.float()
roundtrip_model.eval()
roundtrip_predictions: list[int] = []
roundtrip_expected: list[int] = []
roundtrip_loss_sum = 0.0
roundtrip_rows = 0
with torch.no_grad():
    for batch in loader:
        batch = {name: tensor.to("mps") for name, tensor in batch.items()}
        output = roundtrip_model(**batch)
        batch_size = int(batch["labels"].shape[0])
        roundtrip_loss_sum += float(output.loss.detach().cpu()) * batch_size
        roundtrip_rows += batch_size
        roundtrip_predictions.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
        roundtrip_expected.extend(batch["labels"].detach().cpu().tolist())
torch.mps.synchronize()
roundtrip_metrics = {
    "loss": roundtrip_loss_sum / roundtrip_rows,
    "accuracy": float(accuracy_score(roundtrip_expected, roundtrip_predictions)),
    "macro_f1": float(f1_score(roundtrip_expected, roundtrip_predictions, average="macro", zero_division=0)),
}
if roundtrip_predictions != predictions or any(
    not math.isclose(roundtrip_metrics[key], selected_metrics[key], rel_tol=0.0, abs_tol=1e-9)
    for key in selected_metrics
):
    stop("save/reload validation predictions or metrics changed.")

weight_path = SELECTED_MODEL_DIR / "model.safetensors"
if not weight_path.is_file():
    stop("selected model weights are missing after save.")
parent_validation_accuracy = float(training_result["parent_validation_accuracy"])
result = {
    "experiment": config["experiment"],
    "parent_experiment": config["model"]["parent_experiment"],
    "parent_weight_sha256": training_result["parent_weight_sha256"],
    "train_rows": exp_manifest["train_rows"],
    "validation_rows": len(source_records),
    "test_rows_loaded": 0,
    "test_evaluated": False,
    "selected_checkpoint": str(best_checkpoint),
    "selected_validation": selected_metrics,
    "parent_validation_accuracy": parent_validation_accuracy,
    "strict_validation_improvement": selected_metrics["accuracy"] - parent_validation_accuracy,
    "trainable_encoder_layers": training_result["trainable_encoder_layers"],
    "trainable_parameters": training_result["trainable_parameters"],
    "frozen_parameters": training_result["frozen_parameters"],
    "roundtrip_reload_verified": True,
    "selected_model_path": str(SELECTED_MODEL_DIR),
    "weight_file": weight_path.name,
    "weight_sha256": sha256(weight_path),
    "train_data_sha256": exp_manifest["train_sha256"],
    "elapsed_seconds": time.monotonic() - started,
    "peak_driver_memory_gib": torch.mps.driver_allocated_memory() / (1024**3),
}
RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"selected_validation_loss: {selected_metrics['loss']:.6f}")
print(f"selected_validation_accuracy: {selected_metrics['accuracy']:.6f}")
print(f"selected_validation_macro_f1: {selected_metrics['macro_f1']:.6f}")
print(f"parent_validation_accuracy: {parent_validation_accuracy:.6f}")
print(f"strict_validation_improvement: {result['strict_validation_improvement']:+.6f}")
print("roundtrip_reload_verified: True")
print("test_rows_loaded: 0")
print("test_evaluated: False")
print(f"selected_model: {SELECTED_MODEL_DIR}")
print(f"weight_sha256: {result['weight_sha256']}")
print(f"result: {RESULT_PATH}")
print("exp_011_finalization_ok: True")
