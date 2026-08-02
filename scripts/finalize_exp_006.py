#!/usr/bin/env python3
"""Validate and package Experiment 006's preselected best checkpoint."""

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
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-006-bert-large-classifier.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
TRAINER_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-006" / "trainer-output"
LATEST_CHECKPOINT = TRAINER_DIR / "checkpoint-1445"
SELECTED_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-006" / "selected-model"
RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-006" / "result.json"


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], list[int]]:
    model.to(device)
    model.eval()
    predictions: list[int] = []
    expected: list[int] = []
    loss_sum = 0.0
    row_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            output = model(**batch)
            batch_size = int(batch["labels"].shape[0])
            loss_sum += float(output.loss.detach().cpu()) * batch_size
            row_count += batch_size
            predictions.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
            expected.extend(batch["labels"].detach().cpu().tolist())
    torch.mps.synchronize()
    metrics = {
        "loss": loss_sum / row_count,
        "accuracy": float(accuracy_score(expected, predictions)),
        "macro_f1": float(
            f1_score(expected, predictions, average="macro", zero_division=0)
        ),
    }
    return metrics, predictions


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Experiment 006 finalization stopped: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 006 finalization stopped: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 006 finalization stopped: result already exists.")
if SELECTED_MODEL_DIR.exists():
    raise SystemExit(
        "Experiment 006 finalization stopped: selected-model directory already exists."
    )

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
trainer_state_path = LATEST_CHECKPOINT / "trainer_state.json"
trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
best_checkpoint = Path(trainer_state["best_model_checkpoint"])
best_accuracy = float(trainer_state["best_metric"])
if not best_checkpoint.is_dir():
    raise SystemExit("Experiment 006 finalization stopped: best checkpoint is missing.")

best_history = [
    entry
    for entry in trainer_state["log_history"]
    if entry.get("eval_accuracy") == best_accuracy
]
if not best_history:
    raise SystemExit("Experiment 006 finalization stopped: best metric has no history entry.")
# The predefined primary metric is accuracy. A tie keeps the first checkpoint.
selected_history = best_history[0]
selected_epoch = float(selected_history["epoch"])

validation_file = data_manifest["files"]["valid"]
validation_path = Path(validation_file["path"])
if sha256(validation_path) != validation_file["sha256"]:
    raise SystemExit("Experiment 006 finalization stopped: validation checksum mismatch.")
with validation_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]

print("Open Model Training Lab — Experiment 006 finalization")
print(f"selected_checkpoint: {best_checkpoint}")
print(f"selected_epoch: {selected_epoch:g}")
print(f"recorded_validation_accuracy: {best_accuracy:.6f}")
print("selection_metric: validation_accuracy")
print("tie_policy: first_checkpoint")
print(f"validation_rows: {len(source_records)}")
print("test_rows_loaded: 0")
print("loading_selected_checkpoint: True")

tokenizer = AutoTokenizer.from_pretrained(best_checkpoint, local_files_only=True)
encoded_records: list[dict[str, object]] = []
for record in source_records:
    encoded = tokenizer(
        str(record["text"]),
        max_length=config["tokenization"]["max_length"],
        truncation=True,
    )
    encoded["labels"] = int(record["label"])
    encoded_records.append(encoded)
loader = DataLoader(
    ListDataset(encoded_records),
    batch_size=config["batching"]["eval_batch_size"],
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
)

device = torch.device("mps")
started = time.monotonic()
model = AutoModelForSequenceClassification.from_pretrained(
    best_checkpoint,
    local_files_only=True,
)
selected_metrics, selected_predictions = evaluate(model, loader, device)
if not all(math.isfinite(value) for value in selected_metrics.values()):
    raise SystemExit("Experiment 006 finalization stopped: non-finite validation metric.")
if selected_metrics["accuracy"] != best_accuracy:
    raise SystemExit(
        "Experiment 006 finalization stopped: restored checkpoint accuracy does "
        f"not match training history ({selected_metrics['accuracy']:.6f} != "
        f"{best_accuracy:.6f})."
    )

model.save_pretrained(SELECTED_MODEL_DIR)
tokenizer.save_pretrained(SELECTED_MODEL_DIR)
del model
torch.mps.empty_cache()

print("roundtrip_reload_check: True")
roundtrip_model = AutoModelForSequenceClassification.from_pretrained(
    SELECTED_MODEL_DIR,
    local_files_only=True,
)
roundtrip_metrics, roundtrip_predictions = evaluate(roundtrip_model, loader, device)
if roundtrip_predictions != selected_predictions:
    raise SystemExit("Experiment 006 finalization stopped: roundtrip predictions changed.")
for metric_name, selected_value in selected_metrics.items():
    if not math.isclose(
        roundtrip_metrics[metric_name],
        selected_value,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise SystemExit(
            "Experiment 006 finalization stopped: roundtrip metric changed "
            f"for {metric_name}."
        )

weight_candidates = [
    path
    for path in (
        SELECTED_MODEL_DIR / "model.safetensors",
        SELECTED_MODEL_DIR / "pytorch_model.bin",
    )
    if path.is_file()
]
if len(weight_candidates) != 1:
    raise SystemExit(
        "Experiment 006 finalization stopped: expected one model weight artifact."
    )
weight_path = weight_candidates[0]
elapsed_seconds = time.monotonic() - started
validation_history = [
    {
        key: entry[key]
        for key in ("epoch", "eval_loss", "eval_accuracy", "eval_macro_f1")
    }
    for entry in trainer_state["log_history"]
    if "eval_accuracy" in entry
]
result = {
    "experiment": config["experiment"],
    "model_id": config["model"]["id"],
    "model_revision": config["model"]["revision"],
    "device": "mps",
    "train_rows": config["data"]["train_rows"],
    "validation_rows": len(source_records),
    "test_rows_loaded": 0,
    "test_evaluated": False,
    "selection_metric": "validation_accuracy",
    "tie_policy": "first_checkpoint",
    "selected_epoch": selected_epoch,
    "selected_checkpoint": str(best_checkpoint),
    "selected_validation": selected_metrics,
    "validation_history": validation_history,
    "experiment_accuracy_gate": config["evaluation"]["experiment_accuracy_gate"],
    "program_launch_target": config["evaluation"]["program_launch_target"],
    "experiment_gate_passed": (
        selected_metrics["accuracy"]
        >= config["evaluation"]["experiment_accuracy_gate"]
    ),
    "launch_target_reached_on_validation": (
        selected_metrics["accuracy"]
        >= config["evaluation"]["program_launch_target"]
    ),
    "roundtrip_reload_verified": True,
    "selected_model_path": str(SELECTED_MODEL_DIR),
    "weight_file": weight_path.name,
    "weight_sha256": sha256(weight_path),
    "finalization_elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": torch.mps.driver_allocated_memory() / (1024**3),
}
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"selected_validation_loss: {selected_metrics['loss']:.6f}")
print(f"selected_validation_accuracy: {selected_metrics['accuracy']:.6f}")
print(f"selected_validation_macro_f1: {selected_metrics['macro_f1']:.6f}")
print(f"experiment_gate_passed: {result['experiment_gate_passed']}")
print(
    "launch_target_reached_on_validation: "
    f"{result['launch_target_reached_on_validation']}"
)
print("roundtrip_reload_verified: True")
print(f"selected_model: {SELECTED_MODEL_DIR}")
print(f"weight_sha256: {result['weight_sha256']}")
print("test_evaluated: False")
print(f"result: {RESULT_PATH}")
print("exp_006_finalization_ok: True")
