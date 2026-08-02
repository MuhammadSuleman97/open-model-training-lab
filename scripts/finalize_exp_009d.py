#!/usr/bin/env python3
"""Reload, verify and package Experiment 009d's best validation checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-009d-deberta-v3-large-float32.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009" / "model_manifest.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
TRAINER_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "trainer-output"
SELECTED_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "selected-model"
RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "result.json"


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


def checkpoint_number(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.to(device)
    model.float()
    model.eval()
    predictions: list[int] = []
    expected: list[int] = []
    loss_sum = 0.0
    rows = 0
    with torch.no_grad():
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            output = model(**batch)
            batch_size = int(batch["labels"].shape[0])
            loss_sum += float(output.loss.detach().cpu()) * batch_size
            rows += batch_size
            predictions.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
            expected.extend(batch["labels"].detach().cpu().tolist())
    torch.mps.synchronize()
    metrics = {
        "loss": loss_sum / rows,
        "accuracy": float(accuracy_score(expected, predictions)),
        "macro_f1": float(f1_score(expected, predictions, average="macro", zero_division=0)),
    }
    return metrics, predictions


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Experiment 009d finalization stopped: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 009d finalization stopped: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 009d finalization stopped: result already exists.")
if SELECTED_MODEL_DIR.exists():
    raise SystemExit("Experiment 009d finalization stopped: selected model already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
checkpoints = sorted(
    [path for path in TRAINER_DIR.glob("checkpoint-*") if (path / "trainer_state.json").is_file()],
    key=checkpoint_number,
)
if not checkpoints:
    raise SystemExit("Experiment 009d finalization stopped: no checkpoints found.")
latest_checkpoint = checkpoints[-1]
trainer_state = json.loads((latest_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
best_checkpoint = Path(trainer_state["best_model_checkpoint"])
best_accuracy = float(trainer_state["best_metric"])
if not best_checkpoint.is_dir():
    raise SystemExit("Experiment 009d finalization stopped: best checkpoint missing.")
if not (best_checkpoint / "model.safetensors").is_file():
    raise SystemExit("Experiment 009d finalization stopped: best weights missing.")

validation_file = data_manifest["files"]["valid"]
validation_path = Path(validation_file["path"])
if sha256(validation_path) != validation_file["sha256"]:
    raise SystemExit("Experiment 009d finalization stopped: validation checksum mismatch.")
with validation_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]
labels = data_manifest["label_order"]
tokenizer = AutoTokenizer.from_pretrained(best_checkpoint, local_files_only=True, use_fast=False)
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

print("Open Model Training Lab — Experiment 009d finalization")
print(f"selected_checkpoint: {best_checkpoint}")
print(f"selected_epoch: {trainer_state['epoch'] if best_checkpoint == latest_checkpoint else 'validation-selected'}")
print(f"recorded_validation_accuracy: {best_accuracy:.6f}")
print("selection_metric: validation_accuracy")
print("test_rows_loaded: 0")
print("loading_selected_checkpoint: True")

device = torch.device("mps")
started = time.monotonic()
model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint, local_files_only=True)
selected_metrics, selected_predictions = evaluate(model, loader, device)
if not all(math.isfinite(value) for value in selected_metrics.values()):
    raise SystemExit("Experiment 009d finalization stopped: non-finite validation metric.")
if not math.isclose(selected_metrics["accuracy"], best_accuracy, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("Experiment 009d finalization stopped: checkpoint accuracy mismatch.")

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
    raise SystemExit("Experiment 009d finalization stopped: roundtrip predictions changed.")
if any(
    not math.isclose(roundtrip_metrics[key], selected_metrics[key], rel_tol=0.0, abs_tol=1e-9)
    for key in selected_metrics
):
    raise SystemExit("Experiment 009d finalization stopped: roundtrip metrics changed.")

weight_path = SELECTED_MODEL_DIR / "model.safetensors"
if not weight_path.is_file():
    raise SystemExit("Experiment 009d finalization stopped: selected weight missing.")
validation_history = [
    {
        key: entry[key]
        for key in ("epoch", "eval_loss", "eval_accuracy", "eval_macro_f1")
    }
    for entry in trainer_state["log_history"]
    if "eval_accuracy" in entry
]
logged_losses = [entry["loss"] for entry in trainer_state["log_history"] if "loss" in entry]
result = {
    "experiment": config["experiment"],
    "model_id": model_manifest["model_id"],
    "model_revision": model_manifest["resolved_revision"],
    "device": "mps",
    "model_parameter_dtype": "float32",
    "train_rows": config["data"]["train_rows"],
    "validation_rows": len(source_records),
    "test_rows_loaded": 0,
    "test_evaluated": False,
    "selection_metric": "validation_accuracy",
    "selected_checkpoint": str(best_checkpoint),
    "selected_validation": selected_metrics,
    "validation_history": validation_history,
    "last_logged_train_loss": float(logged_losses[-1]),
    "bert_champion_validation_accuracy": config["evaluation"]["bert_champion_validation_accuracy"],
    "strict_improvement_vs_bert": selected_metrics["accuracy"] - config["evaluation"]["bert_champion_validation_accuracy"],
    "experiment_accuracy_gate": config["evaluation"]["experiment_accuracy_gate"],
    "program_launch_target": config["evaluation"]["program_launch_target"],
    "experiment_gate_passed": selected_metrics["accuracy"] >= config["evaluation"]["experiment_accuracy_gate"],
    "launch_target_reached_on_validation": selected_metrics["accuracy"] >= config["evaluation"]["program_launch_target"],
    "roundtrip_reload_verified": True,
    "selected_model_path": str(SELECTED_MODEL_DIR),
    "weight_file": weight_path.name,
    "weight_sha256": sha256(weight_path),
    "finalization_elapsed_seconds": time.monotonic() - started,
    "peak_driver_memory_gib": torch.mps.driver_allocated_memory() / (1024**3),
}
RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"selected_validation_loss: {selected_metrics['loss']:.6f}")
print(f"selected_validation_accuracy: {selected_metrics['accuracy']:.6f}")
print(f"selected_validation_macro_f1: {selected_metrics['macro_f1']:.6f}")
print(f"bert_champion_validation_accuracy: {result['bert_champion_validation_accuracy']:.6f}")
print(f"strict_improvement_vs_bert: {result['strict_improvement_vs_bert']:+.6f}")
print(f"experiment_gate_passed: {result['experiment_gate_passed']}")
print(f"launch_target_reached_on_validation: {result['launch_target_reached_on_validation']}")
print("roundtrip_reload_verified: True")
print(f"selected_model: {SELECTED_MODEL_DIR}")
print(f"weight_sha256: {result['weight_sha256']}")
print("test_evaluated: False")
print(f"result: {RESULT_PATH}")
print("exp_009d_finalization_ok: True")
