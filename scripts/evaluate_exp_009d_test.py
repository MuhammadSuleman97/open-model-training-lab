#!/usr/bin/env python3
"""Run the one-time sealed BANKING77 test evaluation for Experiment 009d."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-009d-deberta-v3-large-float32.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009" / "model_manifest.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
FINAL_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "result.json"
TEST_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "test_result.json"
PREDICTIONS_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "test_predictions.jsonl"
SELECTED_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "selected-model"


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


def fail(message: str) -> None:
    raise SystemExit(f"Experiment 009d test evaluation stopped: {message}")


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    fail("use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    fail("PyTorch MPS is unavailable.")
if TEST_RESULT_PATH.exists() or PREDICTIONS_PATH.exists():
    fail("sealed test evaluation artifacts already exist; refusing to overwrite.")
if not FINAL_RESULT_PATH.is_file() or not SELECTED_MODEL_DIR.is_dir():
    fail("run finalize_exp_009d.py first.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
final_result = json.loads(FINAL_RESULT_PATH.read_text(encoding="utf-8"))
model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
weight_path = SELECTED_MODEL_DIR / "model.safetensors"
if not weight_path.is_file():
    fail("selected model weights are missing.")
if sha256(weight_path) != final_result["weight_sha256"]:
    fail("selected model weight checksum does not match the finalized result.")
if final_result["test_evaluated"] is not False:
    fail("finalized result does not state that the test was sealed.")

test_file = data_manifest["files"]["test"]
test_path = Path(test_file["path"])
if sha256(test_path) != test_file["sha256"]:
    fail("sealed test checksum mismatch.")
with test_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]
if len(source_records) != config["data"]["test_rows"]:
    fail("sealed test row count mismatch.")

labels = data_manifest["label_order"]
tokenizer = AutoTokenizer.from_pretrained(
    SELECTED_MODEL_DIR,
    local_files_only=True,
    use_fast=False,
)
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
    num_workers=0,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
)

print("Open Model Training Lab — Experiment 009d sealed test evaluation")
print(f"model_id: {model_manifest['model_id']}")
print(f"model_revision: {model_manifest['resolved_revision']}")
print(f"weight_sha256: {final_result['weight_sha256']}")
print("device: mps")
print(f"test_rows: {len(source_records)}")
print(f"labels: {len(labels)}")
print("loading_selected_model: True")

started = time.monotonic()
model = AutoModelForSequenceClassification.from_pretrained(
    SELECTED_MODEL_DIR,
    local_files_only=True,
)
model.to(torch.device("mps"))
model.float()
model.eval()
predicted_ids: list[int] = []
expected_ids: list[int] = []
loss_sum = 0.0
rows = 0
with torch.no_grad():
    for batch in loader:
        batch = {name: tensor.to("mps") for name, tensor in batch.items()}
        output = model(**batch)
        batch_size = int(batch["labels"].shape[0])
        loss_sum += float(output.loss.detach().cpu()) * batch_size
        rows += batch_size
        predicted_ids.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
        expected_ids.extend(batch["labels"].detach().cpu().tolist())
torch.mps.synchronize()

if rows != len(source_records):
    fail("evaluation row count changed during inference.")
metrics = {
    "loss": loss_sum / rows,
    "accuracy": float(accuracy_score(expected_ids, predicted_ids)),
    "macro_f1": float(f1_score(expected_ids, predicted_ids, average="macro", zero_division=0)),
}
if not all(math.isfinite(value) for value in metrics.values()):
    fail("non-finite test metric.")

per_label: dict[str, dict[str, float | int]] = {}
for label_id, label in enumerate(labels):
    label_expected = [value == label_id for value in expected_ids]
    label_predicted = [value == label_id for value in predicted_ids]
    true_positives = sum(expected == label_id and predicted == label_id for expected, predicted in zip(expected_ids, predicted_ids))
    support = sum(label_expected)
    per_label[label] = {
        "support": support,
        "correct": true_positives,
        "accuracy": true_positives / support if support else 0.0,
        "f1": float(f1_score(label_expected, label_predicted, zero_division=0)),
    }

with PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
    for index, (record, expected, predicted) in enumerate(zip(source_records, expected_ids, predicted_ids)):
        handle.write(
            json.dumps(
                {
                    "index": index,
                    "expected_id": expected,
                    "expected_label": labels[expected],
                    "predicted_id": predicted,
                    "predicted_label": labels[predicted],
                    "correct": expected == predicted,
                    "text": record["text"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )

confusion = confusion_matrix(expected_ids, predicted_ids, labels=list(range(len(labels)))).tolist()
result = {
    "experiment": config["experiment"],
    "model_id": model_manifest["model_id"],
    "model_revision": model_manifest["resolved_revision"],
    "weight_sha256": final_result["weight_sha256"],
    "device": "mps",
    "model_parameter_dtype": "float32",
    "test_rows": len(source_records),
    "test_evaluated": True,
    "metrics": metrics,
    "validation_accuracy": final_result["selected_validation"]["accuracy"],
    "validation_macro_f1": final_result["selected_validation"]["macro_f1"],
    "accuracy_delta_validation_to_test": metrics["accuracy"] - final_result["selected_validation"]["accuracy"],
    "program_launch_target": config["evaluation"]["program_launch_target"],
    "launch_target_reached_on_test": metrics["accuracy"] >= config["evaluation"]["program_launch_target"],
    "per_label": per_label,
    "confusion_matrix": confusion,
    "test_source_sha256": test_file["sha256"],
    "predictions_path": str(PREDICTIONS_PATH),
    "predictions_sha256": sha256(PREDICTIONS_PATH),
    "elapsed_seconds": time.monotonic() - started,
    "peak_driver_memory_gib": torch.mps.driver_allocated_memory() / (1024**3),
}
TEST_RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"test_loss: {metrics['loss']:.6f}")
print(f"test_accuracy: {metrics['accuracy']:.6f}")
print(f"test_macro_f1: {metrics['macro_f1']:.6f}")
print(f"validation_accuracy: {result['validation_accuracy']:.6f}")
print(f"accuracy_delta_validation_to_test: {result['accuracy_delta_validation_to_test']:+.6f}")
print(f"launch_target_reached_on_test: {result['launch_target_reached_on_test']}")
print(f"peak_driver_memory_gib: {result['peak_driver_memory_gib']:.3f}")
print(f"elapsed_seconds: {result['elapsed_seconds']:.3f}")
print(f"predictions: {PREDICTIONS_PATH}")
print(f"result: {TEST_RESULT_PATH}")
print("test_evaluation_ok: True")
