#!/usr/bin/env python3
"""Inspect the Exp011 champion's validation errors without test access."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "result.json"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "selected-model"
ANALYSIS_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "validation_error_analysis.json"


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
    raise SystemExit(f"Exp011 validation error analysis stopped: {message}")


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    stop("use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    stop("PyTorch MPS is unavailable.")
if not RESULT_PATH.is_file() or not MODEL_DIR.is_dir():
    stop("finalized Exp011 model is missing.")
if ANALYSIS_PATH.exists():
    stop("analysis already exists; refusing to overwrite.")

result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
if result.get("test_rows_loaded") != 0 or result.get("test_evaluated") is not False:
    stop("Exp011 result reports test access.")
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
valid_file = data_manifest["files"]["valid"]
valid_path = Path(valid_file["path"])
if sha256(valid_path) != valid_file["sha256"]:
    stop("validation checksum mismatch.")
with valid_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]
if len(source_records) != 770:
    stop("expected 770 validation rows.")

labels = data_manifest["label_order"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=False)
encoded_records: list[dict[str, object]] = []
for record in source_records:
    encoded = tokenizer(str(record["text"]), max_length=128, truncation=True)
    encoded["labels"] = int(record["label"])
    encoded_records.append(encoded)
loader = DataLoader(
    ListDataset(encoded_records),
    batch_size=32,
    shuffle=False,
    num_workers=0,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
)

print("Open Model Training Lab — Exp011 validation error analysis")
print(f"model: {MODEL_DIR}")
print(f"validation_rows: {len(source_records)}")
print("test_rows_loaded: 0")
print("loading_selected_model: True")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.to(torch.device("mps"))
model.float()
model.eval()
predicted_ids: list[int] = []
expected_ids: list[int] = []
predicted_confidences: list[float] = []
runner_up_ids: list[int] = []
runner_up_probabilities: list[float] = []
with torch.no_grad():
    for batch in loader:
        batch = {name: tensor.to("mps") for name, tensor in batch.items()}
        output = model(**batch)
        probabilities = torch.softmax(output.logits, dim=-1)
        top_values, top_ids = probabilities.topk(k=2, dim=-1)
        predicted_ids.extend(top_ids[:, 0].detach().cpu().tolist())
        runner_up_ids.extend(top_ids[:, 1].detach().cpu().tolist())
        predicted_confidences.extend(top_values[:, 0].detach().cpu().tolist())
        runner_up_probabilities.extend(top_values[:, 1].detach().cpu().tolist())
        expected_ids.extend(batch["labels"].detach().cpu().tolist())
torch.mps.synchronize()

accuracy = float(accuracy_score(expected_ids, predicted_ids))
macro_f1 = float(f1_score(expected_ids, predicted_ids, average="macro", zero_division=0))
if not math.isclose(accuracy, float(result["selected_validation"]["accuracy"]), rel_tol=0.0, abs_tol=1e-12):
    stop("recomputed validation accuracy does not match the finalized result.")

per_label: list[dict[str, object]] = []
for label_id, label in enumerate(labels):
    indices = [index for index, expected in enumerate(expected_ids) if expected == label_id]
    correct = sum(predicted_ids[index] == label_id for index in indices)
    per_label.append(
        {
            "label": label,
            "correct": correct,
            "support": len(indices),
            "accuracy": correct / len(indices) if indices else 0.0,
            "errors": len(indices) - correct,
        }
    )
per_label.sort(key=lambda item: (float(item["accuracy"]), str(item["label"])))
confusions = Counter(
    (labels[expected], labels[predicted])
    for expected, predicted in zip(expected_ids, predicted_ids)
    if expected != predicted
)

errors: list[dict[str, object]] = []
for index, (expected, predicted) in enumerate(zip(expected_ids, predicted_ids)):
    if expected == predicted:
        continue
    errors.append(
        {
            "row_index": index,
            "text": source_records[index]["text"],
            "expected": labels[expected],
            "predicted": labels[predicted],
            "predicted_confidence": predicted_confidences[index],
            "runner_up": labels[runner_up_ids[index]],
            "runner_up_probability": runner_up_probabilities[index],
            "top_two_margin": predicted_confidences[index] - runner_up_probabilities[index],
        }
    )

high_confidence_errors = sum(error["predicted_confidence"] >= 0.9 for error in errors)
low_margin_errors = sum(error["top_two_margin"] <= 0.1 for error in errors)
analysis = {
    "experiment": result["experiment"],
    "model_weight_sha256": result["weight_sha256"],
    "validation_rows": len(source_records),
    "test_rows_loaded": 0,
    "validation_accuracy": accuracy,
    "validation_macro_f1": macro_f1,
    "finalized_validation_accuracy": result["selected_validation"]["accuracy"],
    "error_count": len(errors),
    "high_confidence_error_count": high_confidence_errors,
    "low_margin_error_count": low_margin_errors,
    "worst_labels": per_label[:15],
    "top_confusions": [
        {"expected_label": expected, "predicted_label": predicted, "count": count}
        for (expected, predicted), count in confusions.most_common(20)
    ],
    "error_examples": sorted(errors, key=lambda item: (-float(item["predicted_confidence"]), int(item["row_index"]))),
    "selection_policy": "validation_only; test metrics not read",
}
ANALYSIS_PATH.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"correct: {len(source_records) - len(errors)}/{len(source_records)}")
print(f"errors: {len(errors)}")
print(f"validation_accuracy: {accuracy:.6f}")
print(f"validation_macro_f1: {macro_f1:.6f}")
print(f"high_confidence_errors_ge_0.9: {high_confidence_errors}")
print(f"low_margin_errors_le_0.1: {low_margin_errors}")
print("worst_labels:")
for item in per_label[:12]:
    print(f"  {item['label']}: {item['correct']}/{item['support']} ({item['accuracy']:.3f})")
print("top_confusions:")
for (expected, predicted), count in confusions.most_common(12):
    print(f"  {expected} -> {predicted}: {count}")
print("highest_confidence_errors:")
for error in analysis["error_examples"][:8]:
    print(
        f"  [{error['row_index']}] {error['expected']} -> {error['predicted']} "
        f"(confidence={error['predicted_confidence']:.3f}, margin={error['top_two_margin']:.3f})"
    )
print(f"analysis: {ANALYSIS_PATH}")
print("validation_error_analysis_ok: True")
