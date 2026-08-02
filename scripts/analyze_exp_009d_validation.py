#!/usr/bin/env python3
"""Analyze Experiment 009d errors using validation data only."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-009d-deberta-v3-large-float32.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
FINAL_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "result.json"
ANALYSIS_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "validation_error_analysis.json"
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


if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 009d validation analysis stopped: MPS unavailable.")
if not FINAL_RESULT_PATH.is_file() or not SELECTED_MODEL_DIR.is_dir():
    raise SystemExit("Experiment 009d validation analysis stopped: finalize the model first.")
if ANALYSIS_PATH.exists():
    raise SystemExit("Experiment 009d validation analysis stopped: analysis already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
final_result = json.loads(FINAL_RESULT_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
valid_file = data_manifest["files"]["valid"]
valid_path = Path(valid_file["path"])
if sha256(valid_path) != valid_file["sha256"]:
    raise SystemExit("Experiment 009d validation analysis stopped: checksum mismatch.")
with valid_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]
if len(source_records) != config["data"]["validation_rows"]:
    raise SystemExit("Experiment 009d validation analysis stopped: row count mismatch.")

labels = data_manifest["label_order"]
tokenizer = AutoTokenizer.from_pretrained(SELECTED_MODEL_DIR, local_files_only=True, use_fast=False)
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

print("Open Model Training Lab — Experiment 009d validation error analysis")
print(f"validation_rows: {len(source_records)}")
print("test_rows_loaded: 0")
print("loading_selected_model: True")
model = AutoModelForSequenceClassification.from_pretrained(SELECTED_MODEL_DIR, local_files_only=True)
model.to(torch.device("mps"))
model.float()
model.eval()
predicted_ids: list[int] = []
expected_ids: list[int] = []
with torch.no_grad():
    for batch in loader:
        batch = {name: tensor.to("mps") for name, tensor in batch.items()}
        output = model(**batch)
        predicted_ids.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
        expected_ids.extend(batch["labels"].detach().cpu().tolist())
torch.mps.synchronize()

accuracy = float(accuracy_score(expected_ids, predicted_ids))
macro_f1 = float(f1_score(expected_ids, predicted_ids, average="macro", zero_division=0))
per_label = []
for label_id, label in enumerate(labels):
    support = sum(value == label_id for value in expected_ids)
    correct = sum(expected == label_id and predicted == label_id for expected, predicted in zip(expected_ids, predicted_ids))
    per_label.append(
        {
            "label": label,
            "correct": correct,
            "support": support,
            "accuracy": correct / support if support else 0.0,
            "errors": support - correct,
        }
    )
per_label.sort(key=lambda item: (item["accuracy"], item["label"]))
confusions = Counter(
    (labels[expected], labels[predicted])
    for expected, predicted in zip(expected_ids, predicted_ids)
    if expected != predicted
)
analysis = {
    "experiment": final_result["experiment"],
    "validation_rows": len(source_records),
    "test_rows_loaded": 0,
    "validation_accuracy": accuracy,
    "validation_macro_f1": macro_f1,
    "finalized_validation_accuracy": final_result["selected_validation"]["accuracy"],
    "worst_labels": per_label[:15],
    "top_confusions": [
        {"expected_label": expected, "predicted_label": predicted, "count": count}
        for (expected, predicted), count in confusions.most_common(20)
    ],
    "selection_policy": "validation_only; test metrics not read",
}
ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"correct: {sum(expected == predicted for expected, predicted in zip(expected_ids, predicted_ids))}")
print(f"errors: {sum(expected != predicted for expected, predicted in zip(expected_ids, predicted_ids))}")
print(f"validation_accuracy: {accuracy:.6f}")
print(f"validation_macro_f1: {macro_f1:.6f}")
print("worst_labels:")
for item in per_label[:12]:
    print(f"  {item['label']}: {item['correct']}/{item['support']} ({item['accuracy']:.3f})")
print("top_confusions:")
for (expected, predicted), count in confusions.most_common(12):
    print(f"  {expected} -> {predicted}: {count}")
print(f"analysis: {ANALYSIS_PATH}")
print("validation_error_analysis_ok: True")
