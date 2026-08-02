#!/usr/bin/env python3
"""Compare Experiment 013 with Exp011 on the validation split only."""

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
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
PARENT_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "result.json"
CHILD_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-013" / "result.json"
PARENT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "selected-model"
CHILD_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-013" / "selected-model"
ANALYSIS_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-013" / "validation_transition_analysis.json"


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
    raise SystemExit("Experiment 013 validation analysis stopped: MPS unavailable.")
if not PARENT_RESULT_PATH.is_file() or not CHILD_RESULT_PATH.is_file():
    raise SystemExit("Experiment 013 validation analysis stopped: finalized models are missing.")
if ANALYSIS_PATH.exists():
    raise SystemExit("Experiment 013 validation analysis stopped: analysis already exists.")

data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
parent_result = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
child_result = json.loads(CHILD_RESULT_PATH.read_text(encoding="utf-8"))
if child_result.get("test_rows_loaded") != 0 or child_result.get("test_evaluated") is not False:
    raise SystemExit("Experiment 013 validation analysis stopped: child reports test access.")
valid_file = data_manifest["files"]["valid"]
valid_path = Path(valid_file["path"])
if sha256(valid_path) != valid_file["sha256"]:
    raise SystemExit("Experiment 013 validation analysis stopped: validation checksum mismatch.")
with valid_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]
labels = data_manifest["label_order"]
tokenizer = AutoTokenizer.from_pretrained(CHILD_MODEL_DIR, local_files_only=True, use_fast=False)
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


def predict(model_dir: Path) -> list[int]:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(torch.device("mps"))
    model.float()
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {name: tensor.to("mps") for name, tensor in batch.items()}
            output = model(**batch)
            predictions.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
    torch.mps.synchronize()
    return predictions


expected = [int(record["label"]) for record in source_records]
print("Open Model Training Lab — Experiment 013 validation transition analysis")
print(f"validation_rows: {len(expected)}")
print("test_rows_loaded: 0")
print("loading_parent_and_child_models: True")
parent_predictions = predict(PARENT_MODEL_DIR)
child_predictions = predict(CHILD_MODEL_DIR)
parent_correct = [actual == predicted for actual, predicted in zip(expected, parent_predictions)]
child_correct = [actual == predicted for actual, predicted in zip(expected, child_predictions)]
parent_accuracy = float(accuracy_score(expected, parent_predictions))
child_accuracy = float(accuracy_score(expected, child_predictions))
parent_macro_f1 = float(f1_score(expected, parent_predictions, average="macro", zero_division=0))
child_macro_f1 = float(f1_score(expected, child_predictions, average="macro", zero_division=0))
both_correct = sum(left and right for left, right in zip(parent_correct, child_correct))
parent_only_correct = sum(left and not right for left, right in zip(parent_correct, child_correct))
child_only_correct = sum(not left and right for left, right in zip(parent_correct, child_correct))
both_wrong = sum(not left and not right for left, right in zip(parent_correct, child_correct))
changed = sum(left != right for left, right in zip(parent_predictions, child_predictions))

class_deltas: dict[str, dict[str, int | float]] = {}
for label_id, label in enumerate(labels):
    indices = [index for index, actual in enumerate(expected) if actual == label_id]
    parent_correct_count = sum(parent_predictions[index] == label_id for index in indices)
    child_correct_count = sum(child_predictions[index] == label_id for index in indices)
    class_deltas[label] = {
        "parent_correct": parent_correct_count,
        "child_correct": child_correct_count,
        "delta": child_correct_count - parent_correct_count,
        "support": len(indices),
    }
transition_counts = Counter(
    (labels[parent], labels[child])
    for parent, child in zip(parent_predictions, child_predictions)
    if parent != child
)
analysis = {
    "experiment": child_result["experiment"],
    "parent_experiment": child_result["parent_experiment"],
    "validation_rows": len(expected),
    "test_rows_loaded": 0,
    "parent_accuracy": parent_accuracy,
    "child_accuracy": child_accuracy,
    "strict_accuracy_change": child_accuracy - parent_accuracy,
    "parent_macro_f1": parent_macro_f1,
    "child_macro_f1": child_macro_f1,
    "macro_f1_change": child_macro_f1 - parent_macro_f1,
    "changed_predictions": changed,
    "parent_only_correct": parent_only_correct,
    "child_only_correct": child_only_correct,
    "both_correct": both_correct,
    "both_wrong": both_wrong,
    "hard_negative_rows": child_result["hard_negative_rows"],
    "hard_negative_margin": child_result["hard_negative_margin"],
    "hard_negative_weight": child_result["hard_negative_weight"],
    "class_deltas": class_deltas,
    "top_changed_label_transitions": [
        {"parent_label": parent, "child_label": child, "count": count}
        for (parent, child), count in transition_counts.most_common(20)
    ],
    "selection_policy": "validation_only; test data not read",
}
ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"parent_accuracy: {parent_accuracy:.6f}")
print(f"child_accuracy: {child_accuracy:.6f}")
print(f"strict_accuracy_change: {child_accuracy - parent_accuracy:+.6f}")
print(f"parent_macro_f1: {parent_macro_f1:.6f}")
print(f"child_macro_f1: {child_macro_f1:.6f}")
print(f"changed_predictions: {changed}")
print(f"parent_only_correct: {parent_only_correct}")
print(f"child_only_correct: {child_only_correct}")
print(f"both_correct: {both_correct}")
print(f"both_wrong: {both_wrong}")
print("largest_class_deltas:")
for label, stats in sorted(class_deltas.items(), key=lambda item: (item[1]["delta"], item[0]))[:12]:
    print(f"  {label}: {stats['parent_correct']} -> {stats['child_correct']} ({stats['delta']:+d})")
print("top_changed_label_transitions:")
for item in analysis["top_changed_label_transitions"][:12]:
    print(f"  {item['parent_label']} -> {item['child_label']}: {item['count']}")
print(f"analysis: {ANALYSIS_PATH}")
print("validation_transition_analysis_ok: True")
