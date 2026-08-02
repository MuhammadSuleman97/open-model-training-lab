#!/usr/bin/env python3
"""Analyze Experiment 006 validation errors without touching the test set."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-006-bert-large-classifier.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
EXP_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-006" / "result.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "encoder"
    / "exp-006"
    / "validation_error_analysis.json"
)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Validation analysis stopped: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Validation analysis stopped: PyTorch MPS is unavailable.")
if OUTPUT_PATH.exists():
    raise SystemExit("Validation analysis stopped: analysis already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
experiment_result = json.loads(EXP_RESULT_PATH.read_text(encoding="utf-8"))
model_path = Path(experiment_result["selected_model_path"])
weight_path = model_path / experiment_result["weight_file"]
if sha256(weight_path) != experiment_result["weight_sha256"]:
    raise SystemExit("Validation analysis stopped: selected model checksum mismatch.")

validation_file = data_manifest["files"]["valid"]
validation_path = Path(validation_file["path"])
if sha256(validation_path) != validation_file["sha256"]:
    raise SystemExit("Validation analysis stopped: validation checksum mismatch.")
with validation_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]

labels = data_manifest["label_order"]
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
encoded_records: list[dict[str, object]] = []
for index, record in enumerate(source_records):
    encoded = tokenizer(
        str(record["text"]),
        max_length=config["tokenization"]["max_length"],
        truncation=True,
    )
    encoded["labels"] = int(record["label"])
    encoded["record_index"] = index
    encoded_records.append(encoded)


def collate(records: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    indices = [int(record.pop("record_index")) for record in records]
    batch = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )(records)
    batch["record_index"] = torch.tensor(indices, dtype=torch.long)
    return batch


loader = DataLoader(
    ListDataset(encoded_records),
    batch_size=config["batching"]["eval_batch_size"],
    shuffle=False,
    collate_fn=collate,
)
device = torch.device("mps")
print("Open Model Training Lab — Experiment 006 validation error analysis")
print(f"selected_model: {model_path}")
print(f"validation_rows: {len(source_records)}")
print("test_rows_loaded: 0")
print("loading_model_and_analyzing: True")

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=True,
)
model.to(device)
model.eval()
prediction_rows: list[dict[str, object]] = []
with torch.no_grad():
    for batch in loader:
        record_indices = batch.pop("record_index").tolist()
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        logits = model(**batch).logits
        probabilities = torch.softmax(logits.float(), dim=-1)
        top_probabilities, top_ids = probabilities.topk(k=3, dim=-1)
        for row_index, probability_row, id_row in zip(
            record_indices,
            top_probabilities.cpu().tolist(),
            top_ids.cpu().tolist(),
            strict=True,
        ):
            source = source_records[row_index]
            expected_id = int(source["label"])
            predicted_id = int(id_row[0])
            prediction_rows.append(
                {
                    "text": source["text"],
                    "expected": labels[expected_id],
                    "predicted": labels[predicted_id],
                    "correct": predicted_id == expected_id,
                    "confidence": float(probability_row[0]),
                    "top_3": [
                        {
                            "label": labels[int(label_id)],
                            "probability": float(probability),
                        }
                        for label_id, probability in zip(
                            id_row,
                            probability_row,
                            strict=True,
                        )
                    ],
                }
            )
torch.mps.synchronize()

correct = sum(bool(row["correct"]) for row in prediction_rows)
accuracy = correct / len(prediction_rows)
recorded_accuracy = float(experiment_result["selected_validation"]["accuracy"])
if not math.isclose(accuracy, recorded_accuracy, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("Validation analysis stopped: accuracy does not match result.")

label_totals: Counter[str] = Counter()
label_correct: Counter[str] = Counter()
confusions: Counter[tuple[str, str]] = Counter()
errors_by_expected: dict[str, list[dict[str, object]]] = defaultdict(list)
for row in prediction_rows:
    expected = str(row["expected"])
    predicted = str(row["predicted"])
    label_totals[expected] += 1
    if row["correct"]:
        label_correct[expected] += 1
    else:
        confusions[(expected, predicted)] += 1
        errors_by_expected[expected].append(row)

per_label = [
    {
        "label": label,
        "correct": label_correct[label],
        "total": label_totals[label],
        "recall": label_correct[label] / label_totals[label],
        "errors": errors_by_expected[label],
    }
    for label in labels
]
per_label.sort(key=lambda row: (row["recall"], row["label"]))
top_confusions = [
    {
        "expected": expected,
        "predicted": predicted,
        "count": count,
    }
    for (expected, predicted), count in confusions.most_common()
]
target_correct = math.ceil(
    config["evaluation"]["program_launch_target"] * len(prediction_rows)
)
analysis = {
    "experiment": config["experiment"],
    "split": "validation",
    "validation_rows": len(prediction_rows),
    "test_rows_loaded": 0,
    "correct": correct,
    "errors": len(prediction_rows) - correct,
    "accuracy": accuracy,
    "launch_target": config["evaluation"]["program_launch_target"],
    "target_correct": target_correct,
    "additional_correct_needed": target_correct - correct,
    "per_label": per_label,
    "confusion_pairs": top_confusions,
    "all_predictions": prediction_rows,
}
OUTPUT_PATH.write_text(
    json.dumps(analysis, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"correct: {correct}/{len(prediction_rows)}")
print(f"errors: {len(prediction_rows) - correct}")
print(f"accuracy: {accuracy:.6f}")
print(f"target_correct_for_95_percent: {target_correct}")
print(f"additional_correct_needed: {target_correct - correct}")
print("worst_validation_labels:")
for row in per_label[:10]:
    print(f"  {row['label']}: {row['correct']}/{row['total']}")
print("most_common_confusions:")
for row in top_confusions[:10]:
    print(f"  {row['expected']} -> {row['predicted']}: {row['count']}")
print(f"analysis: {OUTPUT_PATH}")
print("test_evaluated: False")
print("validation_error_analysis_ok: True")
