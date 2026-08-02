#!/usr/bin/env python3
"""Compare rejected class-balanced refinement against its BERT parent."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-007-bert-class-balanced-refinement.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
PARENT_ANALYSIS_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-006" / "validation_error_analysis.json"
TRAINING_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-007" / "training_result.json"
REFINEMENT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-007" / "refinement_manifest.json"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-007" / "validation_transition_analysis.json"


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
    raise SystemExit("Experiment 007 analysis stopped: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 007 analysis stopped: PyTorch MPS is unavailable.")
if OUTPUT_PATH.exists():
    raise SystemExit("Experiment 007 analysis stopped: result already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
parent_analysis = json.loads(PARENT_ANALYSIS_PATH.read_text(encoding="utf-8"))
training_result = json.loads(TRAINING_RESULT_PATH.read_text(encoding="utf-8"))
refinement_manifest = json.loads(REFINEMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
if training_result["promoted"]:
    raise SystemExit("Experiment 007 analysis stopped: expected a rejected child.")

validation_file = data_manifest["files"]["valid"]
validation_path = Path(validation_file["path"])
if sha256(validation_path) != validation_file["sha256"]:
    raise SystemExit("Experiment 007 analysis stopped: validation checksum mismatch.")
with validation_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]
parent_rows = parent_analysis["all_predictions"]
if len(parent_rows) != len(source_records):
    raise SystemExit("Experiment 007 analysis stopped: parent prediction mismatch.")

child_checkpoint = Path(training_result["best_child_checkpoint"])
tokenizer = AutoTokenizer.from_pretrained(child_checkpoint, local_files_only=True)
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

print("Open Model Training Lab — Experiment 007 validation transition analysis")
print(f"child_checkpoint: {child_checkpoint}")
print(f"validation_rows: {len(source_records)}")
print("test_rows_loaded: 0")
print("loading_rejected_child: True")
device = torch.device("mps")
model = AutoModelForSequenceClassification.from_pretrained(
    child_checkpoint,
    local_files_only=True,
)
model.to(device)
model.eval()
child_predictions: list[int] = []
with torch.no_grad():
    for batch in loader:
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        child_predictions.extend(
            model(**batch).logits.argmax(dim=-1).detach().cpu().tolist()
        )
torch.mps.synchronize()

labels = data_manifest["label_order"]
parent_correct: list[bool] = []
child_correct: list[bool] = []
changed_by_expected: Counter[str] = Counter()
fixes_by_expected: Counter[str] = Counter()
harms_by_expected: Counter[str] = Counter()
for index, (source, parent, child_id) in enumerate(
    zip(source_records, parent_rows, child_predictions, strict=True)
):
    expected_id = int(source["label"])
    expected_name = labels[expected_id]
    parent_is_correct = bool(parent["correct"])
    child_is_correct = child_id == expected_id
    parent_correct.append(parent_is_correct)
    child_correct.append(child_is_correct)
    if str(parent["predicted"]) != labels[child_id]:
        changed_by_expected[expected_name] += 1
    if not parent_is_correct and child_is_correct:
        fixes_by_expected[expected_name] += 1
    if parent_is_correct and not child_is_correct:
        harms_by_expected[expected_name] += 1

both_correct = sum(p and c for p, c in zip(parent_correct, child_correct, strict=True))
parent_only = sum(p and not c for p, c in zip(parent_correct, child_correct, strict=True))
child_only = sum(not p and c for p, c in zip(parent_correct, child_correct, strict=True))
both_wrong = sum(not p and not c for p, c in zip(parent_correct, child_correct, strict=True))
child_accuracy = sum(child_correct) / len(child_correct)
if not math.isclose(
    child_accuracy,
    training_result["best_child_validation_accuracy"],
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit("Experiment 007 analysis stopped: child accuracy mismatch.")

weighted_labels = {
    label
    for label, weight in refinement_manifest["class_weights"].items()
    if weight > 1.0
}
weighted_fixes = sum(fixes_by_expected[label] for label in weighted_labels)
weighted_harms = sum(harms_by_expected[label] for label in weighted_labels)
all_changed_labels = sorted(
    set(changed_by_expected) | set(fixes_by_expected) | set(harms_by_expected)
)
per_label_changes = [
    {
        "label": label,
        "class_weight": refinement_manifest["class_weights"][label],
        "changed_predictions": changed_by_expected[label],
        "parent_errors_fixed": fixes_by_expected[label],
        "parent_correct_harmed": harms_by_expected[label],
        "net_correct_change": fixes_by_expected[label] - harms_by_expected[label],
    }
    for label in all_changed_labels
]
per_label_changes.sort(
    key=lambda row: (row["net_correct_change"], -row["changed_predictions"], row["label"])
)
analysis = {
    "experiment": config["experiment"],
    "split": "validation",
    "validation_rows": len(source_records),
    "test_rows_loaded": 0,
    "parent_accuracy": training_result["parent_validation_accuracy"],
    "child_accuracy": child_accuracy,
    "strict_accuracy_change": (
        child_accuracy - training_result["parent_validation_accuracy"]
    ),
    "transitions": {
        "both_correct": both_correct,
        "parent_only_correct": parent_only,
        "child_only_correct": child_only,
        "both_wrong": both_wrong,
        "changed_predictions": sum(changed_by_expected.values()),
    },
    "upweighted_classes": len(weighted_labels),
    "upweighted_class_fixes": weighted_fixes,
    "upweighted_class_harms": weighted_harms,
    "per_label_changes": per_label_changes,
    "decision": "reject_child_keep_exp_006",
}
OUTPUT_PATH.write_text(
    json.dumps(analysis, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"parent_accuracy: {analysis['parent_accuracy']:.6f}")
print(f"child_accuracy: {child_accuracy:.6f}")
print(f"strict_accuracy_change: {analysis['strict_accuracy_change']:+.6f}")
print(f"changed_predictions: {analysis['transitions']['changed_predictions']}")
print(f"parent_only_correct: {parent_only}")
print(f"child_only_correct: {child_only}")
print(f"upweighted_class_fixes: {weighted_fixes}")
print(f"upweighted_class_harms: {weighted_harms}")
print("largest_negative_class_changes:")
for row in per_label_changes[:10]:
    if row["net_correct_change"] < 0:
        print(
            f"  {row['label']}: net={row['net_correct_change']}, "
            f"weight={row['class_weight']:.6f}"
        )
print("decision: reject_child_keep_exp_006")
print(f"analysis: {OUTPUT_PATH}")
print("test_evaluated: False")
print("exp_007_validation_analysis_ok: True")
