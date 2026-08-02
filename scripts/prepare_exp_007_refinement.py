#!/usr/bin/env python3
"""Prepare train-only class weights for Experiment 007 BERT refinement."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-007-bert-class-balanced-refinement.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
PARENT_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-006" / "result.json"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-007" / "refinement_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Experiment 007 preparation failed: use .venv-encoder/bin/python.")
if OUTPUT_PATH.exists():
    raise SystemExit("Experiment 007 preparation stopped: manifest already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
parent_result = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
parent_model_path = Path(parent_result["selected_model_path"])
parent_weight_path = parent_model_path / parent_result["weight_file"]
expected_parent_hash = config["model"]["parent_weight_sha256"]
if sha256(parent_weight_path) != expected_parent_hash:
    raise SystemExit("Experiment 007 preparation failed: parent weight checksum mismatch.")
if parent_result["test_evaluated"]:
    raise SystemExit("Experiment 007 preparation failed: parent test policy mismatch.")

train_file = data_manifest["files"]["train"]
train_path = Path(train_file["path"])
if sha256(train_path) != train_file["sha256"]:
    raise SystemExit("Experiment 007 preparation failed: training checksum mismatch.")
with train_path.open(encoding="utf-8") as handle:
    train_records = [json.loads(line) for line in handle]

labels = data_manifest["label_order"]
counts = Counter(str(record["label_name"]) for record in train_records)
if set(counts) != set(labels) or len(train_records) != config["data"]["train_rows"]:
    raise SystemExit("Experiment 007 preparation failed: training label mismatch.")

# Inverse square root is deliberately moderate: inverse frequency would give
# rare classes too much influence during continued fine-tuning.
raw_weights = {
    label: math.sqrt(len(train_records) / (len(labels) * counts[label]))
    for label in labels
}
mean_weight = sum(raw_weights.values()) / len(raw_weights)
weights = {label: raw_weights[label] / mean_weight for label in labels}
if not all(math.isfinite(value) and value > 0 for value in weights.values()):
    raise SystemExit("Experiment 007 preparation failed: invalid class weight.")
if not math.isclose(sum(weights.values()) / len(weights), 1.0, abs_tol=1e-12):
    raise SystemExit("Experiment 007 preparation failed: weights are not normalized.")

ranked = sorted(
    (
        {
            "label": label,
            "training_rows": counts[label],
            "weight": weights[label],
        }
        for label in labels
    ),
    key=lambda row: (-row["weight"], row["label"]),
)
manifest = {
    "experiment": config["experiment"],
    "parent_experiment": config["model"]["parent_experiment"],
    "parent_model_path": str(parent_model_path),
    "parent_weight_sha256": expected_parent_hash,
    "parent_validation_accuracy": config["evaluation"][
        "parent_validation_accuracy"
    ],
    "formula": config["objective"]["class_weight_formula"],
    "normalization": "unweighted mean across 77 labels equals 1.0",
    "training_rows": len(train_records),
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
    "labels": len(labels),
    "minimum_weight": min(weights.values()),
    "maximum_weight": max(weights.values()),
    "mean_weight": sum(weights.values()) / len(weights),
    "class_weights": weights,
    "ranked_classes": ranked,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — Experiment 007 refinement preparation")
print(f"parent_experiment: {manifest['parent_experiment']}")
print(f"parent_validation_accuracy: {manifest['parent_validation_accuracy']:.6f}")
print(f"training_rows: {len(train_records)}")
print("validation_rows_loaded: 0")
print("test_rows_loaded: 0")
print(f"labels: {len(labels)}")
print(f"minimum_weight: {manifest['minimum_weight']:.6f}")
print(f"maximum_weight: {manifest['maximum_weight']:.6f}")
print(f"mean_weight: {manifest['mean_weight']:.6f}")
print("highest_weight_classes:")
for row in ranked[:10]:
    print(f"  {row['label']}: rows={row['training_rows']}, weight={row['weight']:.6f}")
print(f"manifest: {OUTPUT_PATH}")
print("exp_007_refinement_preparation_ok: True")
