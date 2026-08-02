#!/usr/bin/env python3
"""Prepare Experiment 011 train-only paired-boundary refinement data."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-011-deberta-upper-layer-refinement.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
VALIDATION_ANALYSIS_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "validation_error_analysis.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "data"
TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

SEED = 3416
TARGET_MULTIPLIER = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if TRAIN_OUTPUT_PATH.exists() or MANIFEST_PATH.exists():
    raise SystemExit("Experiment 011 data preparation stopped: output already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
validation_analysis = json.loads(VALIDATION_ANALYSIS_PATH.read_text(encoding="utf-8"))
if validation_analysis.get("test_rows_loaded") != 0:
    raise SystemExit("Experiment 011 data preparation stopped: validation analysis loaded test data.")
if data_manifest.get("rows", {}).get("test") != 3080:
    raise SystemExit("Experiment 011 data preparation stopped: canonical split manifest is unexpected.")

train_file = data_manifest["files"]["train"]
train_path = Path(train_file["path"])
if sha256(train_path) != train_file["sha256"]:
    raise SystemExit("Experiment 011 data preparation stopped: source train checksum mismatch.")
with train_path.open(encoding="utf-8") as handle:
    source_rows = [json.loads(line) for line in handle]
if len(source_rows) != data_manifest["rows"]["train"]:
    raise SystemExit("Experiment 011 data preparation stopped: source train row count mismatch.")

target_labels = set(config["target_labels"])
unknown_labels = target_labels.difference(data_manifest["label_order"])
if unknown_labels:
    raise SystemExit(f"Experiment 011 data preparation stopped: unknown target labels {sorted(unknown_labels)}.")
target_rows = [row for row in source_rows if row["label_name"] in target_labels]
if not target_rows:
    raise SystemExit("Experiment 011 data preparation stopped: no target rows found.")

expanded_rows = list(source_rows)
for _ in range(TARGET_MULTIPLIER - 1):
    expanded_rows.extend(target_rows)
random.Random(SEED).shuffle(expanded_rows)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with TRAIN_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
    for row in expanded_rows:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

source_counts = Counter(row["label_name"] for row in source_rows)
expanded_counts = Counter(row["label_name"] for row in expanded_rows)
manifest = {
    "experiment": "exp-011-deberta-upper-layer-refinement",
    "parent_experiment": "exp-010-deberta-targeted-oversampling",
    "parent_validation_accuracy": 0.9285714285714286,
    "selection_policy": "validation_only; observed test metrics excluded",
    "seed": SEED,
    "target_labels": sorted(target_labels),
    "target_label_count": len(target_labels),
    "target_multiplier": TARGET_MULTIPLIER,
    "target_rows_original": len(target_rows),
    "target_rows_added": len(target_rows) * (TARGET_MULTIPLIER - 1),
    "source_train_path": str(train_path),
    "source_train_sha256": train_file["sha256"],
    "source_train_rows": len(source_rows),
    "train_rows": len(expanded_rows),
    "train_path": str(TRAIN_OUTPUT_PATH),
    "train_sha256": sha256(TRAIN_OUTPUT_PATH),
    "source_counts": dict(sorted(source_counts.items())),
    "expanded_counts": dict(sorted(expanded_counts.items())),
    "validation_analysis_path": str(VALIDATION_ANALYSIS_PATH),
    "validation_rows": data_manifest["rows"]["valid"],
    "test_rows_loaded": 0,
}
MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print("Open Model Training Lab — Experiment 011 train-only boundary data preparation")
print(f"seed: {SEED}")
print(f"source_train_rows: {len(source_rows)}")
print(f"target_labels: {len(target_labels)}")
print(f"target_rows_original: {len(target_rows)}")
print(f"target_multiplier: {TARGET_MULTIPLIER}")
print(f"train_rows: {len(expanded_rows)}")
print(f"target_rows_added: {manifest['target_rows_added']}")
print(f"train_sha256: {manifest['train_sha256']}")
print(f"manifest: {MANIFEST_PATH}")
print("validation_rows_loaded: 0")
print("test_rows_loaded: 0")
print("exp_011_data_preparation_ok: True")
