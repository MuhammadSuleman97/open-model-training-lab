#!/usr/bin/env python3
"""Prepare a nested 1,925-example balanced data-scaling experiment."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "manifest.json"
SFT_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"
EXP_002_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "exp_002_data_manifest.json"
)
SFT_DIR = PROJECT_ROOT / "data" / "banking77" / "sft"
EXP_002_DIR = (
    PROJECT_ROOT / "data" / "banking77" / "exp_002_balanced_539"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "banking77" / "exp_003_balanced_1925"
)
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VALID_PATH = OUTPUT_DIR / "valid.jsonl"
MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "exp_003_data_manifest.json"
)

SEED = 3408
TRAIN_PER_LABEL = 25
VALID_PER_LABEL = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def label(record: dict[str, object]) -> str:
    messages = record["messages"]
    return str(messages[-1]["content"])


def user_text(record: dict[str, object]) -> str:
    messages = record["messages"]
    return str(messages[-2]["content"])


dataset_manifest = json.loads(
    DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
)
sft_manifest = json.loads(SFT_MANIFEST_PATH.read_text(encoding="utf-8"))
exp_002_manifest = json.loads(
    EXP_002_MANIFEST_PATH.read_text(encoding="utf-8")
)

for name in ("train.jsonl", "valid.jsonl"):
    source_path = SFT_DIR / name
    if sha256(source_path) != sft_manifest["files"][name]["sha256"]:
        raise SystemExit(
            f"Experiment 003 data preparation failed: source {name} "
            "checksum mismatch."
        )
    exp_002_path = EXP_002_DIR / name
    if sha256(exp_002_path) != exp_002_manifest["files"][name]["sha256"]:
        raise SystemExit(
            f"Experiment 003 data preparation failed: Experiment 002 "
            f"{name} checksum mismatch."
        )

labels = dataset_manifest["labels"]
source_train = read_jsonl(SFT_DIR / "train.jsonl")
source_valid = read_jsonl(SFT_DIR / "valid.jsonl")
exp_002_train = read_jsonl(EXP_002_DIR / "train.jsonl")
valid_records = read_jsonl(EXP_002_DIR / "valid.jsonl")

grouped_train: dict[str, list[dict[str, object]]] = defaultdict(list)
grouped_valid: dict[str, list[dict[str, object]]] = defaultdict(list)
for record in source_train:
    grouped_train[label(record)].append(record)
for record in source_valid:
    grouped_valid[label(record)].append(record)

if sorted(grouped_train) != labels or sorted(grouped_valid) != labels:
    raise SystemExit(
        "Experiment 003 data preparation failed: source labels do not match."
    )

# Repeat Experiment 002's within-label randomization exactly. Shuffling the
# validation candidates is retained even though the already-locked Experiment
# 002 validation file is reused, because it preserves the RNG sequence that
# selected the nested training examples.
rng = random.Random(SEED)
train_records: list[dict[str, object]] = []
for intent in labels:
    train_candidates = list(grouped_train[intent])
    valid_candidates = list(grouped_valid[intent])
    rng.shuffle(train_candidates)
    rng.shuffle(valid_candidates)
    train_records.extend(train_candidates[:TRAIN_PER_LABEL])

rng.shuffle(train_records)

train_texts = {user_text(record) for record in train_records}
valid_texts = {user_text(record) for record in valid_records}
exp_002_train_texts = {user_text(record) for record in exp_002_train}
if train_texts & valid_texts:
    raise SystemExit(
        "Experiment 003 data preparation failed: train/validation overlap."
    )
if not exp_002_train_texts < train_texts:
    raise SystemExit(
        "Experiment 003 data preparation failed: Experiment 002 is not a "
        "strict subset of the new training set."
    )

train_counts = Counter(label(record) for record in train_records)
valid_counts = Counter(label(record) for record in valid_records)
if set(train_counts.values()) != {TRAIN_PER_LABEL}:
    raise SystemExit(
        "Experiment 003 data preparation failed: training is not balanced."
    )
if set(valid_counts.values()) != {VALID_PER_LABEL}:
    raise SystemExit(
        "Experiment 003 data preparation failed: validation changed."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
write_jsonl(TRAIN_PATH, train_records)
write_jsonl(VALID_PATH, valid_records)

manifest = {
    "name": "exp-003-balanced-1925-data",
    "purpose": (
        "Nested data-scaling experiment with exactly 25 training examples "
        "per BANKING77 intent and the unchanged Experiment 002 validation set."
    ),
    "seed": SEED,
    "source_sft_train_sha256": sft_manifest["files"]["train.jsonl"]["sha256"],
    "source_sft_valid_sha256": sft_manifest["files"]["valid.jsonl"]["sha256"],
    "parent_experiment": "exp-002-balanced-539",
    "parent_training_rows_included": len(exp_002_train_texts),
    "validation_reused_from_parent": True,
    "labels": len(labels),
    "train_per_label": TRAIN_PER_LABEL,
    "valid_per_label": VALID_PER_LABEL,
    "train_rows": len(train_records),
    "valid_rows": len(valid_records),
    "train_valid_text_overlap": len(train_texts & valid_texts),
    "files": {
        "train.jsonl": {
            "sha256": sha256(TRAIN_PATH),
            "bytes": TRAIN_PATH.stat().st_size,
        },
        "valid.jsonl": {
            "sha256": sha256(VALID_PATH),
            "bytes": VALID_PATH.stat().st_size,
        },
    },
}
MANIFEST_PATH.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — Experiment 003 data preparation")
print(f"seed: {SEED}")
print(f"labels: {len(labels)}")
print(f"train_per_label: {TRAIN_PER_LABEL}")
print(f"train_rows: {len(train_records)}")
print(f"parent_training_rows_included: {len(exp_002_train_texts)}")
print(f"valid_per_label: {VALID_PER_LABEL}")
print(f"valid_rows: {len(valid_records)}")
print("validation_reused_from_parent: True")
print(f"train_valid_text_overlap: {len(train_texts & valid_texts)}")
print(f"train_sha256: {sha256(TRAIN_PATH)}")
print(f"valid_sha256: {sha256(VALID_PATH)}")
print(f"manifest: {MANIFEST_PATH}")
print("exp_003_data_preparation_ok: True")
