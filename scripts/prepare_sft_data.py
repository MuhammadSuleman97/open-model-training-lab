#!/usr/bin/env python3
"""Create deterministic, leakage-safe BANKING77 SFT train/validation files."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "banking77" / "raw" / "train.csv"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "manifest.json"
PROMPT_CONFIG_PATH = PROJECT_ROOT / "configs" / "banking77_prompt.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "banking77" / "sft"
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VALID_PATH = OUTPUT_DIR / "valid.jsonl"
SFT_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"

SEED = 3407
VALIDATION_PER_LABEL = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_chat_record(
    source_record: dict[str, str],
    system_prompt: str,
) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": source_record["text"]},
            {"role": "assistant", "content": source_record["category"]},
        ]
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


dataset_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
prompt_config = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
labels = dataset_manifest["labels"]

source_sha256 = sha256(SOURCE_PATH)
if source_sha256 != dataset_manifest["files"]["train.csv"]["sha256"]:
    raise SystemExit("SFT preparation failed: source train checksum mismatch.")

with SOURCE_PATH.open(encoding="utf-8", newline="") as handle:
    source_rows = list(csv.DictReader(handle))

if len(source_rows) != dataset_manifest["splits"]["train"]["rows"]:
    raise SystemExit("SFT preparation failed: source train row count mismatch.")

grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in source_rows:
    grouped[row["category"]].append(row)
if sorted(grouped) != labels:
    raise SystemExit("SFT preparation failed: source labels do not match manifest.")

rng = random.Random(SEED)
train_source: list[dict[str, str]] = []
valid_source: list[dict[str, str]] = []
for label in labels:
    label_rows = list(grouped[label])
    rng.shuffle(label_rows)
    valid_source.extend(label_rows[:VALIDATION_PER_LABEL])
    train_source.extend(label_rows[VALIDATION_PER_LABEL:])

rng.shuffle(train_source)
rng.shuffle(valid_source)

train_texts = {record["text"] for record in train_source}
valid_texts = {record["text"] for record in valid_source}
if train_texts & valid_texts:
    raise SystemExit("SFT preparation failed: train/validation text overlap.")
if len(train_source) + len(valid_source) != len(source_rows):
    raise SystemExit("SFT preparation failed: source rows were lost or duplicated.")

system_prompt = prompt_config["system_template"].format(
    labels=prompt_config["label_separator"].join(labels)
)
train_records = [
    to_chat_record(record, system_prompt) for record in train_source
]
valid_records = [
    to_chat_record(record, system_prompt) for record in valid_source
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
write_jsonl(TRAIN_PATH, train_records)
write_jsonl(VALID_PATH, valid_records)

train_label_counts = Counter(record["category"] for record in train_source)
valid_label_counts = Counter(record["category"] for record in valid_source)
if set(valid_label_counts.values()) != {VALIDATION_PER_LABEL}:
    raise SystemExit("SFT preparation failed: validation set is not balanced.")

manifest = {
    "name": "banking77-sft-v1",
    "source_dataset_revision": dataset_manifest["source_revision"],
    "source_train_sha256": source_sha256,
    "prompt_version": prompt_config["prompt_version"],
    "format": "MLX-LM chat JSONL",
    "selection": (
        "seeded shuffle within each label; reserve exactly 10 per label for "
        "validation; independently shuffle final train and validation records"
    ),
    "seed": SEED,
    "validation_per_label": VALIDATION_PER_LABEL,
    "source_rows": len(source_rows),
    "train_rows": len(train_records),
    "valid_rows": len(valid_records),
    "labels": len(labels),
    "train_label_counts": dict(sorted(train_label_counts.items())),
    "valid_label_counts": dict(sorted(valid_label_counts.items())),
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
SFT_MANIFEST_PATH.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — SFT data preparation")
print(f"source_rows: {len(source_rows)}")
print(f"train_rows: {len(train_records)}")
print(f"valid_rows: {len(valid_records)}")
print(f"labels: {len(labels)}")
print(f"validation_per_label: {VALIDATION_PER_LABEL}")
print(f"train_valid_text_overlap: {len(train_texts & valid_texts)}")
print(f"train_sha256: {sha256(TRAIN_PATH)}")
print(f"valid_sha256: {sha256(VALID_PATH)}")
print(f"manifest: {SFT_MANIFEST_PATH}")
print("sft_data_preparation_ok: True")
