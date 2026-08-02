#!/usr/bin/env python3
"""Create a deterministic, intent-balanced baseline pilot set."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = PROJECT_ROOT / "data" / "banking77" / "raw" / "test.csv"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "manifest.json"
PILOT_PATH = PROJECT_ROOT / "evaluation" / "pilot.jsonl"
PILOT_MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "pilot_manifest.json"

SEED = 3407
SAMPLES_PER_LABEL = 2

dataset_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
labels = dataset_manifest["labels"]

with TEST_DATA_PATH.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
for source_index, row in enumerate(rows):
    grouped[row["category"]].append(
        {
            "source_index": source_index,
            "text": row["text"],
            "expected": row["category"],
        }
    )

if sorted(grouped) != labels:
    raise SystemExit("Pilot preparation failed: dataset labels do not match manifest.")
if set(len(records) for records in grouped.values()) != {40}:
    raise SystemExit("Pilot preparation failed: expected 40 test rows per intent.")

rng = random.Random(SEED)
pilot: list[dict[str, object]] = []
for label in labels:
    pilot.extend(rng.sample(grouped[label], SAMPLES_PER_LABEL))
rng.shuffle(pilot)

label_counts = Counter(str(record["expected"]) for record in pilot)
expected_total = len(labels) * SAMPLES_PER_LABEL
if len(pilot) != expected_total:
    raise SystemExit("Pilot preparation failed: unexpected total row count.")
if set(label_counts.values()) != {SAMPLES_PER_LABEL}:
    raise SystemExit("Pilot preparation failed: pilot is not label-balanced.")

pilot_content = "".join(
    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    for record in pilot
)
PILOT_PATH.write_text(pilot_content, encoding="utf-8")
pilot_sha256 = hashlib.sha256(pilot_content.encode("utf-8")).hexdigest()

manifest = {
    "name": "banking77-balanced-pilot-v1",
    "source_dataset_revision": dataset_manifest["source_revision"],
    "source_split": "test",
    "source_split_rows": len(rows),
    "selection": "independent seeded random sample within each intent, then shuffle",
    "seed": SEED,
    "samples_per_label": SAMPLES_PER_LABEL,
    "labels": len(labels),
    "pilot_rows": len(pilot),
    "label_counts": dict(sorted(label_counts.items())),
    "pilot_sha256": pilot_sha256,
}
PILOT_MANIFEST_PATH.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — evaluation pilot preparation")
print(f"source_test_rows: {len(rows)}")
print(f"labels: {len(labels)}")
print(f"samples_per_label: {SAMPLES_PER_LABEL}")
print(f"pilot_rows: {len(pilot)}")
print(f"seed: {SEED}")
print(f"pilot_sha256: {pilot_sha256}")
print(f"pilot: {PILOT_PATH}")
print(f"manifest: {PILOT_MANIFEST_PATH}")
print("pilot_preparation_ok: True")
