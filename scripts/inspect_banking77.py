#!/usr/bin/env python3
"""Download, validate, and summarize the canonical BANKING77 dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


SOURCE_REPOSITORY = "PolyAI-LDN/task-specific-datasets"
SOURCE_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_ROOT = (
    "https://raw.githubusercontent.com/"
    f"{SOURCE_REPOSITORY}/{SOURCE_REVISION}/banking_data"
)
SOURCE_FILES = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
}

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "banking77"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(filename: str, expected_sha256: str) -> Path:
    destination = RAW_DIR / filename
    if destination.exists() and sha256(destination) == expected_sha256:
        print(f"{filename}: cached and verified")
        return destination

    request = Request(
        f"{SOURCE_ROOT}/{filename}",
        headers={"User-Agent": "open-model-training-lab/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        content = response.read()

    observed_sha256 = hashlib.sha256(content).hexdigest()
    if observed_sha256 != expected_sha256:
        raise SystemExit(
            f"{filename} checksum mismatch: expected {expected_sha256}, "
            f"received {observed_sha256}"
        )

    destination.write_bytes(content)
    print(f"{filename}: downloaded and verified")
    return destination


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "category"]:
            raise SystemExit(
                f"Unexpected columns in {path.name}: {reader.fieldnames}"
            )
        return list(reader)


print("Open Model Training Lab — BANKING77 inspection")
print(f"source_repository: {SOURCE_REPOSITORY}")
print(f"source_revision: {SOURCE_REVISION}")

RAW_DIR.mkdir(parents=True, exist_ok=True)
paths = {
    filename: download_verified(filename, expected_sha256)
    for filename, expected_sha256 in SOURCE_FILES.items()
}

train = read_split(paths["train.csv"])
test = read_split(paths["test.csv"])
categories = json.loads(paths["categories.json"].read_text(encoding="utf-8"))

train_labels = {record["category"] for record in train}
test_labels = {record["category"] for record in test}
train_texts = {record["text"] for record in train}
test_texts = {record["text"] for record in test}

checks = {
    "train_rows": len(train),
    "test_rows": len(test),
    "label_count": len(categories),
    "all_rows_have_known_labels": (
        train_labels <= set(categories) and test_labels <= set(categories)
    ),
    "same_labels_in_both_splits": train_labels == test_labels == set(categories),
    "train_unique_texts": len(train_texts),
    "test_unique_texts": len(test_texts),
    "train_test_text_overlap": len(train_texts & test_texts),
}

expected = {
    "train_rows": 10003,
    "test_rows": 3080,
    "label_count": 77,
    "all_rows_have_known_labels": True,
    "same_labels_in_both_splits": True,
    "train_unique_texts": 10003,
    "test_unique_texts": 3080,
    "train_test_text_overlap": 0,
}

for key, value in checks.items():
    print(f"{key}: {value}")

if checks != expected:
    raise SystemExit(
        "BANKING77 validation failed. The observed source does not match the "
        "recorded experiment source."
    )

manifest = {
    "source_repository": SOURCE_REPOSITORY,
    "source_revision": SOURCE_REVISION,
    "license": "CC-BY-4.0",
    "files": {
        filename: {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for filename, path in sorted(paths.items())
    },
    "splits": {
        "train": {"rows": len(train)},
        "test": {"rows": len(test)},
    },
    "columns": ["text", "category"],
    "labels": sorted(categories),
    "checks": checks,
}

MANIFEST_PATH.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"manifest: {MANIFEST_PATH}")
print("sample_train_records:")
for record in train[:3]:
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
print("dataset_validation_ok: True")
