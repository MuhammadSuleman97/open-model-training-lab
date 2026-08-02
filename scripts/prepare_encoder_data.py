#!/usr/bin/env python3
"""Build and verify raw-text BANKING77 records for Experiment 006."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "exp-006-bert-large-classifier.json"
)
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "manifest.json"
SFT_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "model_manifest.json"
SFT_DIR = PROJECT_ROOT / "data" / "banking77" / "sft"
RAW_TEST_PATH = PROJECT_ROOT / "data" / "banking77" / "raw" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "data"
OUTPUT_MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sft_split(path: Path, label2id: dict[str, int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            source = json.loads(line)
            messages = source.get("messages", [])
            if [message.get("role") for message in messages] != [
                "system",
                "user",
                "assistant",
            ]:
                raise SystemExit(
                    f"Encoder data preparation failed: malformed messages at "
                    f"{path.name}:{line_number}."
                )
            text = messages[1].get("content")
            label_name = messages[2].get("content")
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(
                    f"Encoder data preparation failed: empty text at "
                    f"{path.name}:{line_number}."
                )
            if label_name not in label2id:
                raise SystemExit(
                    f"Encoder data preparation failed: unknown label {label_name!r}."
                )
            records.append(
                {
                    "text": text,
                    "label": label2id[label_name],
                    "label_name": label_name,
                }
            )
    return records


def read_test_split(label2id: dict[str, int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with RAW_TEST_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label_name = row["category"]
            if label_name not in label2id:
                raise SystemExit(
                    f"Encoder data preparation failed: unknown test label {label_name!r}."
                )
            records.append(
                {
                    "text": row["text"],
                    "label": label2id[label_name],
                    "label_name": label_name,
                }
            )
    return records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Encoder data preparation failed: use .venv-encoder/bin/python.")

experiment_config = json.loads(EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8"))
dataset_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
sft_manifest = json.loads(SFT_MANIFEST_PATH.read_text(encoding="utf-8"))
model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
labels = dataset_manifest["labels"]
label2id = {label: index for index, label in enumerate(labels)}

for split_name in ("train", "valid"):
    source_path = SFT_DIR / f"{split_name}.jsonl"
    expected_hash = sft_manifest["files"][f"{split_name}.jsonl"]["sha256"]
    if sha256(source_path) != expected_hash:
        raise SystemExit(
            f"Encoder data preparation failed: {split_name} checksum mismatch."
        )
if sha256(RAW_TEST_PATH) != dataset_manifest["files"]["test.csv"]["sha256"]:
    raise SystemExit("Encoder data preparation failed: test checksum mismatch.")

train_records = read_sft_split(SFT_DIR / "train.jsonl", label2id)
valid_records = read_sft_split(SFT_DIR / "valid.jsonl", label2id)
test_records = read_test_split(label2id)
expected_rows = experiment_config["data"]
observed_rows = {
    "train": len(train_records),
    "valid": len(valid_records),
    "test": len(test_records),
}
expected_by_split = {
    "train": expected_rows["train_rows"],
    "valid": expected_rows["validation_rows"],
    "test": expected_rows["test_rows"],
}
if observed_rows != expected_by_split:
    raise SystemExit(
        f"Encoder data preparation failed: row mismatch {observed_rows!r}."
    )

text_sets = {
    "train": {str(record["text"]) for record in train_records},
    "valid": {str(record["text"]) for record in valid_records},
    "test": {str(record["text"]) for record in test_records},
}
overlap = {
    "train_valid": len(text_sets["train"] & text_sets["valid"]),
    "train_test": len(text_sets["train"] & text_sets["test"]),
    "valid_test": len(text_sets["valid"] & text_sets["test"]),
}
if any(overlap.values()):
    raise SystemExit(f"Encoder data preparation failed: split overlap {overlap!r}.")

tokenizer = AutoTokenizer.from_pretrained(
    model_manifest["snapshot_path"],
    local_files_only=True,
)
# Max length is selected from train/validation only. Test remains uninspected.
development_texts = [
    str(record["text"]) for record in train_records + valid_records
]
tokenized = tokenizer(
    development_texts,
    add_special_tokens=True,
    truncation=False,
)["input_ids"]
lengths = [len(token_ids) for token_ids in tokenized]
max_length = experiment_config["tokenization"]["max_length"]
length_report = {
    "min": min(lengths),
    "p50": percentile(lengths, 0.50),
    "p95": percentile(lengths, 0.95),
    "p99": percentile(lengths, 0.99),
    "max": max(lengths),
    "over_configured_max": sum(length > max_length for length in lengths),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
paths: dict[str, Path] = {}
for split_name, records in (
    ("train", train_records),
    ("valid", valid_records),
    ("test", test_records),
):
    output_path = OUTPUT_DIR / f"{split_name}.jsonl"
    write_jsonl(output_path, records)
    paths[split_name] = output_path

output_manifest = {
    "experiment": experiment_config["experiment"],
    "label_order": labels,
    "rows": observed_rows,
    "text_overlap": overlap,
    "configured_max_length": max_length,
    "development_token_lengths": length_report,
    "test_inspection_policy": (
        "test rows/checksum/overlap verified; test token lengths and model outputs "
        "not inspected before final evaluation"
    ),
    "files": {
        split_name: {
            "path": str(path),
            "sha256": sha256(path),
        }
        for split_name, path in paths.items()
    },
}
OUTPUT_MANIFEST_PATH.write_text(
    json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — encoder data preparation")
print(f"train_rows: {observed_rows['train']}")
print(f"validation_rows: {observed_rows['valid']}")
print(f"test_rows_sealed: {observed_rows['test']}")
print(f"labels: {len(labels)}")
print(f"text_overlap: {overlap}")
print(f"development_token_lengths: {length_report}")
print(f"configured_max_length: {max_length}")
print(f"manifest: {OUTPUT_MANIFEST_PATH}")
print("encoder_data_preparation_ok: True")
