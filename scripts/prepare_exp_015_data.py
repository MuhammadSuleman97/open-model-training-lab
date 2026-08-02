#!/usr/bin/env python3
"""Prepare a train-only, conservatively noise-pruned BANKING77 split."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-015-deberta-noise-pruned.json"
SOURCE_DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
AUDIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-012-label-audit" / "manifest.json"
AUDIT_PREDICTIONS_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-012-label-audit" / "oof_predictions.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-015" / "data"
OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stop(message: str) -> None:
    raise SystemExit(f"Experiment 015 data preparation stopped: {message}")


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    stop("use .venv-encoder/bin/python.")
if OUTPUT_DIR.exists() or OUTPUT_PATH.exists() or MANIFEST_PATH.exists():
    stop("output already exists; refusing to overwrite.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
source_manifest = json.loads(SOURCE_DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
audit_manifest = json.loads(AUDIT_MANIFEST_PATH.read_text(encoding="utf-8"))
source_file = source_manifest["files"]["train"]
source_path = Path(source_file["path"])
if sha256(source_path) != source_file["sha256"]:
    stop("canonical training checksum mismatch.")
if audit_manifest.get("source_train_sha256") != source_file["sha256"]:
    stop("audit and canonical training checksums do not match.")
if audit_manifest.get("validation_rows_loaded") != 0 or audit_manifest.get("test_rows_loaded") != 0:
    stop("the train-only audit reports held-out split access.")
if sha256(AUDIT_PREDICTIONS_PATH) != audit_manifest["files"]["oof_predictions"]["sha256"]:
    stop("audit prediction checksum mismatch.")

with source_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]
with AUDIT_PREDICTIONS_PATH.open(encoding="utf-8") as handle:
    audit_records = [json.loads(line) for line in handle]

expected_rows = int(config["data"]["source_train_rows"])
if len(source_records) != expected_rows or len(audit_records) != expected_rows:
    stop(
        f"row mismatch: source={len(source_records)} audit={len(audit_records)} "
        f"expected={expected_rows}."
    )

threshold = float(config["data"]["audit_probability_threshold"])
kept: list[dict[str, object]] = []
removed: list[dict[str, object]] = []
for index, (source, audit) in enumerate(zip(source_records, audit_records)):
    if audit.get("row_index") != index:
        stop(f"audit row index mismatch at row {index}.")
    if source.get("text") != audit.get("text"):
        stop(f"text mismatch at row {index}.")
    if source.get("label_name") != audit.get("given_label"):
        stop(f"label mismatch at row {index}.")
    should_remove = bool(audit.get("model_disagrees")) and float(
        audit.get("given_probability", 1.0)
    ) < threshold
    if should_remove:
        removed.append(
            {
                "row_index": index,
                "given_label": source["label_name"],
                "given_probability": float(audit["given_probability"]),
                "predicted_label": audit["predicted_label"],
                "predicted_probability": float(audit["predicted_probability"]),
            }
        )
    else:
        kept.append(source)

if not removed:
    stop("filter selected no rows.")
if any(record not in source_records for record in kept):
    stop("prepared data contains a non-source row.")
if any("label" not in record or "label_name" not in record for record in kept):
    stop("prepared data contains a malformed label.")

source_counts = Counter(str(record["label_name"]) for record in source_records)
kept_counts = Counter(str(record["label_name"]) for record in kept)
removed_counts = Counter(str(record["given_label"]) for record in removed)
if set(kept_counts) != set(source_counts):
    stop("filter removed an entire label.")
if min(kept_counts.values()) < 10:
    stop("filter leaves fewer than ten examples for a label.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
OUTPUT_PATH.write_text(
    "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in kept),
    encoding="utf-8",
)
manifest = {
    "experiment": config["experiment"],
    "selection_policy": config["selection_policy"],
    "filter_rule": config["data"]["filter_rule"],
    "audit_probability_threshold": threshold,
    "source_data_manifest_path": str(SOURCE_DATA_MANIFEST_PATH),
    "source_data_manifest_sha256": sha256(SOURCE_DATA_MANIFEST_PATH),
    "source_train_path": str(source_path),
    "source_train_sha256": source_file["sha256"],
    "audit_manifest_path": str(AUDIT_MANIFEST_PATH),
    "audit_manifest_sha256": sha256(AUDIT_MANIFEST_PATH),
    "audit_predictions_path": str(AUDIT_PREDICTIONS_PATH),
    "audit_predictions_sha256": sha256(AUDIT_PREDICTIONS_PATH),
    "source_train_rows": len(source_records),
    "train_rows": len(kept),
    "removed_rows": len(removed),
    "removed_fraction": len(removed) / len(source_records),
    "labels_rewritten": False,
    "label_order": source_manifest["label_order"],
    "minimum_retained_rows_per_label": min(kept_counts.values()),
    "source_rows_per_label": dict(sorted(source_counts.items())),
    "retained_rows_per_label": dict(sorted(kept_counts.items())),
    "removed_rows_per_label": dict(sorted(removed_counts.items())),
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
    "files": {
        "train": {
            "path": str(OUTPUT_PATH),
            "sha256": sha256(OUTPUT_PATH),
        }
    },
}
MANIFEST_PATH.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — Experiment 015 train-only noise pruning")
print(f"source_train_rows: {len(source_records)}")
print(f"removed_rows: {len(removed)}")
print(f"retained_train_rows: {len(kept)}")
print(f"removed_fraction: {len(removed) / len(source_records):.6f}")
print(f"audit_probability_threshold: {threshold}")
print(f"minimum_retained_rows_per_label: {min(kept_counts.values())}")
print("labels_rewritten: False")
print("validation_rows_loaded: 0")
print("test_rows_loaded: 0")
print(f"train_sha256: {sha256(OUTPUT_PATH)}")
print(f"manifest: {MANIFEST_PATH}")
print("exp_015_data_preparation_ok: True")
