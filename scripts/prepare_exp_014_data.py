#!/usr/bin/env python3
"""Prepare broader train-only rival labels for Experiment 014."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-014-deberta-broader-hard-negative.json"
TRAIN_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "train.jsonl"
AUDIT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-012-label-audit" / "manifest.json"
AUDIT_PREDICTIONS_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-012-label-audit" / "oof_predictions.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-014" / "data"
OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stop(message: str) -> None:
    raise SystemExit(f"Experiment 014 data preparation stopped: {message}")


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    stop("use .venv-encoder/bin/python.")
if OUTPUT_DIR.exists() or OUTPUT_PATH.exists() or MANIFEST_PATH.exists():
    stop("output already exists; refusing to overwrite.")
if not TRAIN_PATH.is_file() or not AUDIT_MANIFEST_PATH.is_file() or not AUDIT_PREDICTIONS_PATH.is_file():
    stop("required train or audit artifact is missing.")

audit_manifest = json.loads(AUDIT_MANIFEST_PATH.read_text(encoding="utf-8"))
source_hash = sha256(TRAIN_PATH)
if audit_manifest.get("source_train_sha256") != source_hash:
    stop("audit source checksum does not match the canonical training file.")
if audit_manifest.get("validation_rows_loaded") != 0 or audit_manifest.get("test_rows_loaded") != 0:
    stop("audit manifest reports held-out split access.")

train_records: list[dict[str, object]] = []
with TRAIN_PATH.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        record = json.loads(line)
        if not isinstance(record.get("text"), str) or not isinstance(record.get("label"), int):
            stop(f"malformed training row {line_number}.")
        train_records.append(record)

audit_records: list[dict[str, object]] = []
with AUDIT_PREDICTIONS_PATH.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        record = json.loads(line)
        if record.get("row_index") != line_number - 1:
            stop(f"audit row index mismatch at line {line_number}.")
        audit_records.append(record)

expected_rows = int(CONFIG["data"]["source_train_rows"])
if len(train_records) != expected_rows or len(audit_records) != expected_rows:
    stop(f"row mismatch: train={len(train_records)} audit={len(audit_records)} expected={expected_rows}.")

threshold = float(CONFIG["data"]["hard_negative_probability_threshold"])
prepared: list[dict[str, object]] = []
active_rows = 0
for index, (train_record, audit_record) in enumerate(zip(train_records, audit_records)):
    if train_record["text"] != audit_record.get("text"):
        stop(f"text mismatch at row {index}.")
    given_label = train_record["label_name"]
    if given_label != audit_record.get("given_label"):
        stop(f"given-label mismatch at row {index}.")
    predicted_label = audit_record.get("predicted_label")
    predicted_probability = float(audit_record.get("predicted_probability", 0.0))
    active = bool(audit_record.get("model_disagrees")) and predicted_probability >= threshold
    if active:
        active_rows += 1
    prepared.append(
        {
            "text": train_record["text"],
            "label": train_record["label"],
            "label_name": given_label,
            "hard_negative_label": predicted_label if active else None,
            "hard_negative_probability": predicted_probability if active else None,
            "hard_negative_active": active,
        }
    )

if active_rows <= 162:
    stop("broader threshold did not activate more rows than Exp013.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
OUTPUT_PATH.write_text(
    "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in prepared),
    encoding="utf-8",
)
manifest = {
    "experiment": CONFIG["experiment"],
    "parent_experiment": CONFIG["model"]["parent_experiment"],
    "source_train_path": str(TRAIN_PATH),
    "source_train_sha256": source_hash,
    "audit_manifest_path": str(AUDIT_MANIFEST_PATH),
    "audit_manifest_sha256": sha256(AUDIT_MANIFEST_PATH),
    "audit_predictions_path": str(AUDIT_PREDICTIONS_PATH),
    "audit_predictions_sha256": sha256(AUDIT_PREDICTIONS_PATH),
    "train_rows": len(prepared),
    "hard_negative_rows": active_rows,
    "hard_negative_probability_threshold": threshold,
    "labels_rewritten": False,
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
    "selection_policy": CONFIG["selection_policy"],
    "files": {"train.jsonl": {"path": str(OUTPUT_PATH), "sha256": sha256(OUTPUT_PATH)}},
}
MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print("Open Model Training Lab — Experiment 014 broader train-only data preparation")
print(f"source_train_rows: {len(train_records)}")
print(f"hard_negative_rows: {active_rows}")
print(f"hard_negative_probability_threshold: {threshold}")
print("labels_rewritten: False")
print("validation_rows_loaded: 0")
print("test_rows_loaded: 0")
print(f"train_sha256: {sha256(OUTPUT_PATH)}")
print(f"manifest: {MANIFEST_PATH}")
print("exp_014_data_preparation_ok: True")
