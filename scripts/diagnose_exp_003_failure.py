#!/usr/bin/env python3
"""Record and verify Experiment 003's numerical-instability failure."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-003-balanced-1925.yaml"
DATA_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "exp_003_data_manifest.json"
)
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "exp-003-balanced-1925"
LOG_PATH = EXPERIMENT_DIR / "training.log"
RESULT_PATH = EXPERIMENT_DIR / "result.json"
ADAPTER_FILE = (
    PROJECT_ROOT
    / "adapters"
    / "exp-003-balanced-1925"
    / "adapters.safetensors"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_float(value: str) -> float:
    return float(value.lower())


def json_safe_report(report: dict[str, object]) -> dict[str, object]:
    safe = dict(report)
    loss = float(safe["loss"])
    if not math.isfinite(loss):
        safe["loss"] = "nan" if math.isnan(loss) else str(loss)
    return safe


data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
log_text = LOG_PATH.read_text(encoding="utf-8")

training_reports = [
    {
        "iteration": int(iteration),
        "loss": parse_float(loss),
        "peak_memory_gb": float(peak_memory),
    }
    for iteration, loss, peak_memory in re.findall(
        r"Iter (\d+): Train loss ([^,]+),.*?Peak mem ([0-9.]+) GB",
        log_text,
    )
]
validation_reports = [
    {
        "iteration": int(iteration),
        "loss": parse_float(loss),
    }
    for iteration, loss in re.findall(
        r"Iter (\d+): Val loss ([^,]+),",
        log_text,
    )
]
if not training_reports or not validation_reports:
    raise SystemExit(
        "Experiment 003 diagnosis failed: expected loss reports are missing."
    )

first_nonfinite_train = next(
    (
        report
        for report in training_reports
        if not math.isfinite(report["loss"])
    ),
    None,
)
first_nonfinite_valid = next(
    (
        report
        for report in validation_reports
        if not math.isfinite(report["loss"])
    ),
    None,
)
finite_training_reports = [
    report for report in training_reports if math.isfinite(report["loss"])
]
if first_nonfinite_train is None or first_nonfinite_valid is None:
    raise SystemExit(
        "Experiment 003 diagnosis failed: non-finite loss was not found."
    )

bad_tensors: list[dict[str, object]] = []
tensor_count = 0
value_count = 0
nonfinite_value_count = 0
with safe_open(str(ADAPTER_FILE), framework="np") as handle:
    for name in handle.keys():
        tensor = handle.get_tensor(name)
        tensor_count += 1
        value_count += tensor.size
        nonfinite_count = int((~np.isfinite(tensor)).sum())
        nonfinite_value_count += nonfinite_count
        if nonfinite_count:
            bad_tensors.append(
                {
                    "name": name,
                    "values": tensor.size,
                    "nonfinite_values": nonfinite_count,
                }
            )

if not bad_tensors:
    raise SystemExit(
        "Experiment 003 diagnosis failed: adapter tensors were unexpectedly "
        "finite."
    )

minimum_finite_report = min(
    finite_training_reports,
    key=lambda report: report["loss"],
)
last_finite_report = finite_training_reports[-1]
result = {
    "name": "exp-003-balanced-1925",
    "status": "failed_numerical_instability",
    "failure_reason": (
        "Training loss diverged and became non-finite; the saved adapter "
        "contains non-finite values and must not be used for inference."
    ),
    "artifact_usable": False,
    "training_rows": data_manifest["train_rows"],
    "validation_rows": data_manifest["valid_rows"],
    "config_sha256": sha256(CONFIG_PATH),
    "training_data_sha256": data_manifest["files"]["train.jsonl"]["sha256"],
    "validation_data_sha256": data_manifest["files"]["valid.jsonl"]["sha256"],
    "training_log_sha256": sha256(LOG_PATH),
    "failed_adapter_sha256": sha256(ADAPTER_FILE),
    "minimum_finite_training_report": minimum_finite_report,
    "last_finite_training_report": last_finite_report,
    "first_nonfinite_training_report": json_safe_report(
        first_nonfinite_train
    ),
    "first_nonfinite_validation_report": json_safe_report(
        first_nonfinite_valid
    ),
    "peak_memory_gb": max(
        report["peak_memory_gb"] for report in training_reports
    ),
    "adapter_tensor_count": tensor_count,
    "adapter_value_count": value_count,
    "nonfinite_tensor_count": len(bad_tensors),
    "nonfinite_value_count": nonfinite_value_count,
    "all_adapter_values_nonfinite": nonfinite_value_count == value_count,
}
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — Experiment 003 failure diagnosis")
print(f"status: {result['status']}")
print(
    "minimum_finite_train_loss: "
    f"{minimum_finite_report['loss']} at iteration "
    f"{minimum_finite_report['iteration']}"
)
print(
    "last_finite_train_report: "
    f"{last_finite_report['loss']} at iteration "
    f"{last_finite_report['iteration']}"
)
print(
    "first_nonfinite_train_report: iteration "
    f"{first_nonfinite_train['iteration']}"
)
print(
    "first_nonfinite_validation_report: iteration "
    f"{first_nonfinite_valid['iteration']}"
)
print(f"peak_memory_gb: {result['peak_memory_gb']}")
print(f"adapter_tensors: {tensor_count}")
print(f"nonfinite_adapter_tensors: {len(bad_tensors)}")
print(f"adapter_values: {value_count}")
print(f"nonfinite_adapter_values: {nonfinite_value_count}")
print(f"artifact_usable: {result['artifact_usable']}")
print(f"result: {RESULT_PATH}")
print("exp_003_failure_diagnosis_ok: True")
