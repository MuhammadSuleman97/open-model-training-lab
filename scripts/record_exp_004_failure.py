#!/usr/bin/env python3
"""Record Experiment 004's early non-finite-loss safety stop."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-004-full-data.yaml"
SFT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"
)
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "exp-004-full-data"
LOG_PATH = EXPERIMENT_DIR / "training.log"
RESULT_PATH = EXPERIMENT_DIR / "result.json"
ADAPTER_FILE = (
    PROJECT_ROOT / "adapters" / "exp-004-full-data" / "adapters.safetensors"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


sft_manifest = json.loads(SFT_MANIFEST_PATH.read_text(encoding="utf-8"))
log_text = LOG_PATH.read_text(encoding="utf-8")
validation_match = re.search(r"Iter (\d+): Val loss ([^,]+),", log_text)
training_match = re.search(
    r"Iter (\d+): Train loss ([^,]+),.*?Peak mem ([0-9.]+) GB",
    log_text,
)
if validation_match is None or training_match is None:
    raise SystemExit(
        "Experiment 004 failure recording failed: loss reports missing."
    )
if training_match.group(2).lower() != "nan":
    raise SystemExit(
        "Experiment 004 failure recording failed: expected NaN loss."
    )
if ADAPTER_FILE.exists():
    raise SystemExit(
        "Experiment 004 failure recording failed: unexpected adapter exists."
    )

result = {
    "name": "exp-004-full-data",
    "status": "failed_early_numerical_instability",
    "failure_reason": (
        "Batch-size-7 training produced non-finite reported loss by "
        "iteration 50. The safety guard terminated training before an "
        "adapter was saved."
    ),
    "artifact_usable": False,
    "adapter_saved": False,
    "training_rows_available": sft_manifest["train_rows"],
    "validation_rows_available": sft_manifest["valid_rows"],
    "batch_size": 7,
    "learning_rate": 3.5e-6,
    "config_sha256": sha256(CONFIG_PATH),
    "training_data_sha256": sft_manifest["files"]["train.jsonl"]["sha256"],
    "validation_data_sha256": sft_manifest["files"]["valid.jsonl"]["sha256"],
    "training_log_sha256": sha256(LOG_PATH),
    "initial_validation_report": {
        "iteration": int(validation_match.group(1)),
        "loss": float(validation_match.group(2)),
    },
    "first_training_report": {
        "iteration": int(training_match.group(1)),
        "loss": "nan",
        "peak_memory_gb": float(training_match.group(3)),
    },
    "safety_stop_triggered": True,
}
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — Experiment 004 failure record")
print(f"status: {result['status']}")
print("first_nonfinite_training_report: iteration 50")
print(f"peak_memory_gb: {result['first_training_report']['peak_memory_gb']}")
print("adapter_saved: False")
print("artifact_usable: False")
print(f"result: {RESULT_PATH}")
print("exp_004_failure_recorded: True")
