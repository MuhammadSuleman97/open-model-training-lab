#!/usr/bin/env python3
"""Run the batch-size-7 probe for the full BANKING77 SFT dataset."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "exp-004-full-data-batch7-probe.yaml"
)
SFT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"
)
TOKEN_REPORT_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "tokenization_report.json"
)
EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "exp-004-full-data-batch7-probe"
)
LOG_PATH = EXPERIMENT_DIR / "training.log"
RESULT_PATH = EXPERIMENT_DIR / "result.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
sft_manifest = json.loads(SFT_MANIFEST_PATH.read_text(encoding="utf-8"))
token_report = json.loads(TOKEN_REPORT_PATH.read_text(encoding="utf-8"))

expected = {
    "train": True,
    "fine_tune_type": "lora",
    "data": "data/banking77/sft",
    "seed": 3409,
    "batch_size": 7,
    "iters": 10,
    "val_batches": 5,
    "learning_rate": 3.5e-6,
    "max_seq_length": 576,
    "mask_prompt": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"Experiment 004 probe failed: {key} must be {value!r}."
        )

train_rows = int(sft_manifest["train_rows"])
valid_rows = int(sft_manifest["valid_rows"])
batch_size = int(config["batch_size"])
if train_rows % batch_size != 0 or valid_rows % batch_size != 0:
    raise SystemExit(
        "Experiment 004 probe failed: batch size must divide both splits."
    )
full_epoch_updates = train_rows // batch_size
full_validation_batches = valid_rows // batch_size
if token_report["recommended_max_seq_length"] != config["max_seq_length"]:
    raise SystemExit(
        "Experiment 004 probe failed: sequence length mismatch."
    )
if not token_report["mask_prompt_safe"]:
    raise SystemExit(
        "Experiment 004 probe failed: prompt masking was not verified."
    )

for name in ("train.jsonl", "valid.jsonl"):
    path = PROJECT_ROOT / config["data"] / name
    if sha256(path) != sft_manifest["files"][name]["sha256"]:
        raise SystemExit(
            f"Experiment 004 probe failed: {name} checksum mismatch."
        )

model_path = PROJECT_ROOT / config["model"]
if not model_path.is_dir():
    raise SystemExit(
        f"Experiment 004 probe failed: model not found at {model_path}."
    )

adapter_path = PROJECT_ROOT / config["adapter_path"]
adapter_file = adapter_path / "adapters.safetensors"
if adapter_file.exists():
    raise SystemExit(
        "Experiment 004 probe stopped: the adapter already exists at "
        f"{adapter_file}. Refusing to overwrite it."
    )

if "--check" in sys.argv[1:]:
    if sys.argv[1:] != ["--check"]:
        raise SystemExit("Usage: run_exp_004_probe.py [--check]")
    print("Open Model Training Lab — Experiment 004 probe preflight")
    print(f"training_rows: {train_rows}")
    print(f"validation_rows: {valid_rows}")
    print(f"batch_size: {batch_size}")
    print(f"full_epoch_updates: {full_epoch_updates}")
    print(f"full_validation_batches: {full_validation_batches}")
    print(f"learning_rate: {config['learning_rate']}")
    print("exp_004_probe_preflight_ok: True")
    raise SystemExit(0)
if sys.argv[1:]:
    raise SystemExit("Usage: run_exp_004_probe.py [--check]")

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
command = [
    str(PROJECT_ROOT / ".venv" / "bin" / "mlx_lm.lora"),
    "--config",
    str(CONFIG_PATH),
]
environment = os.environ.copy()
environment["PYTHONUNBUFFERED"] = "1"
started_at = datetime.now(timezone.utc)

print("Open Model Training Lab — Experiment 004 batch-size probe")
print("purpose: validate batch size 7 before full-data training")
print(f"training_rows: {train_rows}")
print(f"validation_rows: {valid_rows}")
print(f"batch_size: {batch_size}")
print(f"probe_iterations: {config['iters']}")
print(f"planned_full_epoch_updates: {full_epoch_updates}")
print(f"planned_full_validation_batches: {full_validation_batches}")
print(f"learning_rate: {config['learning_rate']}")
print(f"adapter_path: {adapter_path}")
print(f"log: {LOG_PATH}")
print("loading_and_training: True")

with LOG_PATH.open("w", encoding="utf-8") as log:
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    return_code = process.wait()

if return_code != 0:
    raise SystemExit(
        f"Experiment 004 probe failed with exit code {return_code}. "
        f"Output: {LOG_PATH}"
    )
if not adapter_file.is_file():
    raise SystemExit(
        "Experiment 004 probe failed: final adapter file is missing."
    )

log_text = LOG_PATH.read_text(encoding="utf-8")
validation_losses = [
    {
        "iteration": int(iteration),
        "loss": float(loss),
    }
    for iteration, loss in re.findall(
        r"Iter (\d+): Val loss ([^,]+),",
        log_text,
    )
]
training_reports = [
    {
        "iteration": int(iteration),
        "loss": float(loss),
        "peak_memory_gb": float(peak_memory),
    }
    for iteration, loss, peak_memory in re.findall(
        r"Iter (\d+): Train loss ([^,]+),.*?Peak mem ([0-9.]+) GB",
        log_text,
    )
]
all_losses = [
    report["loss"] for report in validation_losses + training_reports
]
if len(validation_losses) != 2 or not all(
    math.isfinite(loss) for loss in all_losses
):
    raise SystemExit(
        "Experiment 004 probe failed: loss reports are incomplete or "
        "non-finite."
    )

nonfinite_adapter_values = 0
adapter_value_count = 0
with safe_open(str(adapter_file), framework="np") as handle:
    for name in handle.keys():
        tensor = handle.get_tensor(name)
        adapter_value_count += tensor.size
        nonfinite_adapter_values += int((~np.isfinite(tensor)).sum())
if nonfinite_adapter_values:
    raise SystemExit(
        "Experiment 004 probe failed: adapter contains non-finite values."
    )

finished_at = datetime.now(timezone.utc)
result = {
    "name": "exp-004-full-data-batch7-probe",
    "purpose": "Batch-size and memory probe before full-data training",
    "status": "complete",
    "started_at_utc": started_at.isoformat(),
    "finished_at_utc": finished_at.isoformat(),
    "elapsed_seconds": round(
        (finished_at - started_at).total_seconds(),
        3,
    ),
    "training_rows_available": train_rows,
    "validation_rows_available": valid_rows,
    "batch_size": batch_size,
    "probe_iterations": config["iters"],
    "planned_full_epoch_updates": full_epoch_updates,
    "planned_full_validation_batches": full_validation_batches,
    "learning_rate": config["learning_rate"],
    "config_sha256": sha256(CONFIG_PATH),
    "training_data_sha256": sft_manifest["files"]["train.jsonl"]["sha256"],
    "validation_data_sha256": sft_manifest["files"]["valid.jsonl"]["sha256"],
    "training_log_sha256": sha256(LOG_PATH),
    "adapter_sha256": sha256(adapter_file),
    "adapter_bytes": adapter_file.stat().st_size,
    "adapter_value_count": adapter_value_count,
    "nonfinite_adapter_values": nonfinite_adapter_values,
    "validation_losses": validation_losses,
    "final_training_report": training_reports[-1],
    "peak_memory_gb": max(
        report["peak_memory_gb"] for report in training_reports
    ),
}
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)

print(f"validation_losses: {validation_losses}")
print(f"final_train_loss: {training_reports[-1]['loss']}")
print(f"peak_memory_gb: {result['peak_memory_gb']}")
print(f"adapter_values: {adapter_value_count}")
print("nonfinite_adapter_values: 0")
print(f"result: {RESULT_PATH}")
print("exp_004_probe_ok: True")
