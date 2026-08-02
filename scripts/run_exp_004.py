#!/usr/bin/env python3
"""Train one stable LoRA epoch on all 9,233 SFT records."""

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
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-004-full-data.yaml"
SFT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"
)
PROBE_RESULT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "exp-004-full-data-batch7-probe"
    / "result.json"
)
TOKEN_REPORT_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "tokenization_report.json"
)
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "exp-004-full-data"
LOG_PATH = EXPERIMENT_DIR / "training.log"
RESULT_PATH = EXPERIMENT_DIR / "result.json"

LOSS_ABORT_THRESHOLD = 5.0
LOSS_ABORT_AFTER_ITERATION = 200


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
sft_manifest = json.loads(SFT_MANIFEST_PATH.read_text(encoding="utf-8"))
probe_result = json.loads(PROBE_RESULT_PATH.read_text(encoding="utf-8"))
token_report = json.loads(TOKEN_REPORT_PATH.read_text(encoding="utf-8"))

expected = {
    "train": True,
    "fine_tune_type": "lora",
    "data": "data/banking77/sft",
    "seed": 3409,
    "batch_size": 7,
    "iters": 1319,
    "val_batches": 110,
    "learning_rate": 3.5e-6,
    "max_seq_length": 576,
    "mask_prompt": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"Experiment 004 setup failed: {key} must be {value!r}."
        )

if probe_result["status"] != "complete":
    raise SystemExit(
        "Experiment 004 setup failed: batch-size probe is incomplete."
    )
if probe_result["nonfinite_adapter_values"] != 0:
    raise SystemExit(
        "Experiment 004 setup failed: probe adapter was non-finite."
    )

train_rows = int(sft_manifest["train_rows"])
valid_rows = int(sft_manifest["valid_rows"])
batch_size = int(config["batch_size"])
if train_rows // batch_size != config["iters"]:
    raise SystemExit(
        "Experiment 004 setup failed: configuration is not one full epoch."
    )
if train_rows % batch_size != 0:
    raise SystemExit(
        "Experiment 004 setup failed: training rows do not divide evenly."
    )
if valid_rows // batch_size != config["val_batches"]:
    raise SystemExit(
        "Experiment 004 setup failed: validation does not cover all rows."
    )
if valid_rows % batch_size != 0:
    raise SystemExit(
        "Experiment 004 setup failed: validation rows do not divide evenly."
    )
if token_report["recommended_max_seq_length"] != config["max_seq_length"]:
    raise SystemExit(
        "Experiment 004 setup failed: sequence length mismatch."
    )
if not token_report["mask_prompt_safe"]:
    raise SystemExit(
        "Experiment 004 setup failed: prompt masking was not verified."
    )

for name in ("train.jsonl", "valid.jsonl"):
    path = PROJECT_ROOT / config["data"] / name
    if sha256(path) != sft_manifest["files"][name]["sha256"]:
        raise SystemExit(
            f"Experiment 004 setup failed: {name} checksum mismatch."
        )

model_path = PROJECT_ROOT / config["model"]
if not model_path.is_dir():
    raise SystemExit(
        f"Experiment 004 setup failed: model not found at {model_path}."
    )

adapter_path = PROJECT_ROOT / config["adapter_path"]
adapter_file = adapter_path / "adapters.safetensors"
if adapter_file.exists():
    raise SystemExit(
        "Experiment 004 stopped: the final adapter already exists at "
        f"{adapter_file}. Refusing to overwrite it."
    )

if "--check" in sys.argv[1:]:
    if sys.argv[1:] != ["--check"]:
        raise SystemExit("Usage: run_exp_004.py [--check]")
    print("Open Model Training Lab — Experiment 004 preflight")
    print(f"training_rows: {train_rows}")
    print(f"validation_rows: {valid_rows}")
    print(f"batch_size: {batch_size}")
    print(f"one_epoch_updates: {config['iters']}")
    print(f"full_validation_batches: {config['val_batches']}")
    print(f"learning_rate: {config['learning_rate']}")
    print(f"probe_peak_memory_gb: {probe_result['peak_memory_gb']}")
    print(
        "loss_safety_abort: "
        f">{LOSS_ABORT_THRESHOLD} after iteration "
        f"{LOSS_ABORT_AFTER_ITERATION}, or any non-finite loss"
    )
    print("exp_004_preflight_ok: True")
    raise SystemExit(0)
if sys.argv[1:]:
    raise SystemExit("Usage: run_exp_004.py [--check]")

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
command = [
    str(PROJECT_ROOT / ".venv" / "bin" / "mlx_lm.lora"),
    "--config",
    str(CONFIG_PATH),
]
environment = os.environ.copy()
environment["PYTHONUNBUFFERED"] = "1"
started_at = datetime.now(timezone.utc)

print("Open Model Training Lab — Experiment 004 full-data training")
print("purpose: one stable LoRA epoch over all 9,233 training records")
print(f"training_rows: {train_rows}")
print(f"validation_rows: {valid_rows}")
print(f"batch_size: {batch_size}")
print(f"iterations: {config['iters']}")
print(f"validation_batches: {config['val_batches']}")
print(f"learning_rate: {config['learning_rate']}")
print(
    "loss_safety_abort: "
    f">{LOSS_ABORT_THRESHOLD} after iteration "
    f"{LOSS_ABORT_AFTER_ITERATION}, or any non-finite loss"
)
print(f"adapter_path: {adapter_path}")
print(f"log: {LOG_PATH}")
print("loading_and_training: True")

abort_reason = None
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

        match = re.search(r"Iter (\d+): Train loss ([^,]+),", line)
        if match:
            iteration = int(match.group(1))
            loss = float(match.group(2).lower())
            unsafe = not math.isfinite(loss) or (
                iteration > LOSS_ABORT_AFTER_ITERATION
                and loss > LOSS_ABORT_THRESHOLD
            )
            if unsafe:
                abort_reason = (
                    f"unsafe reported train loss {loss!r} at iteration "
                    f"{iteration}"
                )
                process.terminate()
                break

    if abort_reason is not None:
        remainder = process.stdout.read()
        if remainder:
            print(remainder, end="")
            log.write(remainder)
        process.wait()
    else:
        return_code = process.wait()

if abort_reason is not None:
    raise SystemExit(
        "Experiment 004 safety stop triggered: "
        f"{abort_reason}. No final adapter was saved. Output: {LOG_PATH}"
    )
if return_code != 0:
    raise SystemExit(
        f"Experiment 004 failed with exit code {return_code}. "
        f"The complete output is saved at {LOG_PATH}."
    )
if not adapter_file.is_file():
    raise SystemExit(
        "Experiment 004 failed: final adapter file is missing."
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
if len(validation_losses) != 3 or not all(
    math.isfinite(loss) for loss in all_losses
):
    raise SystemExit(
        "Experiment 004 result capture failed: expected three finite "
        "validation measurements and finite training reports."
    )
if not training_reports or training_reports[-1]["iteration"] != config["iters"]:
    raise SystemExit(
        "Experiment 004 result capture failed: final training report missing."
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
        "Experiment 004 result capture failed: adapter is non-finite."
    )

finished_at = datetime.now(timezone.utc)
result = {
    "name": "exp-004-full-data",
    "purpose": "One stable LoRA epoch over all 9,233 SFT training records",
    "status": "complete",
    "parent_probe": "exp-004-full-data-batch7-probe",
    "safety_abort": {
        "after_iteration": LOSS_ABORT_AFTER_ITERATION,
        "train_loss_threshold": LOSS_ABORT_THRESHOLD,
        "triggered": False,
    },
    "started_at_utc": started_at.isoformat(),
    "finished_at_utc": finished_at.isoformat(),
    "elapsed_seconds": round(
        (finished_at - started_at).total_seconds(),
        3,
    ),
    "training_rows": train_rows,
    "validation_rows": valid_rows,
    "batch_size": batch_size,
    "iterations": config["iters"],
    "epochs_equivalent": 1.0,
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
print(f"adapter_sha256: {result['adapter_sha256']}")
print(f"elapsed_seconds: {result['elapsed_seconds']}")
print("exp_004_training_ok: True")
