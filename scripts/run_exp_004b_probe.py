#!/usr/bin/env python3
"""Probe full-data training with batch size 1 and a lower learning rate."""

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
    PROJECT_ROOT / "configs" / "exp-004b-full-data-batch1-probe.yaml"
)
SFT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"
)
FAILED_RESULT_PATH = (
    PROJECT_ROOT / "experiments" / "exp-004-full-data" / "result.json"
)
TOKEN_REPORT_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "tokenization_report.json"
)
EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "exp-004b-full-data-batch1-probe"
)
LOG_PATH = EXPERIMENT_DIR / "training.log"
RESULT_PATH = EXPERIMENT_DIR / "result.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
sft_manifest = json.loads(SFT_MANIFEST_PATH.read_text(encoding="utf-8"))
failed_result = json.loads(FAILED_RESULT_PATH.read_text(encoding="utf-8"))
token_report = json.loads(TOKEN_REPORT_PATH.read_text(encoding="utf-8"))

expected = {
    "train": True,
    "data": "data/banking77/sft",
    "seed": 3409,
    "batch_size": 1,
    "iters": 50,
    "val_batches": 1,
    "learning_rate": 5.0e-7,
    "max_seq_length": 576,
    "mask_prompt": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"Experiment 004b probe failed: {key} must be {value!r}."
        )

if failed_result["status"] != "failed_early_numerical_instability":
    raise SystemExit(
        "Experiment 004b probe failed: parent failure is not recorded."
    )
if failed_result["adapter_saved"]:
    raise SystemExit(
        "Experiment 004b probe failed: parent unexpectedly saved an adapter."
    )
if token_report["recommended_max_seq_length"] != config["max_seq_length"]:
    raise SystemExit(
        "Experiment 004b probe failed: sequence length mismatch."
    )
if token_report.get("mask_prompt_safe") is not True:
    raise SystemExit(
        "Experiment 004b probe failed: prompt masking is not verified safe."
    )

model_path = PROJECT_ROOT / config["model"]
if not model_path.is_dir():
    raise SystemExit(
        f"Experiment 004b probe failed: model snapshot is missing at {model_path}."
    )
if not any(model_path.glob("*.safetensors")):
    raise SystemExit(
        "Experiment 004b probe failed: model snapshot has no weight file."
    )

for name in ("train.jsonl", "valid.jsonl"):
    path = PROJECT_ROOT / config["data"] / name
    if sha256(path) != sft_manifest["files"][name]["sha256"]:
        raise SystemExit(
            f"Experiment 004b probe failed: {name} checksum mismatch."
        )

adapter_path = PROJECT_ROOT / config["adapter_path"]
adapter_file = adapter_path / "adapters.safetensors"
if adapter_file.exists():
    raise SystemExit(
        "Experiment 004b probe stopped: adapter already exists at "
        f"{adapter_file}. Refusing to overwrite it."
    )

if "--check" in sys.argv[1:]:
    if sys.argv[1:] != ["--check"]:
        raise SystemExit("Usage: run_exp_004b_probe.py [--check]")
    print("Open Model Training Lab — Experiment 004b probe preflight")
    print(f"training_rows: {sft_manifest['train_rows']}")
    print("batch_size: 1")
    print("probe_iterations: 50")
    print("planned_full_epoch_updates: 9233")
    print(f"learning_rate: {config['learning_rate']}")
    print("exp_004b_probe_preflight_ok: True")
    raise SystemExit(0)
if sys.argv[1:]:
    raise SystemExit("Usage: run_exp_004b_probe.py [--check]")

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
command = [
    str(PROJECT_ROOT / ".venv" / "bin" / "mlx_lm.lora"),
    "--config",
    str(CONFIG_PATH),
]
environment = os.environ.copy()
environment["PYTHONUNBUFFERED"] = "1"
started_at = datetime.now(timezone.utc)

print("Open Model Training Lab — Experiment 004b stability probe")
print("purpose: test batch size 1 and 5e-7 learning rate for 50 updates")
print("batch_size: 1")
print("probe_iterations: 50")
print("planned_full_epoch_updates: 9233")
print(f"learning_rate: {config['learning_rate']}")
print(f"adapter_path: {adapter_path}")
print(f"log: {LOG_PATH}")
print("loading_and_training: True")

nonfinite_report = None
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
        if match and not math.isfinite(float(match.group(2).lower())):
            nonfinite_report = int(match.group(1))
            process.terminate()
            break

    if nonfinite_report is not None:
        remainder = process.stdout.read()
        if remainder:
            print(remainder, end="")
            log.write(remainder)
        process.wait()
    else:
        return_code = process.wait()

if nonfinite_report is not None:
    raise SystemExit(
        "Experiment 004b probe safety stop: non-finite loss at iteration "
        f"{nonfinite_report}. No final adapter was saved."
    )
if return_code != 0:
    raise SystemExit(
        f"Experiment 004b probe failed with exit code {return_code}. "
        f"Output: {LOG_PATH}"
    )
if not adapter_file.is_file():
    raise SystemExit(
        "Experiment 004b probe failed: final adapter is missing."
    )

log_text = LOG_PATH.read_text(encoding="utf-8")
validation_losses = [
    {"iteration": int(i), "loss": float(loss)}
    for i, loss in re.findall(r"Iter (\d+): Val loss ([^,]+),", log_text)
]
training_reports = [
    {
        "iteration": int(i),
        "loss": float(loss),
        "peak_memory_gb": float(memory),
    }
    for i, loss, memory in re.findall(
        r"Iter (\d+): Train loss ([^,]+),.*?Peak mem ([0-9.]+) GB",
        log_text,
    )
]
if len(validation_losses) != 2 or len(training_reports) != 50:
    raise SystemExit(
        "Experiment 004b probe failed: expected loss reports are missing."
    )
if not all(
    math.isfinite(report["loss"])
    for report in validation_losses + training_reports
):
    raise SystemExit(
        "Experiment 004b probe failed: a recorded loss is non-finite."
    )

adapter_values = 0
nonfinite_adapter_values = 0
with safe_open(str(adapter_file), framework="np") as handle:
    for name in handle.keys():
        tensor = handle.get_tensor(name)
        adapter_values += tensor.size
        nonfinite_adapter_values += int((~np.isfinite(tensor)).sum())
if nonfinite_adapter_values:
    raise SystemExit(
        "Experiment 004b probe failed: adapter contains non-finite values."
    )

finished_at = datetime.now(timezone.utc)
result = {
    "name": "exp-004b-full-data-batch1-probe",
    "purpose": "Fifty-update stability probe after batch-size-7 failure",
    "status": "complete",
    "parent_failed_experiment": "exp-004-full-data",
    "batch_size": 1,
    "probe_iterations": 50,
    "planned_full_epoch_updates": sft_manifest["train_rows"],
    "learning_rate": config["learning_rate"],
    "started_at_utc": started_at.isoformat(),
    "finished_at_utc": finished_at.isoformat(),
    "elapsed_seconds": round(
        (finished_at - started_at).total_seconds(),
        3,
    ),
    "config_sha256": sha256(CONFIG_PATH),
    "training_data_sha256": sft_manifest["files"]["train.jsonl"]["sha256"],
    "validation_data_sha256": sft_manifest["files"]["valid.jsonl"]["sha256"],
    "training_log_sha256": sha256(LOG_PATH),
    "adapter_sha256": sha256(adapter_file),
    "adapter_bytes": adapter_file.stat().st_size,
    "adapter_value_count": adapter_values,
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
print(f"adapter_values: {adapter_values}")
print("nonfinite_adapter_values: 0")
print(f"result: {RESULT_PATH}")
print("exp_004b_probe_ok: True")
