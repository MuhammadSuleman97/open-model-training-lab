#!/usr/bin/env python3
"""Run the lower-learning-rate retry of Experiment 003."""

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

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "exp-003b-balanced-1925-lr2p5e-6.yaml"
)
DATA_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "exp_003_data_manifest.json"
)
FAILED_RESULT_PATH = (
    PROJECT_ROOT / "experiments" / "exp-003-balanced-1925" / "result.json"
)
TOKEN_REPORT_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "tokenization_report.json"
)
EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "exp-003b-balanced-1925-lr2p5e-6"
)
LOG_PATH = EXPERIMENT_DIR / "training.log"
RESULT_PATH = EXPERIMENT_DIR / "result.json"

LOSS_ABORT_THRESHOLD = 5.0
LOSS_ABORT_AFTER_ITERATION = 100


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
failed_result = json.loads(FAILED_RESULT_PATH.read_text(encoding="utf-8"))
token_report = json.loads(TOKEN_REPORT_PATH.read_text(encoding="utf-8"))

expected = {
    "train": True,
    "fine_tune_type": "lora",
    "data": "data/banking77/exp_003_balanced_1925",
    "seed": 3408,
    "batch_size": 1,
    "iters": 1925,
    "val_batches": 154,
    "learning_rate": 2.5e-6,
    "max_seq_length": 576,
    "mask_prompt": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"Experiment 003b setup failed: {key} must be {value!r}."
        )

if failed_result["status"] != "failed_numerical_instability":
    raise SystemExit(
        "Experiment 003b setup failed: parent failure was not recorded."
    )
if failed_result["artifact_usable"]:
    raise SystemExit(
        "Experiment 003b setup failed: failed parent marked as usable."
    )
if data_manifest["train_rows"] != config["iters"]:
    raise SystemExit(
        "Experiment 003b setup failed: iterations must equal training rows."
    )
if data_manifest["valid_rows"] != config["val_batches"]:
    raise SystemExit(
        "Experiment 003b setup failed: validation must cover every record."
    )
if token_report["recommended_max_seq_length"] != config["max_seq_length"]:
    raise SystemExit(
        "Experiment 003b setup failed: sequence length mismatch."
    )
if not token_report["mask_prompt_safe"]:
    raise SystemExit(
        "Experiment 003b setup failed: prompt masking was not verified."
    )

for name in ("train.jsonl", "valid.jsonl"):
    path = PROJECT_ROOT / config["data"] / name
    if sha256(path) != data_manifest["files"][name]["sha256"]:
        raise SystemExit(
            f"Experiment 003b setup failed: {name} checksum mismatch."
        )

model_path = PROJECT_ROOT / config["model"]
if not model_path.is_dir():
    raise SystemExit(
        f"Experiment 003b setup failed: model not found at {model_path}."
    )

adapter_path = PROJECT_ROOT / config["adapter_path"]
adapter_file = adapter_path / "adapters.safetensors"
if adapter_file.exists():
    raise SystemExit(
        "Experiment 003b stopped: the final adapter already exists at "
        f"{adapter_file}. Refusing to overwrite it."
    )

if "--check" in sys.argv[1:]:
    if sys.argv[1:] != ["--check"]:
        raise SystemExit("Usage: run_exp_003b.py [--check]")
    print("Open Model Training Lab — Experiment 003b preflight")
    print(f"model: {model_path}")
    print(f"training_rows: {data_manifest['train_rows']}")
    print(f"validation_rows: {data_manifest['valid_rows']}")
    print("parent_failure_recorded: True")
    print(f"learning_rate: {config['learning_rate']}")
    print(
        "loss_safety_abort: "
        f">{LOSS_ABORT_THRESHOLD} after iteration "
        f"{LOSS_ABORT_AFTER_ITERATION}, or any non-finite loss"
    )
    print("exp_003b_preflight_ok: True")
    raise SystemExit(0)
if sys.argv[1:]:
    raise SystemExit("Usage: run_exp_003b.py [--check]")

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
command = [
    str(PROJECT_ROOT / ".venv" / "bin" / "mlx_lm.lora"),
    "--config",
    str(CONFIG_PATH),
]
environment = os.environ.copy()
environment["PYTHONUNBUFFERED"] = "1"
started_at = datetime.now(timezone.utc)

print("Open Model Training Lab — Experiment 003b stable retry")
print("purpose: retry identical data with a 4x lower learning rate")
print(f"training_rows: {data_manifest['train_rows']}")
print(f"validation_rows: {data_manifest['valid_rows']}")
print(f"iterations: {config['iters']}")
print(f"batch_size: {config['batch_size']}")
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
        "Experiment 003b safety stop triggered: "
        f"{abort_reason}. No final adapter was saved. Output: {LOG_PATH}"
    )
if return_code != 0:
    raise SystemExit(
        f"Experiment 003b failed with exit code {return_code}. "
        f"The complete output is saved at {LOG_PATH}."
    )
if not adapter_file.is_file():
    raise SystemExit(
        "Experiment 003b failed: MLX-LM exited successfully but the final "
        "adapter file is missing."
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
        "Experiment 003b result capture failed: expected three finite "
        "validation measurements and finite training reports."
    )
if not training_reports or training_reports[-1]["iteration"] != config["iters"]:
    raise SystemExit(
        "Experiment 003b result capture failed: final training report missing."
    )

finished_at = datetime.now(timezone.utc)
result = {
    "name": "exp-003b-balanced-1925-lr2p5e-6",
    "purpose": "Stable one-pass retry with a 4x lower learning rate",
    "status": "complete",
    "parent_failed_experiment": "exp-003-balanced-1925",
    "changed_parameter": {
        "name": "learning_rate",
        "failed_value": 1.0e-5,
        "retry_value": config["learning_rate"],
    },
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
    "training_rows": data_manifest["train_rows"],
    "validation_rows": data_manifest["valid_rows"],
    "epochs_equivalent": 1.0,
    "config_sha256": sha256(CONFIG_PATH),
    "training_data_sha256": data_manifest["files"]["train.jsonl"]["sha256"],
    "validation_data_sha256": data_manifest["files"]["valid.jsonl"]["sha256"],
    "training_log_sha256": sha256(LOG_PATH),
    "adapter_sha256": sha256(adapter_file),
    "adapter_bytes": adapter_file.stat().st_size,
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
print(f"result: {RESULT_PATH}")
print(f"adapter_sha256: {result['adapter_sha256']}")
print(f"elapsed_seconds: {result['elapsed_seconds']}")
print("exp_003b_training_ok: True")
