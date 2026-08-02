#!/usr/bin/env python3
"""Retry q/k/v/o LoRA training at half the failed learning rate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "exp-005b-attention-qkvo-lr2p5e-7.yaml"
)
SFT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"
)
TOKEN_REPORT_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "tokenization_report.json"
)
FAILED_RESULT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "exp-005-attention-qkvo-full"
    / "result.json"
)
EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "exp-005b-attention-qkvo-lr2p5e-7"
)
LOG_PATH = EXPERIMENT_DIR / "training.log"
RESULT_PATH = EXPERIMENT_DIR / "result.json"
EXPECTED_ADAPTER_VALUES = 1_835_008


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
sft_manifest = json.loads(SFT_MANIFEST_PATH.read_text(encoding="utf-8"))
token_report = json.loads(TOKEN_REPORT_PATH.read_text(encoding="utf-8"))
failed_result = json.loads(FAILED_RESULT_PATH.read_text(encoding="utf-8"))

targets = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]
expected = {
    "train": True,
    "fine_tune_type": "lora",
    "optimizer": "adam",
    "data": "data/banking77/sft",
    "seed": 3409,
    "num_layers": 16,
    "batch_size": 1,
    "iters": 9233,
    "val_batches": 770,
    "learning_rate": 2.5e-7,
    "steps_per_report": 50,
    "steps_per_eval": 4617,
    "save_every": 9233,
    "max_seq_length": 576,
    "mask_prompt": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"Experiment 005b setup failed: {key} must be {value!r}."
        )
if config["lora_parameters"] != {
    "rank": 8,
    "dropout": 0.0,
    "scale": 20.0,
    "keys": targets,
}:
    raise SystemExit(
        "Experiment 005b setup failed: LoRA parameters are unexpected."
    )
if failed_result.get("status") != "failed_numerical_instability":
    raise SystemExit(
        "Experiment 005b setup failed: parent failure is not recorded."
    )
if failed_result.get("first_nonfinite_training_report", {}).get(
    "iteration"
) != 2050:
    raise SystemExit(
        "Experiment 005b setup failed: parent diagnosis is incomplete."
    )
if failed_result.get("adapter_saved"):
    raise SystemExit(
        "Experiment 005b setup failed: failed parent saved weights."
    )

train_rows = int(sft_manifest["train_rows"])
valid_rows = int(sft_manifest["valid_rows"])
if config["iters"] != train_rows or config["val_batches"] != valid_rows:
    raise SystemExit(
        "Experiment 005b setup failed: epoch coverage is not exact."
    )
if token_report.get("recommended_max_seq_length") != config["max_seq_length"]:
    raise SystemExit(
        "Experiment 005b setup failed: sequence length mismatch."
    )
if token_report.get("mask_prompt_safe") is not True:
    raise SystemExit(
        "Experiment 005b setup failed: prompt masking is not verified."
    )
for name in ("train.jsonl", "valid.jsonl"):
    path = PROJECT_ROOT / config["data"] / name
    if sha256(path) != sft_manifest["files"][name]["sha256"]:
        raise SystemExit(
            f"Experiment 005b setup failed: {name} checksum mismatch."
        )
model_path = PROJECT_ROOT / config["model"]
if not model_path.is_dir() or not any(model_path.glob("*.safetensors")):
    raise SystemExit("Experiment 005b setup failed: model is missing.")
adapter_path = PROJECT_ROOT / config["adapter_path"]
adapter_file = adapter_path / "adapters.safetensors"
if adapter_path.exists() and any(adapter_path.iterdir()):
    raise SystemExit(
        "Experiment 005b stopped: adapter directory is not empty."
    )
if RESULT_PATH.exists() or LOG_PATH.exists():
    raise SystemExit(
        "Experiment 005b stopped: experiment artifact already exists."
    )

if "--check" in sys.argv[1:]:
    if sys.argv[1:] != ["--check"]:
        raise SystemExit("Usage: run_exp_005b.py [--check]")
    print("Open Model Training Lab — Experiment 005b preflight")
    print("parent_failure_iteration: 2050")
    print("changed_factor_from_failed_run: learning_rate")
    print("failed_learning_rate: 5e-7")
    print("retry_learning_rate: 2.5e-7")
    print("unchanged_targets: q_proj, k_proj, v_proj, o_proj")
    print("unchanged_adapter_values: 1835008")
    print(f"one_epoch_updates: {config['iters']}")
    print("estimated_runtime: approximately 1.5 to 2 hours")
    print("exp_005b_preflight_ok: True")
    raise SystemExit(0)
if sys.argv[1:]:
    raise SystemExit("Usage: run_exp_005b.py [--check]")

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
command = [
    str(PROJECT_ROOT / ".venv" / "bin" / "mlx_lm.lora"),
    "--config",
    str(CONFIG_PATH),
]
environment = os.environ.copy()
environment["PYTHONUNBUFFERED"] = "1"
started_at = datetime.now(timezone.utc)

print("Open Model Training Lab — Experiment 005b stable retry")
print("purpose: retry q/k/v/o LoRA at half the failed learning rate")
print(f"training_rows: {train_rows}")
print(f"validation_rows: {valid_rows}")
print("batch_size: 1")
print(f"iterations: {config['iters']}")
print("learning_rate: 2.5e-7")
print("safety_stop: any non-finite train or validation loss")
print(f"adapter_path: {adapter_path}")
print(f"log: {LOG_PATH}")
print("loading_and_training: True")

abort_reason = None
interrupted = False
return_code = None
with LOG_PATH.open("w", encoding="utf-8") as log:
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
            match = re.search(
                r"Iter (\d+): (Train|Val) loss ([^,]+),",
                line,
            )
            if match and not math.isfinite(float(match.group(3).lower())):
                abort_reason = (
                    f"non-finite {match.group(2).lower()} loss at "
                    f"iteration {match.group(1)}"
                )
                stop_process_group(process)
                break
    except KeyboardInterrupt:
        interrupted = True
        stop_process_group(process)
    if abort_reason is not None or interrupted:
        remainder = process.stdout.read()
        if remainder:
            print(remainder, end="")
            log.write(remainder)
    else:
        return_code = process.wait()

finished_at = datetime.now(timezone.utc)
if interrupted:
    archive_tag = started_at.strftime("%Y%m%dT%H%M%SZ")
    archived_experiment_dir = EXPERIMENT_DIR.with_name(
        f"{EXPERIMENT_DIR.name}-interrupted-{archive_tag}"
    )
    archived_adapter_path = adapter_path.with_name(
        f"{adapter_path.name}-interrupted-{archive_tag}"
    )
    completed_reports = len(
        re.findall(
            r"Iter \d+: Train loss [^,]+,",
            LOG_PATH.read_text(encoding="utf-8"),
        )
    )
    interrupted_result = {
        "name": "exp-005b-attention-qkvo-lr2p5e-7",
        "status": "interrupted_by_user",
        "artifact_usable": False,
        "safe_to_restart_from_scratch": True,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": round(
            (finished_at - started_at).total_seconds(),
            3,
        ),
        "training_reports_completed": completed_reports,
        "training_log_sha256": sha256(LOG_PATH),
        "adapter_saved": adapter_file.is_file(),
        "archived_experiment_dir": str(archived_experiment_dir),
        "archived_adapter_path": (
            str(archived_adapter_path) if adapter_path.exists() else None
        ),
    }
    RESULT_PATH.write_text(
        json.dumps(interrupted_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if archived_experiment_dir.exists() or archived_adapter_path.exists():
        raise SystemExit(
            "Experiment 005b was interrupted, but its archive target already "
            "exists. Do not rerun automatically."
        )
    EXPERIMENT_DIR.rename(archived_experiment_dir)
    if adapter_path.exists():
        adapter_path.rename(archived_adapter_path)
    raise SystemExit(
        "Experiment 005b interrupted. The process was stopped and partial "
        f"artifacts were archived at {archived_experiment_dir}. It is safe to "
        "restart from scratch."
    )
if abort_reason is not None:
    failure = {
        "name": "exp-005b-attention-qkvo-lr2p5e-7",
        "status": "failed_numerical_instability",
        "artifact_usable": False,
        "failure_reason": abort_reason,
        "learning_rate": config["learning_rate"],
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "training_log_sha256": sha256(LOG_PATH),
        "adapter_saved": adapter_file.is_file(),
    }
    RESULT_PATH.write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(
        f"Experiment 005b safety stop: {abort_reason}. "
        "Do not rerun automatically."
    )
if return_code != 0:
    raise SystemExit(
        f"Experiment 005b failed with exit code {return_code}. "
        f"Output: {LOG_PATH}"
    )
if not adapter_file.is_file():
    raise SystemExit("Experiment 005b failed: final adapter is missing.")

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
if [report["iteration"] for report in validation_losses] != [1, 4617, 9233]:
    raise SystemExit(
        "Experiment 005b result capture failed: validations are missing."
    )
if not all(
    math.isfinite(report["loss"])
    for report in validation_losses + training_reports
):
    raise SystemExit(
        "Experiment 005b result capture failed: a loss is non-finite."
    )
if not training_reports or training_reports[-1]["iteration"] != config["iters"]:
    raise SystemExit(
        "Experiment 005b result capture failed: final report is missing."
    )

adapter_value_count = 0
nonfinite_adapter_values = 0
with safe_open(str(adapter_file), framework="np") as handle:
    for name in handle.keys():
        tensor = handle.get_tensor(name)
        adapter_value_count += tensor.size
        nonfinite_adapter_values += int((~np.isfinite(tensor)).sum())
if adapter_value_count != EXPECTED_ADAPTER_VALUES:
    raise SystemExit(
        "Experiment 005b result capture failed: capacity is unexpected."
    )
if nonfinite_adapter_values:
    raise SystemExit(
        "Experiment 005b result capture failed: adapter is non-finite."
    )

result = {
    "name": "exp-005b-attention-qkvo-lr2p5e-7",
    "purpose": "Numerically stable q/k/v/o learning-rate retry",
    "status": "complete",
    "parent_failed_experiment": "exp-005-attention-qkvo-full",
    "changed_parameter": {
        "name": "learning_rate",
        "failed_value": 5.0e-7,
        "retry_value": config["learning_rate"],
    },
    "started_at_utc": started_at.isoformat(),
    "finished_at_utc": finished_at.isoformat(),
    "elapsed_seconds": round(
        (finished_at - started_at).total_seconds(),
        3,
    ),
    "training_rows": train_rows,
    "validation_rows": valid_rows,
    "batch_size": config["batch_size"],
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
print("exp_005b_training_ok: True")
