#!/usr/bin/env python3
"""Probe LoRA training with q/k/v/o attention targets."""

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
    PROJECT_ROOT / "configs" / "exp-005-attention-qkvo-probe.yaml"
)
SFT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"
)
TOKEN_REPORT_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "tokenization_report.json"
)
PARENT_RESULT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "exp-004c-full-data-batch1-lr5e-7"
    / "result.json"
)
PARENT_ADAPTER_CONFIG_PATH = (
    PROJECT_ROOT
    / "adapters"
    / "exp-004c-full-data-batch1-lr5e-7"
    / "adapter_config.json"
)
EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "exp-005-attention-qkvo-probe"
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
parent_result = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
parent_adapter_config = json.loads(
    PARENT_ADAPTER_CONFIG_PATH.read_text(encoding="utf-8")
)

expected = {
    "train": True,
    "fine_tune_type": "lora",
    "optimizer": "adam",
    "data": "data/banking77/sft",
    "seed": 3409,
    "num_layers": 16,
    "batch_size": 1,
    "iters": 50,
    "val_batches": 1,
    "learning_rate": 5.0e-7,
    "steps_per_report": 1,
    "steps_per_eval": 50,
    "max_seq_length": 576,
    "mask_prompt": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"Experiment 005 probe failed: {key} must be {value!r}."
        )

baseline_targets = ["self_attn.q_proj", "self_attn.v_proj"]
probe_targets = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]
if parent_adapter_config["lora_parameters"]["keys"] != baseline_targets:
    raise SystemExit(
        "Experiment 005 probe failed: parent LoRA targets are unexpected."
    )
if config["lora_parameters"]["keys"] != probe_targets:
    raise SystemExit(
        "Experiment 005 probe failed: q/k/v/o targets are not exact."
    )
for key in (
    "rank",
    "dropout",
    "scale",
):
    if (
        config["lora_parameters"][key]
        != parent_adapter_config["lora_parameters"][key]
    ):
        raise SystemExit(
            f"Experiment 005 probe failed: LoRA {key} changed unexpectedly."
        )
if parent_result.get("status") != "complete":
    raise SystemExit(
        "Experiment 005 probe failed: Experiment 004c is incomplete."
    )
if parent_result.get("nonfinite_adapter_values") != 0:
    raise SystemExit(
        "Experiment 005 probe failed: parent adapter is non-finite."
    )
if token_report.get("recommended_max_seq_length") != config["max_seq_length"]:
    raise SystemExit(
        "Experiment 005 probe failed: sequence length mismatch."
    )
if token_report.get("mask_prompt_safe") is not True:
    raise SystemExit(
        "Experiment 005 probe failed: prompt masking is not verified safe."
    )

for name in ("train.jsonl", "valid.jsonl"):
    path = PROJECT_ROOT / config["data"] / name
    if sha256(path) != sft_manifest["files"][name]["sha256"]:
        raise SystemExit(
            f"Experiment 005 probe failed: {name} checksum mismatch."
        )

model_path = PROJECT_ROOT / config["model"]
if not model_path.is_dir() or not any(model_path.glob("*.safetensors")):
    raise SystemExit(
        "Experiment 005 probe failed: pinned model snapshot is missing."
    )
adapter_path = PROJECT_ROOT / config["adapter_path"]
adapter_file = adapter_path / "adapters.safetensors"
if adapter_path.exists() and any(adapter_path.iterdir()):
    raise SystemExit(
        "Experiment 005 probe stopped: adapter directory is not empty."
    )
if RESULT_PATH.exists() or LOG_PATH.exists():
    raise SystemExit(
        "Experiment 005 probe stopped: experiment artifact already exists."
    )

if "--check" in sys.argv[1:]:
    if sys.argv[1:] != ["--check"]:
        raise SystemExit("Usage: run_exp_005_probe.py [--check]")
    print("Open Model Training Lab — Experiment 005 probe preflight")
    print("changed_factor: LoRA attention targets")
    print("baseline_targets: q_proj, v_proj")
    print("probe_targets: q_proj, k_proj, v_proj, o_proj")
    print("unchanged_rank: 8")
    print("unchanged_layers: final 16 of 28")
    print("unchanged_batch_size: 1")
    print("unchanged_learning_rate: 5e-7")
    print("probe_iterations: 50")
    print(f"expected_adapter_values: {EXPECTED_ADAPTER_VALUES}")
    print("exp_005_probe_preflight_ok: True")
    raise SystemExit(0)
if sys.argv[1:]:
    raise SystemExit("Usage: run_exp_005_probe.py [--check]")

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
command = [
    str(PROJECT_ROOT / ".venv" / "bin" / "mlx_lm.lora"),
    "--config",
    str(CONFIG_PATH),
]
environment = os.environ.copy()
environment["PYTHONUNBUFFERED"] = "1"
started_at = datetime.now(timezone.utc)

print("Open Model Training Lab — Experiment 005 q/k/v/o probe")
print("changed_factor: LoRA attention targets only")
print("baseline_targets: q_proj, v_proj")
print("probe_targets: q_proj, k_proj, v_proj, o_proj")
print("probe_iterations: 50")
print(f"learning_rate: {config['learning_rate']}")
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

if interrupted:
    raise SystemExit(
        "Experiment 005 probe interrupted. Partial artifacts remain protected; "
        "do not rerun automatically."
    )
if abort_reason is not None:
    raise SystemExit(
        f"Experiment 005 probe safety stop: {abort_reason}. "
        "Do not rerun automatically."
    )
if return_code != 0:
    raise SystemExit(
        f"Experiment 005 probe failed with exit code {return_code}. "
        f"Output: {LOG_PATH}"
    )
if not adapter_file.is_file():
    raise SystemExit(
        "Experiment 005 probe failed: final adapter is missing."
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
        "Experiment 005 probe failed: expected loss reports are missing."
    )
if not all(
    math.isfinite(report["loss"])
    for report in validation_losses + training_reports
):
    raise SystemExit(
        "Experiment 005 probe failed: a reported loss is non-finite."
    )

adapter_values = 0
nonfinite_adapter_values = 0
with safe_open(str(adapter_file), framework="np") as handle:
    for name in handle.keys():
        tensor = handle.get_tensor(name)
        adapter_values += tensor.size
        nonfinite_adapter_values += int((~np.isfinite(tensor)).sum())
if adapter_values != EXPECTED_ADAPTER_VALUES:
    raise SystemExit(
        "Experiment 005 probe failed: adapter capacity is unexpected."
    )
if nonfinite_adapter_values:
    raise SystemExit(
        "Experiment 005 probe failed: adapter contains non-finite values."
    )

finished_at = datetime.now(timezone.utc)
result = {
    "name": "exp-005-attention-qkvo-probe",
    "purpose": "Capacity and stability probe for q/k/v/o LoRA targets",
    "status": "complete",
    "parent_experiment": "exp-004c-full-data-batch1-lr5e-7",
    "changed_factor": {
        "name": "lora_attention_targets",
        "baseline": baseline_targets,
        "probe": probe_targets,
    },
    "probe_iterations": config["iters"],
    "batch_size": config["batch_size"],
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
print("exp_005_probe_ok: True")
