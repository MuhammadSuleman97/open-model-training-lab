#!/usr/bin/env python3
"""Validate and run the first small MLX-LM LoRA pipeline experiment."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-001-pipeline.yaml"
SFT_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"
TOKEN_REPORT_PATH = (
    PROJECT_ROOT / "data" / "banking77" / "tokenization_report.json"
)
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "exp-001-pipeline"
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
    "seed": 3407,
    "batch_size": 1,
    "iters": 10,
    "max_seq_length": 576,
    "mask_prompt": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"LoRA pipeline setup failed: {key} must be {value!r}."
        )

if token_report["recommended_max_seq_length"] != config["max_seq_length"]:
    raise SystemExit(
        "LoRA pipeline setup failed: configured sequence length does not "
        "match the tokenization report."
    )
if not token_report["mask_prompt_safe"]:
    raise SystemExit(
        "LoRA pipeline setup failed: prompt masking was not verified."
    )

for name in ("train.jsonl", "valid.jsonl"):
    path = PROJECT_ROOT / config["data"] / name
    if sha256(path) != sft_manifest["files"][name]["sha256"]:
        raise SystemExit(
            f"LoRA pipeline setup failed: {name} checksum mismatch."
        )

model_path = PROJECT_ROOT / config["model"]
if not model_path.is_dir():
    raise SystemExit(
        f"LoRA pipeline setup failed: model not found at {model_path}."
    )

adapter_path = PROJECT_ROOT / config["adapter_path"]
adapter_file = adapter_path / "adapters.safetensors"
if adapter_file.exists():
    raise SystemExit(
        "LoRA pipeline setup stopped: the final adapter already exists at "
        f"{adapter_file}. Refusing to overwrite it."
    )

if "--check" in sys.argv[1:]:
    if sys.argv[1:] != ["--check"]:
        raise SystemExit("Usage: run_lora_pipeline.py [--check]")
    print("Open Model Training Lab — LoRA pipeline preflight")
    print(f"model: {model_path}")
    print(f"data: {PROJECT_ROOT / config['data']}")
    print(f"max_seq_length: {config['max_seq_length']}")
    print("lora_pipeline_preflight_ok: True")
    raise SystemExit(0)
if sys.argv[1:]:
    raise SystemExit("Usage: run_lora_pipeline.py [--check]")

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
command = [
    str(PROJECT_ROOT / ".venv" / "bin" / "mlx_lm.lora"),
    "--config",
    str(CONFIG_PATH),
]
environment = os.environ.copy()
environment["PYTHONUNBUFFERED"] = "1"
started_at = datetime.now(timezone.utc)

print("Open Model Training Lab — first LoRA pipeline run")
print("purpose: validate model + data + LoRA + backward pass + adapter save")
print(f"iterations: {config['iters']}")
print(f"batch_size: {config['batch_size']}")
print(f"max_seq_length: {config['max_seq_length']}")
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
        f"LoRA pipeline run failed with exit code {return_code}. "
        f"The complete output is saved at {LOG_PATH}."
    )
if not adapter_file.is_file():
    raise SystemExit(
        "LoRA pipeline run failed: MLX-LM exited successfully but the final "
        "adapter file is missing."
    )

finished_at = datetime.now(timezone.utc)
result = {
    "name": "exp-001-pipeline",
    "purpose": "LoRA pipeline validation; not an accuracy experiment",
    "status": "complete",
    "started_at_utc": started_at.isoformat(),
    "finished_at_utc": finished_at.isoformat(),
    "elapsed_seconds": round(
        (finished_at - started_at).total_seconds(),
        3,
    ),
    "config_sha256": sha256(CONFIG_PATH),
    "sft_train_sha256": sft_manifest["files"]["train.jsonl"]["sha256"],
    "sft_valid_sha256": sft_manifest["files"]["valid.jsonl"]["sha256"],
    "tokenization_report_sha256": sha256(TOKEN_REPORT_PATH),
    "training_log_sha256": sha256(LOG_PATH),
    "adapter_sha256": sha256(adapter_file),
    "adapter_bytes": adapter_file.stat().st_size,
}
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"result: {RESULT_PATH}")
print(f"adapter_sha256: {result['adapter_sha256']}")
print(f"elapsed_seconds: {result['elapsed_seconds']}")
print("lora_pipeline_ok: True")
