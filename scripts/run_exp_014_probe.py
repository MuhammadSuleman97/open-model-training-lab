#!/usr/bin/env python3
"""Run a finite-loss MPS probe for the broader Exp014 rival objective."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-014-deberta-broader-hard-negative.json"
CANONICAL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-014" / "data" / "manifest.json"
PARENT_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "result.json"
PARENT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "selected-model"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-014-probe"
RESULT_PATH = RESULT_DIR / "result.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop(message: str) -> None:
    raise SystemExit(f"Experiment 014 probe stopped: {message}")


def configure_trainable_parameters(model, config: dict[str, object]) -> tuple[int, int, list[str]]:
    freeze_config = config["freeze"]
    start = int(freeze_config["encoder_layer_start"])
    end = int(freeze_config["encoder_layer_end"])
    prefixes = [f"deberta.encoder.layer.{index}." for index in range(start, end + 1)]
    if bool(freeze_config["train_pooler"]):
        prefixes.append("pooler.")
    if bool(freeze_config["train_classifier"]):
        prefixes.append("classifier.")
    names = [name for name, _ in model.named_parameters()]
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name.startswith(prefix) for prefix in prefixes)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    missing = [prefix for prefix in prefixes if not any(name.startswith(prefix) for name in names)]
    return trainable, frozen, missing


def hard_negative_loss(logits, labels, rival_labels, margin: float, weight: float):
    cross_entropy = F.cross_entropy(logits, labels)
    active = rival_labels >= 0
    if bool(active.any()):
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        rival_logits = logits.gather(1, rival_labels.clamp_min(0).unsqueeze(1)).squeeze(1)
        pairwise = F.relu(margin - (true_logits - rival_logits))[active].mean()
        return cross_entropy + weight * pairwise, cross_entropy, pairwise, int(active.sum())
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    return cross_entropy, cross_entropy, zero, 0


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    stop("use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    stop("PyTorch MPS is unavailable; run this from the user's M2 Max Terminal.")
if RESULT_PATH.exists():
    stop("probe result already exists; refusing to overwrite.")
if not PARENT_MODEL_DIR.is_dir() or not PARENT_RESULT_PATH.is_file():
    stop("Exp011 selected model is missing.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
canonical_manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
parent_result = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
if data_manifest.get("validation_rows_loaded") != 0 or data_manifest.get("test_rows_loaded") != 0:
    stop("Exp014 data manifest reports held-out split access.")
if parent_result.get("test_rows_loaded") != 0 or parent_result.get("test_evaluated") is not False:
    stop("parent result reports test access.")
train_file = Path(data_manifest["files"]["train.jsonl"]["path"])
if sha256(train_file) != data_manifest["files"]["train.jsonl"]["sha256"]:
    stop("Exp014 train checksum mismatch.")
with train_file.open(encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle]
if len(records) != int(config["data"]["source_train_rows"]):
    stop("Exp014 train row count mismatch.")

labels = canonical_manifest["label_order"]
label2id = {label: index for index, label in enumerate(labels)}
active_records = [record for record in records if record["hard_negative_active"]]
inactive_records = [record for record in records if not record["hard_negative_active"]]
probe_rows = 64
if len(active_records) < probe_rows // 2 or len(inactive_records) < probe_rows // 2:
    stop("probe does not contain enough active and inactive rows.")
probe_records = active_records[: probe_rows // 2] + inactive_records[: probe_rows // 2]

torch.manual_seed(int(config["optimization"]["seed"]))
np.random.seed(int(config["optimization"]["seed"]))
device = torch.device("mps")
tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL_DIR, local_files_only=True, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(PARENT_MODEL_DIR, local_files_only=True, num_labels=len(labels))
if config["force_float32"]:
    model.float()
model.to(device)
trainable, frozen, missing = configure_trainable_parameters(model, config)
if missing:
    stop(f"trainable parameter prefixes are missing: {missing}")
if trainable <= 0 or frozen <= 0:
    stop("freeze configuration did not produce both trainable and frozen parameters.")

parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.AdamW(
    parameters,
    lr=float(config["optimization"]["learning_rate"]),
    weight_decay=float(config["optimization"]["weight_decay"]),
)
margin = float(config["hard_negative_loss"]["margin"])
loss_weight = float(config["hard_negative_loss"]["weight"])
batch_size = int(config["batching"]["train_batch_size"])
accumulation_steps = int(config["batching"]["gradient_accumulation_steps"])
model.train()
started = time.monotonic()
loss_history: list[dict[str, float | int]] = []
peak_memory = 0

for optimizer_step in range(1, int(config["optimization"]["probe_steps"]) + 1):
    optimizer.zero_grad(set_to_none=True)
    step_losses: list[float] = []
    step_pair_losses: list[float] = []
    active_count = 0
    for micro_step in range(accumulation_steps):
        start = (optimizer_step - 1) * accumulation_steps * batch_size + micro_step * batch_size
        batch_records = probe_records[start : start + batch_size]
        if len(batch_records) != batch_size:
            stop("probe batch indexing exceeded deterministic probe rows.")
        encoded = tokenizer(
            [str(record["text"]) for record in batch_records],
            max_length=int(config["tokenization"]["max_length"]),
            padding=True,
            truncation=bool(config["tokenization"]["truncation"]),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels_tensor = torch.tensor([int(record["label"]) for record in batch_records], dtype=torch.long, device=device)
        rival_tensor = torch.tensor(
            [label2id[record["hard_negative_label"]] if record["hard_negative_active"] else -1 for record in batch_records],
            dtype=torch.long,
            device=device,
        )
        outputs = model(**encoded)
        loss, cross_entropy, pairwise, active = hard_negative_loss(
            outputs.logits, labels_tensor, rival_tensor, margin, loss_weight
        )
        if not torch.isfinite(loss).item():
            stop(f"non-finite loss at optimizer step {optimizer_step}, micro-step {micro_step + 1}.")
        (loss / accumulation_steps).backward()
        step_losses.append(float(loss.detach().cpu()))
        step_pair_losses.append(float(pairwise.detach().cpu()))
        active_count += active
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(config["optimization"]["max_grad_norm"]))
    if not torch.isfinite(grad_norm).item():
        stop(f"non-finite gradient norm at optimizer step {optimizer_step}.")
    optimizer.step()
    torch.mps.synchronize()
    if not all(torch.isfinite(parameter).all().item() for parameter in parameters):
        stop(f"non-finite trainable parameter at optimizer step {optimizer_step}.")
    peak_memory = max(peak_memory, torch.mps.driver_allocated_memory())
    loss_history.append(
        {
            "optimizer_step": optimizer_step,
            "loss": float(np.mean(step_losses)),
            "cross_entropy": float(np.mean(step_losses) - loss_weight * np.mean(step_pair_losses)),
            "pairwise_loss": float(np.mean(step_pair_losses)),
            "active_rows": active_count,
            "gradient_norm": float(grad_norm.detach().cpu()),
        }
    )
    print(
        f"optimizer_step: {optimizer_step}, loss: {loss_history[-1]['loss']:.6f}, "
        f"pairwise_loss: {loss_history[-1]['pairwise_loss']:.6f}, active_rows: {active_count}"
    )

elapsed_seconds = time.monotonic() - started
RESULT_DIR.mkdir(parents=True, exist_ok=False)
result = {
    "experiment": "exp-014-probe",
    "parent_experiment": config["model"]["parent_experiment"],
    "parent_weight_sha256": parent_result["weight_sha256"],
    "probe_rows": probe_rows,
    "hard_negative_rows_in_probe": sum(bool(record["hard_negative_active"]) for record in probe_records),
    "train_rows": len(records),
    "hard_negative_rows": data_manifest["hard_negative_rows"],
    "learning_rate": config["optimization"]["learning_rate"],
    "hard_negative_margin": margin,
    "hard_negative_weight": loss_weight,
    "trainable_parameters": trainable,
    "frozen_parameters": frozen,
    "loss_history": loss_history,
    "peak_driver_memory_gib": peak_memory / (1024**3),
    "elapsed_seconds": elapsed_seconds,
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
    "weights_saved": False,
}
RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print("Open Model Training Lab — Experiment 014 broader hard-negative MPS probe")
print(f"parent_weight_sha256: {result['parent_weight_sha256']}")
print(f"probe_rows: {probe_rows}")
print(f"hard_negative_rows_in_probe: {result['hard_negative_rows_in_probe']}")
print(f"hard_negative_rows_total: {result['hard_negative_rows']}")
print(f"trainable_parameters: {trainable}")
print(f"frozen_parameters: {frozen}")
print(f"peak_driver_memory_gib: {result['peak_driver_memory_gib']:.3f}")
print("validation_rows_loaded: 0")
print("test_rows_loaded: 0")
print("weights_saved: False")
print(f"result: {RESULT_PATH}")
print("exp_014_probe_ok: True")
