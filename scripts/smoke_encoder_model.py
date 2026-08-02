#!/usr/bin/env python3
"""Load Experiment 006 on MPS and verify one untrained classifier pass."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "exp-006-bert-large-classifier.json"
)
MODEL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "model_manifest.json"
CATEGORIES_PATH = PROJECT_ROOT / "data" / "banking77" / "raw" / "categories.json"
REQUEST = "How do I locate my card?"

if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Encoder smoke test failed: run with .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Encoder smoke test failed: PyTorch MPS is unavailable.")

experiment_config = json.loads(EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8"))
manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
categories = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
if len(categories) != experiment_config["data"]["labels"]:
    raise SystemExit("Encoder smoke test failed: expected exactly 77 categories.")
if manifest["resolved_revision"] != experiment_config["model"]["revision"]:
    raise SystemExit("Encoder smoke test failed: model revision mismatch.")

id2label = {index: label for index, label in enumerate(categories)}
label2id = {label: index for index, label in id2label.items()}
snapshot_path = manifest["snapshot_path"]
device = torch.device("mps")
torch.manual_seed(experiment_config["optimization"]["seed"])

print("Open Model Training Lab — encoder model smoke test")
print(f"model_id: {manifest['model_id']}")
print(f"model_revision: {manifest['resolved_revision']}")
print(f"device: {device}")
print(f"labels: {len(categories)}")
print("classifier_head_status: newly_initialized_untrained")
print(f"request: {REQUEST}")
print("loading_model: True")

tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    snapshot_path,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
    num_labels=len(categories),
)
model.to(device)
model.eval()

encoded = tokenizer(
    REQUEST,
    max_length=experiment_config["tokenization"]["max_length"],
    return_tensors="pt",
    truncation=True,
)
encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
with torch.no_grad():
    logits = model(**encoded).logits
torch.mps.synchronize()

logits_cpu = logits.float().cpu()
prediction_id = int(logits_cpu.argmax(dim=-1).item())
all_finite = all(math.isfinite(value) for value in logits_cpu.flatten().tolist())
expected_shape = [1, len(categories)]
observed_shape = list(logits_cpu.shape)
peak_memory_gib = torch.mps.driver_allocated_memory() / (1024**3)

print(f"input_tokens: {int(encoded['input_ids'].shape[1])}")
print(f"logits_shape: {observed_shape}")
print(f"logits_finite: {all_finite}")
print(f"untrained_prediction: {id2label[prediction_id]}")
print(f"peak_driver_memory_gib: {peak_memory_gib:.3f}")

if observed_shape != expected_shape:
    raise SystemExit(
        f"Encoder smoke test failed: expected logits shape {expected_shape}, "
        f"received {observed_shape}."
    )
if not all_finite:
    raise SystemExit("Encoder smoke test failed: logits contain non-finite values.")

print("encoder_model_smoke_ok: True")
