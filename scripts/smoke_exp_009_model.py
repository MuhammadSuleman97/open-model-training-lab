#!/usr/bin/env python3
"""Load Experiment 009 on MPS and verify one untrained classifier pass."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-009-deberta-v3-large-classifier.json"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009" / "model_manifest.json"
CATEGORIES_PATH = PROJECT_ROOT / "data" / "banking77" / "raw" / "categories.json"
RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009" / "smoke_result.json"
REQUEST = "How do I locate my card?"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Experiment 009 smoke test failed: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 009 smoke test failed: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 009 smoke test stopped: result already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
categories = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
if len(categories) != config["data"]["labels"]:
    raise SystemExit("Experiment 009 smoke test failed: expected 77 categories.")
if manifest["resolved_revision"] != config["model"]["revision"]:
    raise SystemExit("Experiment 009 smoke test failed: model revision mismatch.")
snapshot_path = Path(manifest["snapshot_path"])
weight_path = snapshot_path / manifest["weight_file"]
if sha256(weight_path) != manifest["weight_sha256"]:
    raise SystemExit("Experiment 009 smoke test failed: model checksum mismatch.")

id2label = {index: label for index, label in enumerate(categories)}
label2id = {label: index for index, label in id2label.items()}
device = torch.device("mps")
torch.manual_seed(config["initialization_seed"])

print("Open Model Training Lab — Experiment 009 model smoke test")
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
    max_length=config["tokenization"]["max_length"],
    return_tensors="pt",
    truncation=True,
)
encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
with torch.no_grad():
    logits = model(**encoded).logits
torch.mps.synchronize()

logits_cpu = logits.float().cpu()
prediction_id = int(logits_cpu.argmax(dim=-1).item())
all_finite = bool(torch.isfinite(logits_cpu).all().item())
expected_shape = [1, len(categories)]
observed_shape = list(logits_cpu.shape)
peak_memory_gib = torch.mps.driver_allocated_memory() / (1024**3)
result = {
    "experiment": config["experiment"],
    "model_id": manifest["model_id"],
    "model_revision": manifest["resolved_revision"],
    "device": str(device),
    "labels": len(categories),
    "classifier_head_status": "newly_initialized_untrained",
    "request": REQUEST,
    "input_tokens": int(encoded["input_ids"].shape[1]),
    "logits_shape": observed_shape,
    "logits_finite": all_finite,
    "untrained_prediction": id2label[prediction_id],
    "peak_driver_memory_gib": peak_memory_gib,
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
}
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"input_tokens: {result['input_tokens']}")
print(f"logits_shape: {observed_shape}")
print(f"logits_finite: {all_finite}")
print(f"untrained_prediction: {result['untrained_prediction']}")
print(f"peak_driver_memory_gib: {peak_memory_gib:.3f}")
print("validation_rows_loaded: 0")
print("test_rows_loaded: 0")
if observed_shape != expected_shape:
    raise SystemExit(
        f"Experiment 009 smoke test failed: expected {expected_shape}, "
        f"received {observed_shape}."
    )
if not all_finite or not math.isfinite(peak_memory_gib):
    raise SystemExit("Experiment 009 smoke test failed: non-finite output.")
print(f"result: {RESULT_PATH}")
print("exp_009_model_smoke_ok: True")
