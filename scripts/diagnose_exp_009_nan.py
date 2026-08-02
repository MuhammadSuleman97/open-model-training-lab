#!/usr/bin/env python3
"""Locate the first non-finite value in the DeBERTa MPS update path."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = Path(
    os.environ.get(
        "OPEN_MODEL_TRAINING_CONFIG",
        str(PROJECT_ROOT / "configs" / "exp-009b-deberta-v3-large-stability.json"),
    )
)
MODEL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009" / "model_manifest.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
RESULT_DIR = Path(
    os.environ.get(
        "OPEN_MODEL_TRAINING_RESULT_DIR",
        str(PROJECT_ROOT / "artifacts" / "encoder" / "exp-009b-diagnosis"),
    )
)
RESULT_PATH = RESULT_DIR / "result.json"
DIAGNOSIS_ROWS = 32
ACCUMULATION_STEPS = 4


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Experiment 009 diagnosis failed: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 009 diagnosis failed: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 009 diagnosis stopped: result already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
model_path = Path(model_manifest["snapshot_path"])
if sha256(model_path / model_manifest["weight_file"]) != model_manifest["weight_sha256"]:
    raise SystemExit("Experiment 009 diagnosis failed: model checksum mismatch.")
labels = data_manifest["label_order"]
id2label = {index: label for index, label in enumerate(labels)}
label2id = {label: index for index, label in id2label.items()}
train_file = data_manifest["files"]["train"]
if sha256(Path(train_file["path"])) != train_file["sha256"]:
    raise SystemExit("Experiment 009 diagnosis failed: train checksum mismatch.")

with Path(train_file["path"]).open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]
torch.manual_seed(config["optimization"]["seed"])
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    use_fast=False,
)
ranked_records = sorted(
    source_records,
    key=lambda record: len(
        tokenizer(str(record["text"]), add_special_tokens=True)["input_ids"]
    ),
    reverse=True,
)
encoded_records: list[dict[str, object]] = []
for record in ranked_records[:DIAGNOSIS_ROWS]:
    encoded = tokenizer(
        str(record["text"]),
        max_length=config["tokenization"]["max_length"],
        truncation=True,
    )
    encoded["labels"] = int(record["label"])
    encoded_records.append(encoded)
loader = torch.utils.data.DataLoader(
    ListDataset(encoded_records),
    batch_size=config["batching"]["train_batch_size"],
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
)

print("Open Model Training Lab — Experiment 009 numerical diagnosis")
print("model_id: microsoft/deberta-v3-large")
print(f"learning_rate: {config['optimization']['learning_rate']}")
print(f"diagnosis_rows: {DIAGNOSIS_ROWS}")
print(f"accumulation_steps: {ACCUMULATION_STEPS}")
print(f"frozen_parameter_prefixes: {config.get('frozen_parameter_prefixes', [])}")
print("tokenizer: native_sentencepiece_slow")
print("validation_rows_loaded: 0")
print("test_rows_loaded: 0")
print("loading_model: True")

device = torch.device("mps")
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
    num_labels=len(labels),
)
frozen_parameter_prefixes = config.get("frozen_parameter_prefixes", [])
frozen_parameter_names: list[str] = []
for name, parameter in model.named_parameters():
    if any(name.startswith(prefix) for prefix in frozen_parameter_prefixes):
        parameter.requires_grad = False
        frozen_parameter_names.append(name)
print(f"frozen_parameter_count: {len(frozen_parameter_names)}")
model.to(device)
model.train()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["optimization"]["learning_rate"],
    weight_decay=config["optimization"]["weight_decay"],
)

first_bad_stage: str | None = None
first_bad_name: str | None = None
losses: list[float] = []
step_summaries: list[dict[str, object]] = []
started = time.monotonic()
iterator = iter(loader)
for optimizer_step in range(1, 3):
    optimizer.zero_grad(set_to_none=True)
    micro_losses: list[float] = []
    bad_gradient_names: list[str] = []
    for micro_step in range(1, ACCUMULATION_STEPS + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss_value = float(loss.detach().float().cpu().item())
        micro_losses.append(loss_value)
        losses.append(loss_value)
        print(f"optimizer_step: {optimizer_step}, micro_step: {micro_step}, loss: {loss_value}")
        if not math.isfinite(loss_value) and first_bad_stage is None:
            first_bad_stage = "forward_loss"
            first_bad_name = f"step_{optimizer_step}_micro_{micro_step}"
        (loss / ACCUMULATION_STEPS).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not finite_tensor(parameter.grad):
            bad_gradient_names.append(name)
            if first_bad_stage is None:
                first_bad_stage = "gradient"
                first_bad_name = name
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimization"]["max_grad_norm"])
    optimizer.step()
    bad_parameter_names = [
        name for name, parameter in model.named_parameters() if not finite_tensor(parameter)
    ]
    if bad_parameter_names and first_bad_stage is None:
        first_bad_stage = "parameter_after_optimizer_step"
        first_bad_name = bad_parameter_names[0]
    step_summaries.append(
        {
            "optimizer_step": optimizer_step,
            "micro_losses": micro_losses,
            "bad_gradient_count": len(bad_gradient_names),
            "bad_gradient_names": bad_gradient_names[:10],
            "bad_parameter_count": len(bad_parameter_names),
            "bad_parameter_names": bad_parameter_names[:10],
        }
    )
    if bad_parameter_names:
        break

torch.mps.synchronize()
elapsed_seconds = time.monotonic() - started
result = {
    "experiment": config["experiment"] + "-diagnosis",
    "learning_rate": config["optimization"]["learning_rate"],
    "diagnosis_rows": DIAGNOSIS_ROWS,
    "accumulation_steps": ACCUMULATION_STEPS,
    "tokenizer": "native_sentencepiece_slow",
    "frozen_parameter_prefixes": frozen_parameter_prefixes,
    "frozen_parameter_count": len(frozen_parameter_names),
    "losses": losses,
    "step_summaries": step_summaries,
    "first_bad_stage": first_bad_stage,
    "first_bad_name": first_bad_name,
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": torch.mps.driver_allocated_memory() / (1024**3),
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
    "weights_saved": False,
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"first_bad_stage: {first_bad_stage}")
print(f"first_bad_name: {first_bad_name}")
print(f"elapsed_seconds: {elapsed_seconds:.3f}")
print(f"peak_driver_memory_gib: {result['peak_driver_memory_gib']:.3f}")
print("weights_saved: False")
print(f"result: {RESULT_PATH}")
print(f"{config.get('diagnosis_success_marker', 'exp_009b_diagnosis_ok')}: True")
