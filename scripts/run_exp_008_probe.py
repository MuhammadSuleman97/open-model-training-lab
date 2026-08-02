#!/usr/bin/env python3
"""Probe label-smoothed continued fine-tuning from Experiment 006."""

from __future__ import annotations

import json
import hashlib
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-008-bert-label-smoothing-refinement.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
PARENT_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-006" / "result.json"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-008-probe"
RESULT_PATH = RESULT_DIR / "result.json"
PROBE_ROWS = 64
PROBE_STEPS = 2


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


class SmoothedClassificationTrainer(Trainer):
    def __init__(self, *args, label_smoothing: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits,
            labels,
            label_smoothing=self.label_smoothing,
        )
        return (loss, outputs) if return_outputs else loss


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Experiment 008 probe failed: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 008 probe failed: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 008 probe stopped: result already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
parent_result = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
parent_model_path = Path(parent_result["selected_model_path"])
parent_weight_path = parent_model_path / parent_result["weight_file"]
if parent_result["weight_sha256"] != config["model"]["parent_weight_sha256"]:
    raise SystemExit("Experiment 008 probe failed: configured parent hash mismatch.")
parent_weight_hash = hashlib.sha256(parent_weight_path.read_bytes()).hexdigest()
if parent_weight_hash != config["model"]["parent_weight_sha256"]:
    raise SystemExit("Experiment 008 probe failed: parent weight file hash mismatch.")

torch.manual_seed(config["optimization"]["seed"])
tokenizer = AutoTokenizer.from_pretrained(parent_model_path, local_files_only=True)
source_records = read_jsonl(Path(data_manifest["files"]["train"]["path"]))
ranked_records = sorted(
    source_records,
    key=lambda record: len(
        tokenizer(str(record["text"]), add_special_tokens=True)["input_ids"]
    ),
    reverse=True,
)
probe_source = ranked_records[:PROBE_ROWS]
encoded_records: list[dict[str, object]] = []
for record in probe_source:
    encoded = tokenizer(
        str(record["text"]),
        max_length=config["tokenization"]["max_length"],
        truncation=True,
    )
    encoded["labels"] = int(record["label"])
    encoded_records.append(encoded)

label_smoothing = float(config["objective"]["label_smoothing"])
print("Open Model Training Lab — Experiment 008 label-smoothing probe")
print(f"parent_experiment: {config['model']['parent_experiment']}")
print(f"parent_validation_accuracy: {config['evaluation']['parent_validation_accuracy']:.6f}")
print(f"probe_rows: {PROBE_ROWS}")
print(f"probe_steps: {PROBE_STEPS}")
print(f"label_smoothing: {label_smoothing}")
print("class_weights: uniform")
print("trainer_handles_gradient_accumulation_normalization: True")
print("validation_rows_loaded: 0")
print("test_rows_loaded: 0")
print("loading_parent_model_and_training: True")

model = AutoModelForSequenceClassification.from_pretrained(
    parent_model_path,
    local_files_only=True,
)
arguments = TrainingArguments(
    output_dir=str(RESULT_DIR / "trainer-output"),
    per_device_train_batch_size=config["batching"]["train_batch_size"],
    gradient_accumulation_steps=config["batching"]["gradient_accumulation_steps"],
    max_steps=PROBE_STEPS,
    learning_rate=config["optimization"]["learning_rate"],
    max_grad_norm=config["optimization"]["max_grad_norm"],
    weight_decay=config["optimization"]["weight_decay"],
    optim=config["optimization"]["optimizer"],
    logging_strategy="steps",
    logging_steps=1,
    logging_nan_inf_filter=False,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    seed=config["optimization"]["seed"],
    data_seed=config["optimization"]["seed"],
    dataloader_pin_memory=False,
)
trainer = SmoothedClassificationTrainer(
    model=model,
    args=arguments,
    train_dataset=ListDataset(encoded_records),
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
    processing_class=tokenizer,
    label_smoothing=label_smoothing,
)
started = time.monotonic()
train_output = trainer.train()
torch.mps.synchronize()
elapsed_seconds = time.monotonic() - started
training_loss = float(train_output.training_loss)
peak_memory_gib = torch.mps.driver_allocated_memory() / (1024**3)
if not math.isfinite(training_loss):
    raise SystemExit("Experiment 008 probe failed: non-finite training loss.")

result = {
    "experiment": "exp-008-bert-label-smoothing-refinement-probe",
    "parent_experiment": config["model"]["parent_experiment"],
    "probe_rows": PROBE_ROWS,
    "probe_steps": PROBE_STEPS,
    "label_smoothing": label_smoothing,
    "class_weights": "uniform",
    "training_loss": training_loss,
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": peak_memory_gib,
    "weights_saved": False,
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"training_loss: {training_loss:.6f}")
print(f"elapsed_seconds: {elapsed_seconds:.3f}")
print(f"peak_driver_memory_gib: {peak_memory_gib:.3f}")
print("weights_saved: False")
print(f"result: {RESULT_PATH}")
print("exp_008_probe_ok: True")
