#!/usr/bin/env python3
"""Run a two-update BERT-Large MPS training pipeline probe."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-006-bert-large-classifier.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "model_manifest.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-006-probe-normalized"
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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Experiment 006 probe failed: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 006 probe failed: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 006 probe stopped: result already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
labels = data_manifest["label_order"]
id2label = {index: label for index, label in enumerate(labels)}
label2id = {label: index for index, label in id2label.items()}
snapshot_path = model_manifest["snapshot_path"]
train_path = Path(data_manifest["files"]["train"]["path"])

torch.manual_seed(config["optimization"]["seed"])
tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
source_records = read_jsonl(train_path)
ranked_records = sorted(
    source_records,
    key=lambda record: len(
        tokenizer(str(record["text"]), add_special_tokens=True)["input_ids"]
    ),
    reverse=True,
)
probe_source = ranked_records[:PROBE_ROWS]
encoded_records: list[dict[str, object]] = []
observed_lengths: list[int] = []
for record in probe_source:
    encoded = tokenizer(
        str(record["text"]),
        max_length=config["tokenization"]["max_length"],
        truncation=True,
    )
    observed_lengths.append(len(encoded["input_ids"]))
    encoded["labels"] = int(record["label"])
    encoded_records.append(encoded)

print("Open Model Training Lab — Experiment 006 MPS training probe")
print(f"model_id: {model_manifest['model_id']}")
print(f"model_revision: {model_manifest['resolved_revision']}")
print("device: mps")
print(f"probe_rows: {PROBE_ROWS}")
print(f"probe_steps: {PROBE_STEPS}")
print(f"batch_size: {config['batching']['train_batch_size']}")
print(
    "gradient_accumulation_steps: "
    f"{config['batching']['gradient_accumulation_steps']}"
)
print("trainer_handles_gradient_accumulation_normalization: True")
print(f"max_observed_probe_tokens: {max(observed_lengths)}")
print(f"max_length: {config['tokenization']['max_length']}")
print("loading_model_and_training: True")

model = AutoModelForSequenceClassification.from_pretrained(
    snapshot_path,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
    num_labels=len(labels),
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
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    seed=config["optimization"]["seed"],
    data_seed=config["optimization"]["seed"],
    dataloader_pin_memory=False,
    disable_tqdm=False,
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=ListDataset(encoded_records),
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    processing_class=tokenizer,
)
# BertForSequenceClassification accepts **kwargs but its legacy classification
# loss ignores Trainer's num_items_in_batch. Tell Trainer to apply the required
# gradient-accumulation division itself.
trainer.model_accepts_loss_kwargs = False

started = time.monotonic()
train_output = trainer.train()
torch.mps.synchronize()
elapsed_seconds = time.monotonic() - started
training_loss = float(train_output.training_loss)
peak_memory_gib = torch.mps.driver_allocated_memory() / (1024**3)
if not math.isfinite(training_loss):
    raise SystemExit("Experiment 006 probe failed: training loss is non-finite.")

result = {
    "experiment": "exp-006-bert-large-classifier-probe-normalized",
    "model_id": model_manifest["model_id"],
    "model_revision": model_manifest["resolved_revision"],
    "device": "mps",
    "probe_rows": PROBE_ROWS,
    "probe_steps": PROBE_STEPS,
    "batch_size": config["batching"]["train_batch_size"],
    "gradient_accumulation_steps": config["batching"][
        "gradient_accumulation_steps"
    ],
    "trainer_handles_gradient_accumulation_normalization": True,
    "max_observed_probe_tokens": max(observed_lengths),
    "training_loss": training_loss,
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": peak_memory_gib,
    "weights_saved": False,
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
print("exp_006_probe_ok: True")
