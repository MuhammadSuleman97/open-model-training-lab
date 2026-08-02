#!/usr/bin/env python3
"""Fine-tune float32 DeBERTa-v3-large on BANKING77 with sealed test data."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-009d-deberta-v3-large-float32.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009" / "model_manifest.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d"
TRAINER_OUTPUT_DIR = RESULT_DIR / "trainer-output"
RESULT_PATH = RESULT_DIR / "training_result.json"


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


class StopOnNonFiniteLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, state, control, kwargs
        for name, value in (logs or {}).items():
            if "loss" in name and isinstance(value, (float, int)):
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"Non-finite training metric: {name}={value}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, expected_hash: str) -> list[dict[str, object]]:
    if sha256(path) != expected_hash:
        raise SystemExit(f"Experiment 009d stopped: checksum mismatch for {path.name}.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    predicted_ids = np.asarray(prediction.predictions).argmax(axis=-1)
    expected_ids = np.asarray(prediction.label_ids)
    return {
        "accuracy": float(accuracy_score(expected_ids, predicted_ids)),
        "macro_f1": float(
            f1_score(expected_ids, predicted_ids, average="macro", zero_division=0)
        ),
    }


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Experiment 009d stopped: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 009d stopped: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 009d stopped: completed training result already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
model_path = Path(model_manifest["snapshot_path"])
if sha256(model_path / model_manifest["weight_file"]) != model_manifest["weight_sha256"]:
    raise SystemExit("Experiment 009d stopped: model checksum mismatch.")
labels = data_manifest["label_order"]
id2label = {index: label for index, label in enumerate(labels)}
label2id = {label: index for index, label in id2label.items()}
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    use_fast=False,
)


def encode_split(split_name: str) -> ListDataset:
    file_record = data_manifest["files"][split_name]
    records = read_jsonl(Path(file_record["path"]), file_record["sha256"])
    encoded_records: list[dict[str, object]] = []
    for record in records:
        encoded = tokenizer(
            str(record["text"]),
            max_length=config["tokenization"]["max_length"],
            truncation=True,
        )
        encoded["labels"] = int(record["label"])
        encoded_records.append(encoded)
    return ListDataset(encoded_records)


train_dataset = encode_split("train")
validation_dataset = encode_split("valid")
if len(train_dataset) != config["data"]["train_rows"]:
    raise SystemExit("Experiment 009d stopped: training row count mismatch.")
if len(validation_dataset) != config["data"]["validation_rows"]:
    raise SystemExit("Experiment 009d stopped: validation row count mismatch.")

resume_checkpoint = None
if TRAINER_OUTPUT_DIR.exists():
    resume_checkpoint = get_last_checkpoint(str(TRAINER_OUTPUT_DIR))
    if list(TRAINER_OUTPUT_DIR.iterdir()) and resume_checkpoint is None:
        raise SystemExit(
            "Experiment 009d stopped: output exists without a resumable checkpoint."
        )

torch.manual_seed(config["optimization"]["seed"])
print("Open Model Training Lab — Experiment 009d DeBERTa-v3-large classifier")
print(f"model_id: {model_manifest['model_id']}")
print(f"model_revision: {model_manifest['resolved_revision']}")
print("device: mps")
print("model_parameter_dtype: float32")
print(f"train_rows: {len(train_dataset)}")
print(f"validation_rows: {len(validation_dataset)}")
print("test_rows_loaded: 0")
print(f"labels: {len(labels)}")
print(f"epochs: {config['optimization']['epochs']}")
print(f"learning_rate: {config['optimization']['learning_rate']}")
print(f"train_batch_size: {config['batching']['train_batch_size']}")
print(f"gradient_accumulation_steps: {config['batching']['gradient_accumulation_steps']}")
print(
    "effective_batch_size: "
    f"{config['batching']['train_batch_size'] * config['batching']['gradient_accumulation_steps']}"
)
print(f"eval_batch_size: {config['batching']['eval_batch_size']}")
print(f"max_length: {config['tokenization']['max_length']}")
print("dynamic_padding: True")
print("selection_data: validation_only")
print(f"resume_from_checkpoint: {resume_checkpoint}")
print("loading_model_and_training: True")

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
    num_labels=len(labels),
)
model.float()
arguments = TrainingArguments(
    output_dir=str(TRAINER_OUTPUT_DIR),
    per_device_train_batch_size=config["batching"]["train_batch_size"],
    per_device_eval_batch_size=config["batching"]["eval_batch_size"],
    gradient_accumulation_steps=config["batching"]["gradient_accumulation_steps"],
    num_train_epochs=config["optimization"]["epochs"],
    learning_rate=config["optimization"]["learning_rate"],
    warmup_steps=config["optimization"]["warmup_steps"],
    weight_decay=config["optimization"]["weight_decay"],
    max_grad_norm=config["optimization"]["max_grad_norm"],
    optim=config["optimization"]["optimizer"],
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_strategy="steps",
    logging_steps=25,
    logging_first_step=True,
    logging_nan_inf_filter=False,
    report_to="none",
    seed=config["optimization"]["seed"],
    data_seed=config["optimization"]["seed"],
    dataloader_pin_memory=False,
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[StopOnNonFiniteLoss()],
)
trainer.model_accepts_loss_kwargs = False

started = time.monotonic()
train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
torch.mps.synchronize()
elapsed_seconds = time.monotonic() - started
validation_history = [
    {
        key: entry[key]
        for key in ("epoch", "eval_loss", "eval_accuracy", "eval_macro_f1")
    }
    for entry in trainer.state.log_history
    if "eval_accuracy" in entry
]
if len(validation_history) != config["optimization"]["epochs"]:
    raise SystemExit("Experiment 009d stopped: expected five validation measurements.")
if not math.isfinite(float(train_output.training_loss)):
    raise SystemExit("Experiment 009d stopped: final train loss is non-finite.")
if not all(
    math.isfinite(float(entry[key]))
    for entry in validation_history
    for key in ("eval_loss", "eval_accuracy", "eval_macro_f1")
):
    raise SystemExit("Experiment 009d stopped: validation metric is non-finite.")

best_child = max(
    validation_history,
    key=lambda entry: (entry["eval_accuracy"], -entry["epoch"]),
)
result = {
    "experiment": config["experiment"],
    "model_id": model_manifest["model_id"],
    "model_revision": model_manifest["resolved_revision"],
    "device": "mps",
    "model_parameter_dtype": "float32",
    "train_rows": len(train_dataset),
    "validation_rows": len(validation_dataset),
    "test_rows_loaded": 0,
    "epochs": config["optimization"]["epochs"],
    "learning_rate": config["optimization"]["learning_rate"],
    "effective_batch_size": config["batching"]["train_batch_size"] * config["batching"]["gradient_accumulation_steps"],
    "train_loss": float(train_output.training_loss),
    "validation_history": validation_history,
    "best_child_epoch": float(best_child["epoch"]),
    "best_child_validation_accuracy": float(best_child["eval_accuracy"]),
    "best_child_checkpoint": str(TRAINER_OUTPUT_DIR / f"checkpoint-{trainer.state.global_step}"),
    "bert_champion_validation_accuracy": config["evaluation"]["bert_champion_validation_accuracy"],
    "program_launch_target": config["evaluation"]["program_launch_target"],
    "test_evaluated": False,
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": torch.mps.driver_allocated_memory() / (1024**3),
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"train_loss: {result['train_loss']:.6f}")
for entry in validation_history:
    print(f"epoch_{entry['epoch']:g}_validation_accuracy: {entry['eval_accuracy']:.6f}")
    print(f"epoch_{entry['epoch']:g}_validation_macro_f1: {entry['eval_macro_f1']:.6f}")
print(f"best_child_epoch: {result['best_child_epoch']:g}")
print(f"best_child_validation_accuracy: {result['best_child_validation_accuracy']:.6f}")
print(f"bert_champion_validation_accuracy: {result['bert_champion_validation_accuracy']:.6f}")
print(f"elapsed_seconds: {elapsed_seconds:.3f}")
print(f"peak_driver_memory_gib: {result['peak_driver_memory_gib']:.3f}")
print("test_evaluated: False")
print(f"result: {RESULT_PATH}")
print("exp_009d_training_ok: True")
