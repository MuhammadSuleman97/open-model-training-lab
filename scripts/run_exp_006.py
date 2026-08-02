#!/usr/bin/env python3
"""Fine-tune BERT-Large on BANKING77 using validation-only selection."""

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
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-006-bert-large-classifier.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "model_manifest.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-006"
TRAINER_OUTPUT_DIR = RESULT_DIR / "trainer-output"
BEST_MODEL_DIR = RESULT_DIR / "best-model"
RESULT_PATH = RESULT_DIR / "result.json"


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path, expected_hash: str) -> list[dict[str, object]]:
    if sha256(path) != expected_hash:
        raise SystemExit(f"Experiment 006 stopped: checksum mismatch for {path.name}.")
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
    raise SystemExit("Experiment 006 stopped: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 006 stopped: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 006 stopped: completed result already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
labels = data_manifest["label_order"]
id2label = {index: label for index, label in enumerate(labels)}
label2id = {label: index for index, label in id2label.items()}
snapshot_path = model_manifest["snapshot_path"]

tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)


def encode_split(split_name: str) -> ListDataset:
    file_record = data_manifest["files"][split_name]
    source_records = read_jsonl(Path(file_record["path"]), file_record["sha256"])
    encoded_records: list[dict[str, object]] = []
    for record in source_records:
        encoded = tokenizer(
            str(record["text"]),
            max_length=config["tokenization"]["max_length"],
            truncation=True,
        )
        encoded["labels"] = int(record["label"])
        encoded_records.append(encoded)
    return ListDataset(encoded_records)


# Deliberately load only train and validation. The test artifact remains sealed.
train_dataset = encode_split("train")
validation_dataset = encode_split("valid")
if len(train_dataset) != config["data"]["train_rows"]:
    raise SystemExit("Experiment 006 stopped: training row count mismatch.")
if len(validation_dataset) != config["data"]["validation_rows"]:
    raise SystemExit("Experiment 006 stopped: validation row count mismatch.")

resume_checkpoint = None
if TRAINER_OUTPUT_DIR.exists():
    resume_checkpoint = get_last_checkpoint(str(TRAINER_OUTPUT_DIR))
    unexpected_entries = list(TRAINER_OUTPUT_DIR.iterdir())
    if unexpected_entries and resume_checkpoint is None:
        raise SystemExit(
            "Experiment 006 stopped: output exists without a resumable checkpoint."
        )

torch.manual_seed(config["optimization"]["seed"])
print("Open Model Training Lab — Experiment 006 BERT-Large classifier")
print(f"model_id: {model_manifest['model_id']}")
print(f"model_revision: {model_manifest['resolved_revision']}")
print("device: mps")
print(f"train_rows: {len(train_dataset)}")
print(f"validation_rows: {len(validation_dataset)}")
print("test_rows_loaded: 0")
print(f"labels: {len(labels)}")
print(f"epochs: {config['optimization']['epochs']}")
print(f"learning_rate: {config['optimization']['learning_rate']}")
print(f"train_batch_size: {config['batching']['train_batch_size']}")
print(
    "gradient_accumulation_steps: "
    f"{config['batching']['gradient_accumulation_steps']}"
)
print(f"effective_batch_size: {config['batching']['train_batch_size'] * config['batching']['gradient_accumulation_steps']}")
print(f"eval_batch_size: {config['batching']['eval_batch_size']}")
print(f"max_length: {config['tokenization']['max_length']}")
print("dynamic_padding: True")
print("trainer_handles_gradient_accumulation_normalization: True")
print("selection_data: validation_only")
print(f"resume_from_checkpoint: {resume_checkpoint}")
print("loading_model_and_training: True")

model = AutoModelForSequenceClassification.from_pretrained(
    snapshot_path,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
    num_labels=len(labels),
)
arguments = TrainingArguments(
    output_dir=str(TRAINER_OUTPUT_DIR),
    per_device_train_batch_size=config["batching"]["train_batch_size"],
    per_device_eval_batch_size=config["batching"]["eval_batch_size"],
    gradient_accumulation_steps=config["batching"]["gradient_accumulation_steps"],
    num_train_epochs=config["optimization"]["epochs"],
    learning_rate=config["optimization"]["learning_rate"],
    warmup_ratio=config["optimization"]["warmup_ratio"],
    weight_decay=config["optimization"]["weight_decay"],
    max_grad_norm=config["optimization"]["max_grad_norm"],
    optim=config["optimization"]["optimizer"],
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
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
    data_collator=DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    ),
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[StopOnNonFiniteLoss()],
)
# See the normalized probe: BERT's legacy mean loss ignores num_items_in_batch.
trainer.model_accepts_loss_kwargs = False

started = time.monotonic()
train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
torch.mps.synchronize()
elapsed_seconds = time.monotonic() - started
validation_metrics = trainer.evaluate(validation_dataset)

for name in ("eval_loss", "eval_accuracy", "eval_macro_f1"):
    value = float(validation_metrics[name])
    if not math.isfinite(value):
        raise SystemExit(f"Experiment 006 stopped: non-finite {name}.")

trainer.save_model(str(BEST_MODEL_DIR))
tokenizer.save_pretrained(str(BEST_MODEL_DIR))
weight_candidates = sorted(BEST_MODEL_DIR.glob("*.safetensors")) + sorted(
    BEST_MODEL_DIR.glob("*.bin")
)
if len(weight_candidates) != 1:
    raise SystemExit(
        "Experiment 006 stopped: expected one final model weight artifact."
    )
best_weight_path = weight_candidates[0]
validation_history = [
    {
        key: value
        for key, value in entry.items()
        if key in {"epoch", "eval_loss", "eval_accuracy", "eval_macro_f1"}
    }
    for entry in trainer.state.log_history
    if "eval_accuracy" in entry
]
result = {
    "experiment": config["experiment"],
    "model_id": model_manifest["model_id"],
    "model_revision": model_manifest["resolved_revision"],
    "device": "mps",
    "train_rows": len(train_dataset),
    "validation_rows": len(validation_dataset),
    "test_rows_loaded": 0,
    "epochs": config["optimization"]["epochs"],
    "learning_rate": config["optimization"]["learning_rate"],
    "effective_batch_size": (
        config["batching"]["train_batch_size"]
        * config["batching"]["gradient_accumulation_steps"]
    ),
    "max_length": config["tokenization"]["max_length"],
    "trainer_handles_gradient_accumulation_normalization": True,
    "train_loss": float(train_output.training_loss),
    "best_validation_accuracy": float(trainer.state.best_metric),
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "selected_validation": {
        "loss": float(validation_metrics["eval_loss"]),
        "accuracy": float(validation_metrics["eval_accuracy"]),
        "macro_f1": float(validation_metrics["eval_macro_f1"]),
    },
    "validation_history": validation_history,
    "experiment_accuracy_gate": config["evaluation"]["experiment_accuracy_gate"],
    "program_launch_target": config["evaluation"]["program_launch_target"],
    "test_evaluated": False,
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": torch.mps.driver_allocated_memory() / (1024**3),
    "best_model_path": str(BEST_MODEL_DIR),
    "best_weight_file": best_weight_path.name,
    "best_weight_sha256": sha256(best_weight_path),
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"train_loss: {result['train_loss']:.6f}")
print(f"best_validation_accuracy: {result['best_validation_accuracy']:.6f}")
print(f"selected_validation_accuracy: {result['selected_validation']['accuracy']:.6f}")
print(f"selected_validation_macro_f1: {result['selected_validation']['macro_f1']:.6f}")
print(f"best_checkpoint: {result['best_checkpoint']}")
print(f"elapsed_seconds: {elapsed_seconds:.3f}")
print(f"peak_driver_memory_gib: {result['peak_driver_memory_gib']:.3f}")
print(f"best_model: {BEST_MODEL_DIR}")
print(f"best_weight_sha256: {result['best_weight_sha256']}")
print("test_evaluated: False")
print(f"result: {RESULT_PATH}")
print("exp_006_training_ok: True")
