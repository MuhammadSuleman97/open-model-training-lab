#!/usr/bin/env python3
"""Continue fine-tuning Experiment 006 with class-balanced loss."""

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
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-007-bert-class-balanced-refinement.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
REFINEMENT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-007" / "refinement_manifest.json"
TRAINER_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-007" / "trainer-output"
RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-007" / "training_result.json"


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


class WeightedClassificationTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
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
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


class StopOnNonFiniteLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, state, control, kwargs
        for name, value in (logs or {}).items():
            if "loss" in name and isinstance(value, (float, int)):
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"Non-finite refinement metric: {name}={value}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path, expected_hash: str) -> list[dict[str, object]]:
    if sha256(path) != expected_hash:
        raise SystemExit(f"Experiment 007 stopped: checksum mismatch for {path.name}.")
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
    raise SystemExit("Experiment 007 stopped: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Experiment 007 stopped: PyTorch MPS is unavailable.")
if RESULT_PATH.exists():
    raise SystemExit("Experiment 007 stopped: completed training result already exists.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
refinement_manifest = json.loads(
    REFINEMENT_MANIFEST_PATH.read_text(encoding="utf-8")
)
labels = data_manifest["label_order"]
parent_model_path = Path(refinement_manifest["parent_model_path"])
parent_weight_path = parent_model_path / "model.safetensors"
if sha256(parent_weight_path) != config["model"]["parent_weight_sha256"]:
    raise SystemExit("Experiment 007 stopped: parent model checksum mismatch.")

tokenizer = AutoTokenizer.from_pretrained(parent_model_path, local_files_only=True)


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


train_dataset = encode_split("train")
validation_dataset = encode_split("valid")
class_weights = torch.tensor(
    [refinement_manifest["class_weights"][label] for label in labels],
    dtype=torch.float32,
)
resume_checkpoint = None
if TRAINER_OUTPUT_DIR.exists():
    resume_checkpoint = get_last_checkpoint(str(TRAINER_OUTPUT_DIR))
    if list(TRAINER_OUTPUT_DIR.iterdir()) and resume_checkpoint is None:
        raise SystemExit(
            "Experiment 007 stopped: output exists without a resumable checkpoint."
        )

parent_accuracy = float(config["evaluation"]["parent_validation_accuracy"])
torch.manual_seed(config["optimization"]["seed"])
print("Open Model Training Lab — Experiment 007 BERT refinement")
print(f"parent_experiment: {refinement_manifest['parent_experiment']}")
print(f"parent_validation_accuracy: {parent_accuracy:.6f}")
print(f"train_rows: {len(train_dataset)}")
print(f"validation_rows: {len(validation_dataset)}")
print("test_rows_loaded: 0")
print("loss: weighted_cross_entropy")
print(f"epochs: {config['optimization']['epochs']}")
print(f"learning_rate: {config['optimization']['learning_rate']}")
print(f"effective_batch_size: {config['batching']['train_batch_size'] * config['batching']['gradient_accumulation_steps']}")
print("promotion_policy: strictly_exceed_parent_validation_accuracy")
print(f"resume_from_checkpoint: {resume_checkpoint}")
print("loading_parent_model_and_training: True")

model = AutoModelForSequenceClassification.from_pretrained(
    parent_model_path,
    local_files_only=True,
)
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
trainer = WeightedClassificationTrainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[StopOnNonFiniteLoss()],
    class_weights=class_weights,
)

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
    raise SystemExit("Experiment 007 stopped: expected two validation measurements.")
best_child = max(
    validation_history,
    key=lambda entry: (entry["eval_accuracy"], -entry["epoch"]),
)
best_child_accuracy = float(best_child["eval_accuracy"])
promoted = best_child_accuracy > parent_accuracy
result = {
    "experiment": config["experiment"],
    "parent_experiment": refinement_manifest["parent_experiment"],
    "parent_model_path": str(parent_model_path),
    "parent_validation_accuracy": parent_accuracy,
    "train_rows": len(train_dataset),
    "validation_rows": len(validation_dataset),
    "test_rows_loaded": 0,
    "test_evaluated": False,
    "loss": "weighted_cross_entropy",
    "epochs": config["optimization"]["epochs"],
    "learning_rate": config["optimization"]["learning_rate"],
    "train_loss": float(train_output.training_loss),
    "validation_history": validation_history,
    "best_child_epoch": float(best_child["epoch"]),
    "best_child_validation_accuracy": best_child_accuracy,
    "best_child_checkpoint": trainer.state.best_model_checkpoint,
    "strict_improvement": best_child_accuracy - parent_accuracy,
    "promoted": promoted,
    "canonical_model_after_experiment": (
        "exp-007-pending-finalization" if promoted else "exp-006-selected-model"
    ),
    "launch_target_reached_on_validation": (
        best_child_accuracy >= config["evaluation"]["program_launch_target"]
    ),
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": torch.mps.driver_allocated_memory() / (1024**3),
}
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"train_loss: {result['train_loss']:.6f}")
for entry in validation_history:
    print(
        f"epoch_{entry['epoch']:g}_validation_accuracy: "
        f"{entry['eval_accuracy']:.6f}"
    )
    print(
        f"epoch_{entry['epoch']:g}_validation_macro_f1: "
        f"{entry['eval_macro_f1']:.6f}"
    )
print(f"best_child_epoch: {result['best_child_epoch']:g}")
print(f"best_child_validation_accuracy: {best_child_accuracy:.6f}")
print(f"strict_improvement: {result['strict_improvement']:+.6f}")
print(f"promoted: {promoted}")
print(
    "launch_target_reached_on_validation: "
    f"{result['launch_target_reached_on_validation']}"
)
print(f"elapsed_seconds: {elapsed_seconds:.3f}")
print(f"peak_driver_memory_gib: {result['peak_driver_memory_gib']:.3f}")
print("test_evaluated: False")
print(f"result: {RESULT_PATH}")
print("exp_007_training_ok: True")
