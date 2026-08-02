#!/usr/bin/env python3
"""Train Experiment 013 with a train-only hard-negative objective.

The canonical class label remains the target for every row.  For the small
set of rows identified by the train-only audit, the loss also asks the model
to place the true class above the audit model's strongest rival by a margin.
The validation split is used only for reporting/selection and the observed
test split is never opened by this script.
"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-013-deberta-hard-negative-refinement.json"
CANONICAL_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-013" / "data" / "manifest.json"
PARENT_RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "result.json"
PARENT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-011" / "selected-model"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-013"
TRAINER_OUTPUT_DIR = RESULT_DIR / "trainer-output"
RESULT_PATH = RESULT_DIR / "training_result.json"


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


class HardNegativeCollator:
    """Pad model inputs while carrying rival-label ids as a separate tensor."""

    def __init__(self, tokenizer) -> None:
        self.base = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        copied = [dict(feature) for feature in features]
        rival_ids = torch.tensor(
            [int(feature.pop("hard_negative_labels", -1)) for feature in copied],
            dtype=torch.long,
        )
        batch = self.base(copied)
        batch["hard_negative_labels"] = rival_ids
        return batch


class StopOnNonFiniteLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, state, control, kwargs
        for name, value in (logs or {}).items():
            if "loss" in name and isinstance(value, (float, int)) and not math.isfinite(float(value)):
                raise RuntimeError(f"Non-finite training metric: {name}={value}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, expected_hash: str) -> list[dict[str, object]]:
    if sha256(path) != expected_hash:
        raise SystemExit(f"Experiment 013 stopped: checksum mismatch for {path.name}.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def encode_records(
    records: list[dict[str, object]],
    tokenizer,
    max_length: int,
    label2id: dict[str, int],
    include_rivals: bool,
) -> ListDataset:
    encoded_records: list[dict[str, object]] = []
    for record in records:
        encoded = tokenizer(str(record["text"]), max_length=max_length, truncation=True)
        encoded["labels"] = int(record["label"])
        if include_rivals:
            active = bool(record["hard_negative_active"])
            rival_name = record.get("hard_negative_label")
            if active and (not isinstance(rival_name, str) or rival_name not in label2id):
                raise SystemExit("Experiment 013 stopped: active rival label is unknown.")
            if active and int(record["label"]) == label2id[rival_name]:
                raise SystemExit("Experiment 013 stopped: rival label equals the true label.")
            encoded["hard_negative_labels"] = label2id[rival_name] if active else -1
        else:
            encoded["hard_negative_labels"] = -1
        encoded_records.append(encoded)
    return ListDataset(encoded_records)


def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    predicted_ids = np.asarray(prediction.predictions).argmax(axis=-1)
    expected_ids = np.asarray(prediction.label_ids)
    return {
        "accuracy": float(accuracy_score(expected_ids, predicted_ids)),
        "macro_f1": float(f1_score(expected_ids, predicted_ids, average="macro", zero_division=0)),
    }


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


def hard_negative_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    hard_negative_labels: torch.Tensor,
    margin: float,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    cross_entropy = F.cross_entropy(logits, labels)
    active = hard_negative_labels >= 0
    if bool(active.any()):
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        rival_ids = hard_negative_labels.clamp_min(0)
        rival_logits = logits.gather(1, rival_ids.unsqueeze(1)).squeeze(1)
        pairwise = F.relu(margin - (true_logits - rival_logits))[active].mean()
        return cross_entropy + weight * pairwise, cross_entropy, pairwise, int(active.sum())
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    return cross_entropy, cross_entropy, zero, 0


class HardNegativeTrainer(Trainer):
    def __init__(self, *args, hard_negative_margin: float, hard_negative_weight: float, **kwargs):
        self.hard_negative_margin = hard_negative_margin
        self.hard_negative_weight = hard_negative_weight
        self.training_components: list[dict[str, float | int]] = []
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        rival_labels = inputs.pop("hard_negative_labels", None)
        labels = inputs.get("labels")
        if labels is None:
            raise RuntimeError("Experiment 013 stopped: labels are missing from a batch.")
        if rival_labels is None:
            rival_labels = torch.full_like(labels, -1)
        outputs = model(**inputs)
        loss, cross_entropy, pairwise, active_rows = hard_negative_loss(
            outputs.logits,
            labels,
            rival_labels,
            self.hard_negative_margin,
            self.hard_negative_weight,
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Experiment 013 stopped: non-finite loss ({float(loss.detach().cpu())}).")
        if model.training:
            self.training_components.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "cross_entropy": float(cross_entropy.detach().cpu()),
                    "pairwise_loss": float(pairwise.detach().cpu()),
                    "active_rows": active_rows,
                }
            )
        return (loss, outputs) if return_outputs else loss


def stop(message: str) -> None:
    raise SystemExit(f"Experiment 013 stopped: {message}")


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    stop("use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    stop("PyTorch MPS is unavailable; run this from the user's normal M2 Max Terminal.")
if RESULT_PATH.exists():
    stop("training result already exists; refusing to overwrite.")
if TRAINER_OUTPUT_DIR.exists() and any(TRAINER_OUTPUT_DIR.iterdir()):
    stop("trainer output already exists; refusing to overwrite.")
if not PARENT_MODEL_DIR.is_dir() or not PARENT_RESULT_PATH.is_file():
    stop("Experiment 011 selected model is missing.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
canonical_manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
parent_result = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
if data_manifest.get("validation_rows_loaded") != 0 or data_manifest.get("test_rows_loaded") != 0:
    stop("Exp013 data manifest reports held-out split access.")
if data_manifest.get("labels_rewritten") is not False:
    stop("Exp013 data manifest does not preserve the original labels.")
if parent_result.get("test_rows_loaded") != 0 or parent_result.get("test_evaluated") is not False:
    stop("parent result reports test access.")
train_file = Path(data_manifest["files"]["train.jsonl"]["path"])
train_records = read_jsonl(train_file, data_manifest["files"]["train.jsonl"]["sha256"])
valid_file = canonical_manifest["files"]["valid"]
valid_records = read_jsonl(Path(valid_file["path"]), valid_file["sha256"])
if len(train_records) != int(data_manifest["train_rows"]):
    stop("train row count mismatch.")
if len(train_records) != int(config["data"]["source_train_rows"]):
    stop("configured source train row count mismatch.")
if len(valid_records) != int(config["data"]["validation_rows"]):
    stop("validation row count mismatch.")
active_rows = sum(bool(record["hard_negative_active"]) for record in train_records)
if active_rows != int(data_manifest["hard_negative_rows"]):
    stop("hard-negative row count mismatch.")

labels = canonical_manifest["label_order"]
id2label = {index: label for index, label in enumerate(labels)}
label2id = {label: index for index, label in id2label.items()}
tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL_DIR, local_files_only=True, use_fast=False)
train_dataset = encode_records(
    train_records,
    tokenizer,
    int(config["tokenization"]["max_length"]),
    label2id,
    include_rivals=True,
)
validation_dataset = encode_records(
    valid_records,
    tokenizer,
    int(config["tokenization"]["max_length"]),
    label2id,
    include_rivals=False,
)

print("Open Model Training Lab — Experiment 013 hard-negative refinement")
print(f"parent_model: {PARENT_MODEL_DIR}")
print(f"parent_weight_sha256: {parent_result['weight_sha256']}")
print(f"train_rows: {len(train_dataset)}")
print(f"hard_negative_rows: {active_rows}")
print(f"validation_rows: {len(validation_dataset)}")
print("test_rows_loaded: 0")
print(f"labels: {len(labels)}")
print("epochs: 1")
print(f"learning_rate: {config['optimization']['learning_rate']}")
print(f"hard_negative_margin: {config['hard_negative_loss']['margin']}")
print(f"hard_negative_weight: {config['hard_negative_loss']['weight']}")
print(f"train_batch_size: {config['batching']['train_batch_size']}")
print(f"gradient_accumulation_steps: {config['batching']['gradient_accumulation_steps']}")
print(
    f"trainable_encoder_layers: {config['freeze']['encoder_layer_start']}-"
    f"{config['freeze']['encoder_layer_end']}"
)
print("selection_data: validation_only")
print("loading_parent_model_and_training: True")

model = AutoModelForSequenceClassification.from_pretrained(
    PARENT_MODEL_DIR,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
    num_labels=len(labels),
)
if config["force_float32"]:
    model.float()
trainable_parameters, frozen_parameters, missing_prefixes = configure_trainable_parameters(model, config)
if missing_prefixes:
    stop(f"expected trainable parameter prefixes are missing: {missing_prefixes}")
if trainable_parameters <= 0 or frozen_parameters <= 0:
    stop("freeze configuration did not produce both trainable and frozen parameters.")
print("model_parameter_dtype: float32")
print(f"trainable_parameters: {trainable_parameters}")
print(f"frozen_parameters: {frozen_parameters}")

arguments = TrainingArguments(
    output_dir=str(TRAINER_OUTPUT_DIR),
    per_device_train_batch_size=int(config["batching"]["train_batch_size"]),
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=int(config["batching"]["gradient_accumulation_steps"]),
    num_train_epochs=1,
    learning_rate=float(config["optimization"]["learning_rate"]),
    warmup_steps=int(config["optimization"]["warmup_steps"]),
    weight_decay=float(config["optimization"]["weight_decay"]),
    max_grad_norm=float(config["optimization"]["max_grad_norm"]),
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
    seed=int(config["optimization"]["seed"]),
    data_seed=int(config["optimization"]["seed"]),
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)
trainer = HardNegativeTrainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=HardNegativeCollator(tokenizer),
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[StopOnNonFiniteLoss()],
    hard_negative_margin=float(config["hard_negative_loss"]["margin"]),
    hard_negative_weight=float(config["hard_negative_loss"]["weight"]),
)
trainer.model_accepts_loss_kwargs = False

started = time.monotonic()
train_output = trainer.train()
torch.mps.synchronize()
elapsed_seconds = time.monotonic() - started
validation_history = [
    {
        "epoch": entry["epoch"],
        "step": int(entry.get("step", trainer.state.global_step)),
        "eval_loss": entry["eval_loss"],
        "eval_accuracy": entry["eval_accuracy"],
        "eval_macro_f1": entry["eval_macro_f1"],
    }
    for entry in trainer.state.log_history
    if "eval_accuracy" in entry
]
if len(validation_history) != 1:
    stop("expected one validation measurement in the completed training log.")
if not math.isfinite(float(train_output.training_loss)):
    stop("final train loss is non-finite.")
if not all(
    math.isfinite(float(entry[key]))
    for entry in validation_history
    for key in ("eval_loss", "eval_accuracy", "eval_macro_f1")
):
    stop("validation metric is non-finite.")
if not trainer.training_components:
    stop("no training loss components were recorded.")

best_child = validation_history[0]
parent_validation_accuracy = float(parent_result["selected_validation"]["accuracy"])
component_means = {
    key: float(np.mean([float(item[key]) for item in trainer.training_components]))
    for key in ("loss", "cross_entropy", "pairwise_loss")
}
active_rows_seen = sum(int(item["active_rows"]) for item in trainer.training_components)
result = {
    "experiment": config["experiment"],
    "parent_experiment": config["model"]["parent_experiment"],
    "parent_weight_sha256": parent_result["weight_sha256"],
    "train_rows": len(train_dataset),
    "hard_negative_rows": active_rows,
    "hard_negative_rows_seen_in_batches": active_rows_seen,
    "hard_negative_margin": float(config["hard_negative_loss"]["margin"]),
    "hard_negative_weight": float(config["hard_negative_loss"]["weight"]),
    "validation_rows": len(validation_dataset),
    "test_rows_loaded": 0,
    "test_evaluated": False,
    "epochs": 1,
    "learning_rate": float(config["optimization"]["learning_rate"]),
    "trainable_encoder_layers": [
        config["freeze"]["encoder_layer_start"],
        config["freeze"]["encoder_layer_end"],
    ],
    "trainable_parameters": trainable_parameters,
    "frozen_parameters": frozen_parameters,
    "train_loss": float(train_output.training_loss),
    "training_component_means": component_means,
    "validation_history": validation_history,
    "best_child_epoch": float(best_child["epoch"]),
    "best_child_validation_accuracy": float(best_child["eval_accuracy"]),
    "best_child_validation_macro_f1": float(best_child["eval_macro_f1"]),
    "best_child_checkpoint": str(TRAINER_OUTPUT_DIR / f"checkpoint-{int(best_child['step'])}"),
    "parent_validation_accuracy": parent_validation_accuracy,
    "strict_validation_improvement": float(best_child["eval_accuracy"] - parent_validation_accuracy),
    "train_data_sha256": data_manifest["files"]["train.jsonl"]["sha256"],
    "elapsed_seconds": elapsed_seconds,
    "peak_driver_memory_gib": torch.mps.driver_allocated_memory() / (1024**3),
}
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"train_loss: {result['train_loss']:.6f}")
print(f"training_cross_entropy_mean: {component_means['cross_entropy']:.6f}")
print(f"training_pairwise_loss_mean: {component_means['pairwise_loss']:.6f}")
print(f"hard_negative_rows_seen_in_batches: {active_rows_seen}")
print(f"validation_loss: {best_child['eval_loss']:.6f}")
print(f"validation_accuracy: {best_child['eval_accuracy']:.6f}")
print(f"validation_macro_f1: {best_child['eval_macro_f1']:.6f}")
print(f"parent_validation_accuracy: {parent_validation_accuracy:.6f}")
print(f"strict_validation_improvement: {result['strict_validation_improvement']:+.6f}")
print(f"peak_driver_memory_gib: {result['peak_driver_memory_gib']:.3f}")
print(f"elapsed_seconds: {elapsed_seconds:.3f}")
print("test_rows_loaded: 0")
print("test_evaluated: False")
print(f"result: {RESULT_PATH}")
print("exp_013_training_ok: True")
