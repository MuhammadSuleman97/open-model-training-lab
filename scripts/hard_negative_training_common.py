"""Shared train-time helpers for the guarded hard-negative experiments."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding, EvalPrediction, Trainer, TrainerCallback


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


class HardNegativeCollator:
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


def sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path, expected_hash: str) -> list[dict[str, object]]:
    if sha256(path) != expected_hash:
        raise SystemExit(f"checksum mismatch for {path.name}.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def encode_records(records, tokenizer, max_length: int, label2id: dict[str, int], include_rivals: bool) -> ListDataset:
    encoded_records: list[dict[str, object]] = []
    for record in records:
        encoded = tokenizer(str(record["text"]), max_length=max_length, truncation=True)
        encoded["labels"] = int(record["label"])
        if include_rivals:
            active = bool(record["hard_negative_active"])
            rival_name = record.get("hard_negative_label")
            if active and (not isinstance(rival_name, str) or rival_name not in label2id):
                raise SystemExit("active rival label is unknown.")
            if active and int(record["label"]) == label2id[rival_name]:
                raise SystemExit("rival label equals the true label.")
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


def hard_negative_loss(logits, labels, hard_negative_labels, margin: float, weight: float):
    cross_entropy = F.cross_entropy(logits, labels)
    active = hard_negative_labels >= 0
    if bool(active.any()):
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        rival_logits = logits.gather(1, hard_negative_labels.clamp_min(0).unsqueeze(1)).squeeze(1)
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
            raise RuntimeError("labels are missing from a batch.")
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
            raise RuntimeError(f"non-finite loss ({float(loss.detach().cpu())}).")
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
