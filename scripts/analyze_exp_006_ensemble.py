#!/usr/bin/env python3
"""Measure epoch-4/epoch-5 complementarity on validation only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-006-bert-large-classifier.json"
DATA_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "manifest.json"
EXP_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-006"
EPOCH_4_PATH = EXP_DIR / "trainer-output" / "checkpoint-1156"
EPOCH_5_PATH = EXP_DIR / "trainer-output" / "checkpoint-1445"
OUTPUT_PATH = EXP_DIR / "checkpoint_ensemble_analysis.json"


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_probabilities(
    model_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    print(f"loading_checkpoint: {model_path.name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            logits = model(**batch).logits
            batches.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    torch.mps.synchronize()
    del model
    torch.mps.empty_cache()
    return np.concatenate(batches, axis=0)


def metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, average="macro", zero_division=0)
        ),
    }


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit("Ensemble analysis stopped: use .venv-encoder/bin/python.")
if not torch.backends.mps.is_available():
    raise SystemExit("Ensemble analysis stopped: PyTorch MPS is unavailable.")
if OUTPUT_PATH.exists():
    raise SystemExit("Ensemble analysis stopped: result already exists.")
if not EPOCH_4_PATH.is_dir() or not EPOCH_5_PATH.is_dir():
    raise SystemExit("Ensemble analysis stopped: required checkpoints are missing.")

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
validation_file = data_manifest["files"]["valid"]
validation_path = Path(validation_file["path"])
if sha256(validation_path) != validation_file["sha256"]:
    raise SystemExit("Ensemble analysis stopped: validation checksum mismatch.")
with validation_path.open(encoding="utf-8") as handle:
    source_records = [json.loads(line) for line in handle]

tokenizer = AutoTokenizer.from_pretrained(EPOCH_4_PATH, local_files_only=True)
encoded_records: list[dict[str, object]] = []
for record in source_records:
    encoded = tokenizer(
        str(record["text"]),
        max_length=config["tokenization"]["max_length"],
        truncation=True,
    )
    encoded["labels"] = int(record["label"])
    encoded_records.append(encoded)
loader = DataLoader(
    ListDataset(encoded_records),
    batch_size=config["batching"]["eval_batch_size"],
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
)

print("Open Model Training Lab — Experiment 006 checkpoint ensemble analysis")
print(f"validation_rows: {len(source_records)}")
print("test_rows_loaded: 0")
print("ensemble_method: equal_mean_probability")
device = torch.device("mps")
epoch_4_probabilities = predict_probabilities(EPOCH_4_PATH, loader, device)
epoch_5_probabilities = predict_probabilities(EPOCH_5_PATH, loader, device)
expected = np.asarray([int(record["label"]) for record in source_records])
epoch_4_predictions = epoch_4_probabilities.argmax(axis=-1)
epoch_5_predictions = epoch_5_probabilities.argmax(axis=-1)
ensemble_predictions = (
    (epoch_4_probabilities + epoch_5_probabilities) / 2.0
).argmax(axis=-1)

epoch_4_correct = epoch_4_predictions == expected
epoch_5_correct = epoch_5_predictions == expected
ensemble_correct = ensemble_predictions == expected
both_correct = int(np.sum(epoch_4_correct & epoch_5_correct))
epoch_4_only = int(np.sum(epoch_4_correct & ~epoch_5_correct))
epoch_5_only = int(np.sum(~epoch_4_correct & epoch_5_correct))
both_wrong = int(np.sum(~epoch_4_correct & ~epoch_5_correct))
union_correct = both_correct + epoch_4_only + epoch_5_only
ensemble_repairs = int(np.sum(~epoch_4_correct & ensemble_correct))
ensemble_harms = int(np.sum(epoch_4_correct & ~ensemble_correct))
target_correct = int(
    np.ceil(config["evaluation"]["program_launch_target"] * len(expected))
)

result = {
    "experiment": config["experiment"],
    "split": "validation",
    "validation_rows": len(expected),
    "test_rows_loaded": 0,
    "ensemble_method": "equal_mean_probability",
    "epoch_4": metrics(expected, epoch_4_predictions),
    "epoch_5": metrics(expected, epoch_5_predictions),
    "ensemble": metrics(expected, ensemble_predictions),
    "transitions": {
        "both_correct": both_correct,
        "epoch_4_only_correct": epoch_4_only,
        "epoch_5_only_correct": epoch_5_only,
        "both_wrong": both_wrong,
        "prediction_disagreements": int(
            np.sum(epoch_4_predictions != epoch_5_predictions)
        ),
        "ensemble_repairs_vs_epoch_4": ensemble_repairs,
        "ensemble_harms_vs_epoch_4": ensemble_harms,
    },
    "oracle_union_correct": union_correct,
    "oracle_union_accuracy": union_correct / len(expected),
    "target_correct": target_correct,
    "ensemble_additional_correct_needed": (
        target_correct - int(np.sum(ensemble_correct))
    ),
    "target_reached": int(np.sum(ensemble_correct)) >= target_correct,
}
OUTPUT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"epoch_4_accuracy: {result['epoch_4']['accuracy']:.6f}")
print(f"epoch_5_accuracy: {result['epoch_5']['accuracy']:.6f}")
print(f"ensemble_accuracy: {result['ensemble']['accuracy']:.6f}")
print(f"ensemble_macro_f1: {result['ensemble']['macro_f1']:.6f}")
print(f"prediction_disagreements: {result['transitions']['prediction_disagreements']}")
print(f"epoch_4_only_correct: {epoch_4_only}")
print(f"epoch_5_only_correct: {epoch_5_only}")
print(f"oracle_union_accuracy: {result['oracle_union_accuracy']:.6f}")
print(f"ensemble_repairs_vs_epoch_4: {ensemble_repairs}")
print(f"ensemble_harms_vs_epoch_4: {ensemble_harms}")
print(
    "ensemble_additional_correct_needed: "
    f"{result['ensemble_additional_correct_needed']}"
)
print(f"target_reached: {result['target_reached']}")
print(f"analysis: {OUTPUT_PATH}")
print("test_evaluated: False")
print("checkpoint_ensemble_analysis_ok: True")
