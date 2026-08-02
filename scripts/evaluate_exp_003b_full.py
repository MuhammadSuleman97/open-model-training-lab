#!/usr/bin/env python3
"""Evaluate stable Experiment 003b on all 3,080 BANKING77 test rows."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "manifest.json"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "banking77" / "raw" / "test.csv"
PROMPT_CONFIG_PATH = PROJECT_ROOT / "configs" / "banking77_prompt.json"
EXPERIMENT_RESULT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "exp-003b-balanced-1925-lr2p5e-6"
    / "result.json"
)
BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "baseline_summary.json"
)
ADAPTER_PATH = (
    PROJECT_ROOT / "adapters" / "exp-003b-balanced-1925-lr2p5e-6"
)
ADAPTER_FILE = ADAPTER_PATH / "adapters.safetensors"
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "exp_003b_full_predictions.jsonl"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "exp_003b_full_summary.json"
)

BATCH_SIZE = 8
CHECKPOINT_EVERY = 40


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_predictions(predictions: list[dict[str, object]]) -> str:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in sorted(
            predictions,
            key=lambda item: int(item["source_index"]),
        )
    )
    temporary_path = PREDICTIONS_PATH.with_suffix(".jsonl.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(PREDICTIONS_PATH)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def calculate_metrics(
    predictions: list[dict[str, object]],
    labels: list[str],
) -> dict[str, object]:
    total = len(predictions)
    valid = sum(bool(record["valid_label"]) for record in predictions)
    correct = sum(bool(record["correct"]) for record in predictions)

    f1_by_label: dict[str, float] = {}
    accuracy_by_label: dict[str, float] = {}
    for label in labels:
        tp = sum(
            record["expected"] == label and record["prediction"] == label
            for record in predictions
        )
        fp = sum(
            record["expected"] != label and record["prediction"] == label
            for record in predictions
        )
        fn = sum(
            record["expected"] == label and record["prediction"] != label
            for record in predictions
        )
        denominator = (2 * tp) + fp + fn
        f1_by_label[label] = (2 * tp / denominator) if denominator else 0.0

        label_records = [
            record for record in predictions if record["expected"] == label
        ]
        accuracy_by_label[label] = (
            sum(bool(record["correct"]) for record in label_records)
            / len(label_records)
        )

    confusions = Counter(
        (str(record["expected"]), str(record["prediction"]))
        for record in predictions
        if not record["correct"]
    )

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "macro_f1": sum(f1_by_label.values()) / len(labels),
        "valid_labels": valid,
        "invalid_labels": total - valid,
        "invalid_label_rate": (total - valid) / total,
        "accuracy_by_label": accuracy_by_label,
        "f1_by_label": f1_by_label,
        "top_confusions": [
            {
                "expected": expected,
                "predicted": predicted,
                "count": count,
            }
            for (expected, predicted), count in confusions.most_common(25)
        ],
    }


model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
dataset_manifest = json.loads(
    DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
)
prompt_config = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
experiment_result = json.loads(
    EXPERIMENT_RESULT_PATH.read_text(encoding="utf-8")
)
baseline_summary = json.loads(
    BASELINE_SUMMARY_PATH.read_text(encoding="utf-8")
)
labels = dataset_manifest["labels"]

with TEST_DATA_PATH.open(encoding="utf-8", newline="") as handle:
    test_rows = [
        {
            "source_index": source_index,
            "text": row["text"],
            "expected": row["category"],
        }
        for source_index, row in enumerate(csv.DictReader(handle))
    ]

observed_test_sha256 = sha256(TEST_DATA_PATH)
if observed_test_sha256 != dataset_manifest["files"]["test.csv"]["sha256"]:
    raise SystemExit(
        "Experiment 003b full evaluation failed: test checksum mismatch."
    )
if len(test_rows) != dataset_manifest["splits"]["test"]["rows"]:
    raise SystemExit(
        "Experiment 003b full evaluation failed: test row count mismatch."
    )
if experiment_result["status"] != "complete":
    raise SystemExit(
        "Experiment 003b full evaluation failed: training is incomplete."
    )
if sha256(ADAPTER_FILE) != experiment_result["adapter_sha256"]:
    raise SystemExit(
        "Experiment 003b full evaluation failed: adapter checksum mismatch."
    )
if baseline_summary["test_sha256"] != observed_test_sha256:
    raise SystemExit(
        "Experiment 003b full evaluation failed: stale baseline summary."
    )

system_prompt = prompt_config["system_template"].format(
    labels=prompt_config["label_separator"].join(labels)
)

PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
predictions = read_jsonl(PREDICTIONS_PATH)
completed_indices = {int(record["source_index"]) for record in predictions}
if len(completed_indices) != len(predictions):
    raise SystemExit(
        "Experiment 003b full evaluation failed: duplicate checkpoints."
    )

for record in predictions:
    if record["adapter_sha256"] != experiment_result["adapter_sha256"]:
        raise SystemExit(
            "Experiment 003b full evaluation failed: stale adapter results."
        )
    if record["prompt_version"] != prompt_config["prompt_version"]:
        raise SystemExit(
            "Experiment 003b full evaluation failed: stale prompt results."
        )
    if record["test_sha256"] != observed_test_sha256:
        raise SystemExit(
            "Experiment 003b full evaluation failed: stale test results."
        )

predictions_sha256 = write_predictions(predictions)
remaining = [
    record
    for record in test_rows
    if int(record["source_index"]) not in completed_indices
]

print("Open Model Training Lab — Experiment 003b full test evaluation")
print(f"model_id: {model_manifest['model_id']}")
print(f"adapter_sha256: {experiment_result['adapter_sha256']}")
print(f"training_rows: {experiment_result['training_rows']}")
print(f"learning_rate: {experiment_result['changed_parameter']['retry_value']}")
print(f"test_rows: {len(test_rows)}")
print(f"already_completed: {len(predictions)}")
print(f"remaining: {len(remaining)}")
print(f"batch_size: {BATCH_SIZE}")
print(f"checkpoint_every: {CHECKPOINT_EVERY}")
print("loading_model_and_adapter: True")

started_at = time.perf_counter()
model, tokenizer = load(
    model_manifest["snapshot_path"],
    adapter_path=str(ADAPTER_PATH),
)

for batch_start in range(0, len(remaining), BATCH_SIZE):
    batch = remaining[batch_start : batch_start + BATCH_SIZE]
    prompts = []
    for record in batch:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["text"]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=prompt_config["enable_thinking"],
            )
        )

    response = batch_generate(
        model,
        tokenizer,
        prompts,
        max_tokens=prompt_config["max_output_tokens"],
        prefill_batch_size=min(BATCH_SIZE, len(batch)),
        completion_batch_size=BATCH_SIZE,
    )

    for record, raw_output in zip(batch, response.texts):
        prediction = raw_output.strip()
        expected = str(record["expected"])
        predictions.append(
            {
                "source_index": record["source_index"],
                "text": record["text"],
                "expected": expected,
                "raw_output": raw_output,
                "prediction": prediction,
                "valid_label": prediction in labels,
                "correct": prediction == expected,
                "model_id": model_manifest["model_id"],
                "model_revision": model_manifest["resolved_revision"],
                "adapter_sha256": experiment_result["adapter_sha256"],
                "training_rows": experiment_result["training_rows"],
                "learning_rate": (
                    experiment_result["changed_parameter"]["retry_value"]
                ),
                "dataset_revision": dataset_manifest["source_revision"],
                "test_sha256": observed_test_sha256,
                "prompt_version": prompt_config["prompt_version"],
            }
        )

    if (
        len(predictions) % CHECKPOINT_EVERY == 0
        or len(predictions) == len(test_rows)
    ):
        predictions_sha256 = write_predictions(predictions)
        print(f"completed: {len(predictions)}/{len(test_rows)}")

elapsed_seconds = time.perf_counter() - started_at
if len(predictions) != len(test_rows):
    raise SystemExit(
        "Experiment 003b full evaluation paused with partial results. "
        "Rerun the same command to resume."
    )

metrics = calculate_metrics(predictions, labels)
comparison = {
    "baseline_accuracy": baseline_summary["accuracy"],
    "accuracy_delta": metrics["accuracy"] - baseline_summary["accuracy"],
    "baseline_macro_f1": baseline_summary["macro_f1"],
    "macro_f1_delta": metrics["macro_f1"] - baseline_summary["macro_f1"],
    "baseline_invalid_label_rate": baseline_summary["invalid_label_rate"],
    "invalid_label_rate_delta": (
        metrics["invalid_label_rate"]
        - baseline_summary["invalid_label_rate"]
    ),
}
summary = {
    "name": "exp-003b-balanced-1925-banking77-full-v1",
    "model_id": model_manifest["model_id"],
    "model_revision": model_manifest["resolved_revision"],
    "adapter_sha256": experiment_result["adapter_sha256"],
    "training_rows": experiment_result["training_rows"],
    "learning_rate": experiment_result["changed_parameter"]["retry_value"],
    "dataset_revision": dataset_manifest["source_revision"],
    "test_sha256": observed_test_sha256,
    "predictions_sha256": predictions_sha256,
    "prompt_version": prompt_config["prompt_version"],
    "batch_size": BATCH_SIZE,
    "elapsed_seconds_this_run": round(elapsed_seconds, 3),
    "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
    "comparison_to_untouched": comparison,
    **metrics,
}
SUMMARY_PATH.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"accuracy: {summary['accuracy']:.6f}")
print(f"baseline_accuracy: {comparison['baseline_accuracy']:.6f}")
print(f"accuracy_delta: {comparison['accuracy_delta']:+.6f}")
print(f"macro_f1: {summary['macro_f1']:.6f}")
print(f"baseline_macro_f1: {comparison['baseline_macro_f1']:.6f}")
print(f"macro_f1_delta: {comparison['macro_f1_delta']:+.6f}")
print(f"invalid_label_rate: {summary['invalid_label_rate']:.6f}")
print(
    "baseline_invalid_label_rate: "
    f"{comparison['baseline_invalid_label_rate']:.6f}"
)
print(
    "invalid_label_rate_delta: "
    f"{comparison['invalid_label_rate_delta']:+.6f}"
)
print(f"peak_memory_gb: {summary['peak_memory_gb']:.3f}")
print(f"elapsed_seconds: {summary['elapsed_seconds_this_run']:.3f}")
print(f"predictions: {PREDICTIONS_PATH}")
print(f"summary: {SUMMARY_PATH}")
print("exp_003b_full_evaluation_ok: True")
