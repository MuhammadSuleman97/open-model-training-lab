#!/usr/bin/env python3
"""Evaluate a full-data LoRA experiment on all BANKING77 test rows."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_KEY = os.environ.get("OMTL_FULL_EXPERIMENT", "004c")
CONSTRAINT_MODE = os.environ.get("OMTL_FULL_CONSTRAINT", "none")
if CONSTRAINT_MODE not in {"none", "canonical_labels"}:
    raise SystemExit(f"Unsupported full evaluation constraint: {CONSTRAINT_MODE}")
CONSTRAINED = CONSTRAINT_MODE == "canonical_labels"
BATCH_CONTROL = os.environ.get("OMTL_FULL_BATCH_CONTROL", "standard")
if BATCH_CONTROL not in {"standard", "batch1"}:
    raise SystemExit(f"Unsupported full batch control: {BATCH_CONTROL}")
if CONSTRAINED and BATCH_CONTROL != "standard":
    raise SystemExit("Constrained evaluation cannot select a batch control.")
BATCH_ONE_CONTROL = BATCH_CONTROL == "batch1"
EXPERIMENTS = {
    "004c": {
        "slug": "exp-004c-full-data-batch1-lr5e-7",
        "result_stem": "exp_004c",
        "display": "Experiment 004c",
        "summary_name": "exp-004c-full-data-banking77-full-v1",
    },
    "005b": {
        "slug": "exp-005b-attention-qkvo-lr2p5e-7",
        "result_stem": "exp_005b",
        "display": "Experiment 005b",
        "summary_name": "exp-005b-attention-qkvo-banking77-full-v1",
    },
}
if EXPERIMENT_KEY not in EXPERIMENTS:
    raise SystemExit(f"Unsupported full evaluation experiment: {EXPERIMENT_KEY}")
EXPERIMENT = EXPERIMENTS[EXPERIMENT_KEY]
EXPERIMENT_SLUG = str(EXPERIMENT["slug"])
RESULT_STEM = str(EXPERIMENT["result_stem"])
EXPERIMENT_DISPLAY = str(EXPERIMENT["display"])
OUTPUT_STEM = (
    f"{RESULT_STEM}_constrained"
    if CONSTRAINED
    else f"{RESULT_STEM}_batch1"
    if BATCH_ONE_CONTROL
    else RESULT_STEM
)
RUN_DISPLAY = (
    f"{EXPERIMENT_DISPLAY} canonical-constrained"
    if CONSTRAINED
    else f"{EXPERIMENT_DISPLAY} unconstrained batch-1 control"
    if BATCH_ONE_CONTROL
    else EXPERIMENT_DISPLAY
)
OK_TOKEN = f"{OUTPUT_STEM}_full_evaluation_ok"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "manifest.json"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "banking77" / "raw" / "test.csv"
PROMPT_CONFIG_PATH = PROJECT_ROOT / "configs" / "banking77_prompt.json"
EXPERIMENT_RESULT_PATH = (
    PROJECT_ROOT / "experiments" / EXPERIMENT_SLUG / "result.json"
)
PILOT_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / f"{OUTPUT_STEM}_pilot_summary.json"
)
BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "baseline_summary.json"
)
EXP_003B_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "exp_003b_full_summary.json"
)
EXP_004C_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "exp_004c_full_summary.json"
)
EXP_005B_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "exp_005b_full_summary.json"
)
ADAPTER_PATH = PROJECT_ROOT / "adapters" / EXPERIMENT_SLUG
ADAPTER_FILE = ADAPTER_PATH / "adapters.safetensors"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / f"{OUTPUT_STEM}_full_predictions.jsonl"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / f"{OUTPUT_STEM}_full_summary.json"
)

BATCH_SIZE = 1 if CONSTRAINED or BATCH_ONE_CONTROL else 8
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
    correct = sum(bool(record["correct"]) for record in predictions)
    valid = sum(bool(record["valid_label"]) for record in predictions)
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
pilot_summary = json.loads(PILOT_SUMMARY_PATH.read_text(encoding="utf-8"))
baseline_summary = json.loads(
    BASELINE_SUMMARY_PATH.read_text(encoding="utf-8")
)
exp_003b_summary = json.loads(
    EXP_003B_SUMMARY_PATH.read_text(encoding="utf-8")
)
exp_004c_summary = json.loads(
    EXP_004C_SUMMARY_PATH.read_text(encoding="utf-8")
)
exp_005b_summary = json.loads(
    EXP_005B_SUMMARY_PATH.read_text(encoding="utf-8")
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
        f"{EXPERIMENT_DISPLAY} full evaluation failed: test checksum mismatch."
    )
if len(test_rows) != dataset_manifest["splits"]["test"]["rows"]:
    raise SystemExit(
        f"{EXPERIMENT_DISPLAY} full evaluation failed: test row count mismatch."
    )
if experiment_result.get("status") != "complete":
    raise SystemExit(
        f"{EXPERIMENT_DISPLAY} full evaluation failed: training is incomplete."
    )
if experiment_result.get("nonfinite_adapter_values") != 0:
    raise SystemExit(
        f"{EXPERIMENT_DISPLAY} full evaluation failed: adapter is non-finite."
    )
if sha256(ADAPTER_FILE) != experiment_result["adapter_sha256"]:
    raise SystemExit(
        f"{EXPERIMENT_DISPLAY} full evaluation failed: adapter checksum mismatch."
    )
if pilot_summary["adapter_sha256"] != experiment_result["adapter_sha256"]:
    raise SystemExit(
        f"{EXPERIMENT_DISPLAY} full evaluation failed: pilot used another adapter."
    )
if baseline_summary["test_sha256"] != observed_test_sha256:
    raise SystemExit(
        f"{EXPERIMENT_DISPLAY} full evaluation failed: stale baseline summary."
    )
if exp_003b_summary["test_sha256"] != observed_test_sha256:
    raise SystemExit(
        f"{EXPERIMENT_DISPLAY} full evaluation failed: stale Experiment 003b."
    )
if EXPERIMENT_KEY == "005b" and exp_004c_summary["test_sha256"] != observed_test_sha256:
    raise SystemExit(
        f"{EXPERIMENT_DISPLAY} full evaluation failed: stale Experiment 004c."
    )
if (CONSTRAINED or BATCH_ONE_CONTROL) and exp_005b_summary["test_sha256"] != observed_test_sha256:
    raise SystemExit(
        f"{EXPERIMENT_DISPLAY} full evaluation failed: stale unconstrained 005b."
    )

if "--check" in sys.argv[1:]:
    if sys.argv[1:] != ["--check"]:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} [--check]")
    print(f"Open Model Training Lab — {RUN_DISPLAY} full preflight")
    print(f"test_rows: {len(test_rows)}")
    print(f"test_sha256: {observed_test_sha256}")
    print(f"adapter_sha256: {experiment_result['adapter_sha256']}")
    print(f"pilot_accuracy: {pilot_summary['accuracy']:.6f}")
    comparison_count = 3 if EXPERIMENT_KEY == "005b" else 2
    if CONSTRAINED or BATCH_ONE_CONTROL:
        comparison_count += 1
    print(f"comparison_summaries_verified: {comparison_count}")
    print(f"constraint_mode: {CONSTRAINT_MODE}")
    print(f"batch_control: {BATCH_CONTROL}")
    print("checkpoint_every: 40")
    print(f"{OK_TOKEN.replace('_evaluation', '_preflight')}: True")
    raise SystemExit(0)
if sys.argv[1:]:
    raise SystemExit(f"Usage: {Path(sys.argv[0]).name} [--check]")

import mlx.core as mx
from mlx_lm import batch_generate, load

system_prompt = prompt_config["system_template"].format(
    labels=prompt_config["label_separator"].join(labels)
)
PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
predictions = read_jsonl(PREDICTIONS_PATH)
completed_indices = {int(record["source_index"]) for record in predictions}
if len(completed_indices) != len(predictions):
    raise SystemExit(
        "Experiment 004c full evaluation failed: duplicate checkpoints."
    )
for record in predictions:
    if record["adapter_sha256"] != experiment_result["adapter_sha256"]:
        raise SystemExit(
            "Experiment 004c full evaluation failed: stale adapter results."
        )
    if record["prompt_version"] != prompt_config["prompt_version"]:
        raise SystemExit(
            "Experiment 004c full evaluation failed: stale prompt results."
        )
    if record["test_sha256"] != observed_test_sha256:
        raise SystemExit(
            "Experiment 004c full evaluation failed: stale test results."
        )

predictions_sha256 = write_predictions(predictions)
remaining = [
    record
    for record in test_rows
    if int(record["source_index"]) not in completed_indices
]

print(f"Open Model Training Lab — {RUN_DISPLAY} full test evaluation")
print(f"model_id: {model_manifest['model_id']}")
print(f"adapter_sha256: {experiment_result['adapter_sha256']}")
print(f"training_rows: {experiment_result['training_rows']}")
print(f"learning_rate: {experiment_result['learning_rate']}")
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


def make_canonical_label_processor(
    prompt_length: int,
    label_token_sequences: list[tuple[int, ...]],
    eos_token_ids: tuple[int, ...],
):
    """Restrict each next token to a prefix of an allowed label."""

    def process(tokens, logits):
        generated = tuple(
            int(token) for token in tokens.tolist()[prompt_length:]
        )
        if generated and generated[-1] in eos_token_ids:
            return logits
        allowed: set[int] = set()
        for sequence in label_token_sequences:
            if sequence[: len(generated)] != generated:
                continue
            if len(generated) == len(sequence):
                allowed.update(eos_token_ids)
            else:
                allowed.add(sequence[len(generated)])
        if not allowed:
            raise RuntimeError(
                "Canonical-label constraint reached an invalid token prefix."
            )
        allowed_ids = mx.array(sorted(allowed), dtype=mx.int32)
        masked = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        masked[:, allowed_ids] = logits[:, allowed_ids]
        return masked

    return process


label_token_sequences: list[tuple[int, ...]] = []
eos_token_ids: tuple[int, ...] = ()
if CONSTRAINED:
    label_token_sequences = [
        tuple(tokenizer.encode(label, add_special_tokens=False))
        for label in labels
    ]
    if any(not sequence for sequence in label_token_sequences):
        raise SystemExit("Constrained evaluation failed: an allowed label is empty.")
    if len(set(label_token_sequences)) != len(labels):
        raise SystemExit(
            "Constrained evaluation failed: duplicate label token sequences."
        )
    eos_token_ids = tuple(int(token) for token in tokenizer.eos_token_ids)
    if not eos_token_ids:
        raise SystemExit("Constrained evaluation failed: tokenizer has no EOS token.")

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

    generation_options = {}
    if CONSTRAINED:
        generation_options["logits_processors"] = [
            make_canonical_label_processor(
                len(prompts[0]),
                label_token_sequences,
                eos_token_ids,
            )
        ]
    response = batch_generate(
        model,
        tokenizer,
        prompts,
        max_tokens=prompt_config["max_output_tokens"],
        prefill_batch_size=min(BATCH_SIZE, len(batch)),
        completion_batch_size=BATCH_SIZE,
        **generation_options,
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
                "learning_rate": experiment_result["learning_rate"],
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
        "Experiment 004c full evaluation paused with partial results. "
        "Rerun the same command to resume."
    )

metrics = calculate_metrics(predictions, labels)
comparison_sources = {
    "untouched": baseline_summary,
    "exp_003b_balanced_1925": exp_003b_summary,
}
if EXPERIMENT_KEY == "005b":
    comparison_sources["exp_004c_full_data"] = exp_004c_summary
if CONSTRAINED or BATCH_ONE_CONTROL:
    comparison_sources["exp_005b_unconstrained"] = exp_005b_summary
comparisons = {
    name: {
        "accuracy": source["accuracy"],
        "accuracy_delta": metrics["accuracy"] - source["accuracy"],
        "macro_f1": source["macro_f1"],
        "macro_f1_delta": metrics["macro_f1"] - source["macro_f1"],
        "invalid_label_rate": source["invalid_label_rate"],
        "invalid_label_rate_delta": (
            metrics["invalid_label_rate"] - source["invalid_label_rate"]
        ),
    }
    for name, source in comparison_sources.items()
}
summary = {
    "name": (
        f"{EXPERIMENT['summary_name']}-canonical-constrained"
        if CONSTRAINED
        else EXPERIMENT["summary_name"]
    ),
    "model_id": model_manifest["model_id"],
    "model_revision": model_manifest["resolved_revision"],
    "adapter_sha256": experiment_result["adapter_sha256"],
    "training_rows": experiment_result["training_rows"],
    "learning_rate": experiment_result["learning_rate"],
    "dataset_revision": dataset_manifest["source_revision"],
    "test_sha256": observed_test_sha256,
    "predictions_sha256": predictions_sha256,
    "prompt_version": prompt_config["prompt_version"],
    "constraint_mode": CONSTRAINT_MODE,
    "batch_control": BATCH_CONTROL,
    "batch_size": BATCH_SIZE,
    "elapsed_seconds_this_run": round(elapsed_seconds, 3),
    "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
    "comparisons": comparisons,
    **metrics,
}
SUMMARY_PATH.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"accuracy: {summary['accuracy']:.6f}")
print(f"baseline_accuracy: {baseline_summary['accuracy']:.6f}")
print(
    "accuracy_delta_vs_baseline: "
    f"{comparisons['untouched']['accuracy_delta']:+.6f}"
)
print(f"macro_f1: {summary['macro_f1']:.6f}")
print(f"baseline_macro_f1: {baseline_summary['macro_f1']:.6f}")
print(
    "macro_f1_delta_vs_baseline: "
    f"{comparisons['untouched']['macro_f1_delta']:+.6f}"
)
print(f"invalid_label_rate: {summary['invalid_label_rate']:.6f}")
print(
    "invalid_label_rate_delta_vs_baseline: "
    f"{comparisons['untouched']['invalid_label_rate_delta']:+.6f}"
)
print(
    "accuracy_delta_vs_exp_003b: "
    f"{comparisons['exp_003b_balanced_1925']['accuracy_delta']:+.6f}"
)
if EXPERIMENT_KEY == "005b":
    print(
        "accuracy_delta_vs_exp_004c: "
        f"{comparisons['exp_004c_full_data']['accuracy_delta']:+.6f}"
    )
    print(
        "macro_f1_delta_vs_exp_004c: "
        f"{comparisons['exp_004c_full_data']['macro_f1_delta']:+.6f}"
    )
if CONSTRAINED or BATCH_ONE_CONTROL:
    print(
        "accuracy_delta_vs_unconstrained_005b: "
        f"{comparisons['exp_005b_unconstrained']['accuracy_delta']:+.6f}"
    )
    print(
        "macro_f1_delta_vs_unconstrained_005b: "
        f"{comparisons['exp_005b_unconstrained']['macro_f1_delta']:+.6f}"
    )
print(f"peak_memory_gb: {summary['peak_memory_gb']:.3f}")
print(f"elapsed_seconds: {summary['elapsed_seconds_this_run']:.3f}")
print(f"predictions: {PREDICTIONS_PATH}")
print(f"summary: {SUMMARY_PATH}")
print(f"{OK_TOKEN}: True")
