#!/usr/bin/env python3
"""Evaluate a full-data LoRA experiment on the fixed BANKING77 pilot."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_KEY = os.environ.get("OMTL_PILOT_EXPERIMENT", "004c")
CONSTRAINT_MODE = os.environ.get("OMTL_PILOT_CONSTRAINT", "none")
if CONSTRAINT_MODE not in {"none", "canonical_labels"}:
    raise SystemExit(f"Unsupported pilot constraint: {CONSTRAINT_MODE}")
CONSTRAINED = CONSTRAINT_MODE == "canonical_labels"
BATCH_CONTROL = os.environ.get("OMTL_PILOT_BATCH_CONTROL", "standard")
if BATCH_CONTROL not in {"standard", "batch1"}:
    raise SystemExit(f"Unsupported pilot batch control: {BATCH_CONTROL}")
if CONSTRAINED and BATCH_CONTROL != "standard":
    raise SystemExit("Constrained pilot cannot also select a batch control.")
BATCH_ONE_CONTROL = BATCH_CONTROL == "batch1"
EXPERIMENTS = {
    "004c": {
        "slug": "exp-004c-full-data-batch1-lr5e-7",
        "result_stem": "exp_004c",
        "display": "Experiment 004c",
        "summary_name": "exp-004c-full-data-banking77-pilot-v1",
    },
    "005b": {
        "slug": "exp-005b-attention-qkvo-lr2p5e-7",
        "result_stem": "exp_005b",
        "display": "Experiment 005b",
        "summary_name": "exp-005b-attention-qkvo-banking77-pilot-v1",
    },
}
if EXPERIMENT_KEY not in EXPERIMENTS:
    raise SystemExit(f"Unsupported pilot experiment: {EXPERIMENT_KEY}")
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
OK_TOKEN = f"{OUTPUT_STEM}_pilot_evaluation_ok"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "manifest.json"
PILOT_PATH = PROJECT_ROOT / "evaluation" / "pilot.jsonl"
PILOT_MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "pilot_manifest.json"
PROMPT_CONFIG_PATH = PROJECT_ROOT / "configs" / "banking77_prompt.json"
EXPERIMENT_RESULT_PATH = (
    PROJECT_ROOT / "experiments" / EXPERIMENT_SLUG / "result.json"
)
BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "pilot_summary.json"
)
EXP_002_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "exp_002_pilot_summary.json"
)
EXP_003B_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "exp_003b_pilot_summary.json"
)
EXP_004C_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "exp_004c_pilot_summary.json"
)
EXP_005B_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "exp_005b_pilot_summary.json"
)
ADAPTER_PATH = PROJECT_ROOT / "adapters" / EXPERIMENT_SLUG
ADAPTER_FILE = ADAPTER_PATH / "adapters.safetensors"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / f"{OUTPUT_STEM}_pilot_predictions.jsonl"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / f"{OUTPUT_STEM}_pilot_summary.json"
)

BATCH_SIZE = 1 if CONSTRAINED or BATCH_ONE_CONTROL else 8


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
            key=lambda item: int(item["pilot_position"]),
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
            for (expected, predicted), count in confusions.most_common(15)
        ],
    }


model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
dataset_manifest = json.loads(
    DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
)
pilot_manifest = json.loads(PILOT_MANIFEST_PATH.read_text(encoding="utf-8"))
prompt_config = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
experiment_result = json.loads(
    EXPERIMENT_RESULT_PATH.read_text(encoding="utf-8")
)
baseline_summary = json.loads(
    BASELINE_SUMMARY_PATH.read_text(encoding="utf-8")
)
exp_002_summary = json.loads(
    EXP_002_SUMMARY_PATH.read_text(encoding="utf-8")
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
pilot = read_jsonl(PILOT_PATH)
labels = dataset_manifest["labels"]
pilot_sha256 = sha256(PILOT_PATH)

if pilot_sha256 != pilot_manifest["pilot_sha256"]:
    raise SystemExit(f"{EXPERIMENT_DISPLAY} pilot failed: pilot checksum mismatch.")
if len(pilot) != pilot_manifest["pilot_rows"]:
    raise SystemExit(f"{EXPERIMENT_DISPLAY} pilot failed: pilot row count mismatch.")
if experiment_result.get("status") != "complete":
    raise SystemExit(f"{EXPERIMENT_DISPLAY} pilot failed: training is incomplete.")
if experiment_result.get("nonfinite_adapter_values") != 0:
    raise SystemExit(f"{EXPERIMENT_DISPLAY} pilot failed: adapter is non-finite.")
if sha256(ADAPTER_FILE) != experiment_result["adapter_sha256"]:
    raise SystemExit(f"{EXPERIMENT_DISPLAY} pilot failed: adapter checksum mismatch.")
comparison_summaries = [
    ("untouched", baseline_summary),
    ("Experiment 002", exp_002_summary),
    ("Experiment 003b", exp_003b_summary),
]
if EXPERIMENT_KEY == "005b":
    comparison_summaries.append(("Experiment 004c", exp_004c_summary))
if CONSTRAINED or BATCH_ONE_CONTROL:
    comparison_summaries.append(("Experiment 005b unconstrained", exp_005b_summary))
for name, summary in comparison_summaries:
    if summary["pilot_sha256"] != pilot_sha256:
        raise SystemExit(
            f"{EXPERIMENT_DISPLAY} pilot failed: stale {name} summary."
        )

if "--check" in sys.argv[1:]:
    if sys.argv[1:] != ["--check"]:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} [--check]")
    print(f"Open Model Training Lab — {RUN_DISPLAY} pilot preflight")
    print(f"pilot_rows: {len(pilot)}")
    print(f"adapter_sha256: {experiment_result['adapter_sha256']}")
    print(f"comparison_summaries_verified: {len(comparison_summaries)}")
    print(f"constraint_mode: {CONSTRAINT_MODE}")
    print(f"batch_control: {BATCH_CONTROL}")
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
completed_positions = {int(record["pilot_position"]) for record in predictions}
if len(completed_positions) != len(predictions):
    raise SystemExit(
        "Experiment 004c pilot failed: duplicate checkpoint rows detected."
    )
for record in predictions:
    if record["adapter_sha256"] != experiment_result["adapter_sha256"]:
        raise SystemExit(
            "Experiment 004c pilot failed: stale adapter results detected."
        )
    if record["prompt_version"] != prompt_config["prompt_version"]:
        raise SystemExit(
            "Experiment 004c pilot failed: stale prompt results detected."
        )
    if record["pilot_sha256"] != pilot_sha256:
        raise SystemExit(
            "Experiment 004c pilot failed: stale pilot results detected."
        )

predictions_sha256 = write_predictions(predictions)
remaining = [
    (position, record)
    for position, record in enumerate(pilot)
    if position not in completed_positions
]

print(f"Open Model Training Lab — {RUN_DISPLAY} balanced pilot")
print(f"model_id: {model_manifest['model_id']}")
print(f"adapter_sha256: {experiment_result['adapter_sha256']}")
print(f"training_rows: {experiment_result['training_rows']}")
print(f"learning_rate: {experiment_result['learning_rate']}")
print(f"pilot_rows: {len(pilot)}")
print(f"already_completed: {len(predictions)}")
print(f"remaining: {len(remaining)}")
print(f"batch_size: {BATCH_SIZE}")
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
        # MLX-LM pipelines one token ahead: it invokes processors once more
        # with the stop token in the context before removing the completed
        # sequence. That next-token result is discarded, so pass it through.
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
        # MLX 0.32 supports indexed assignment for replacement. Its ``.at``
        # helper exposes reductions such as ``add`` but has no JAX-style
        # ``set`` method.
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
        raise SystemExit("Constrained pilot failed: an allowed label is empty.")
    if len(set(label_token_sequences)) != len(labels):
        raise SystemExit(
            "Constrained pilot failed: allowed labels have duplicate token sequences."
        )
    eos_token_ids = tuple(int(token) for token in tokenizer.eos_token_ids)
    if not eos_token_ids:
        raise SystemExit("Constrained pilot failed: tokenizer has no EOS token.")

for batch_start in range(0, len(remaining), BATCH_SIZE):
    batch = remaining[batch_start : batch_start + BATCH_SIZE]
    prompts = []
    for _, record in batch:
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
    for (pilot_position, record), raw_output in zip(batch, response.texts):
        prediction = raw_output.strip()
        expected = str(record["expected"])
        predictions.append(
            {
                "pilot_position": pilot_position,
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
                "pilot_sha256": pilot_sha256,
                "prompt_version": prompt_config["prompt_version"],
            }
        )
    predictions_sha256 = write_predictions(predictions)
    print(f"completed: {len(predictions)}/{len(pilot)}")

elapsed_seconds = time.perf_counter() - started_at
if len(predictions) != len(pilot):
    raise SystemExit(
        "Experiment 004c pilot paused with partial results. Rerun the same "
        "command to resume."
    )

metrics = calculate_metrics(predictions, labels)
comparison_sources = {
    "untouched": baseline_summary,
    "exp_002_balanced_539": exp_002_summary,
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
    "pilot_sha256": pilot_sha256,
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
    "exp_003b_accuracy_delta: "
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
print(f"predictions: {PREDICTIONS_PATH}")
print(f"summary: {SUMMARY_PATH}")
print(f"{OK_TOKEN}: True")
