#!/usr/bin/env python3
"""Summarize the already-completed Experiment 009d test evaluation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "test_result.json"
PREDICTIONS_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "test_predictions.jsonl"
ANALYSIS_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d" / "test_error_analysis.json"


if not RESULT_PATH.is_file() or not PREDICTIONS_PATH.is_file():
    raise SystemExit("Experiment 009d analysis stopped: run the sealed test evaluation first.")
if ANALYSIS_PATH.exists():
    raise SystemExit("Experiment 009d analysis stopped: analysis already exists.")

result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
predictions = [json.loads(line) for line in PREDICTIONS_PATH.open(encoding="utf-8")]
errors = [record for record in predictions if not record["correct"]]
target_correct = round(len(predictions) * result["program_launch_target"])
correct = len(predictions) - len(errors)

confusions = Counter(
    (record["expected_label"], record["predicted_label"])
    for record in errors
)
per_label = []
for label, stats in result["per_label"].items():
    per_label.append(
        {
            "label": label,
            "correct": stats["correct"],
            "support": stats["support"],
            "accuracy": stats["accuracy"],
            "errors": stats["support"] - stats["correct"],
        }
    )
per_label.sort(key=lambda item: (item["accuracy"], item["label"]))

analysis = {
    "experiment": result["experiment"],
    "test_rows": len(predictions),
    "correct": correct,
    "errors": len(errors),
    "test_accuracy": result["metrics"]["accuracy"],
    "test_macro_f1": result["metrics"]["macro_f1"],
    "program_launch_target": result["program_launch_target"],
    "target_correct": target_correct,
    "additional_correct_needed": target_correct - correct,
    "worst_labels": per_label[:15],
    "top_confusions": [
        {
            "expected_label": expected,
            "predicted_label": predicted,
            "count": count,
        }
        for (expected, predicted), count in confusions.most_common(20)
    ],
    "test_result": str(RESULT_PATH),
    "predictions": str(PREDICTIONS_PATH),
}
ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print("Open Model Training Lab — Experiment 009d test error analysis")
print(f"test_rows: {analysis['test_rows']}")
print(f"correct: {analysis['correct']}")
print(f"errors: {analysis['errors']}")
print(f"test_accuracy: {analysis['test_accuracy']:.6f}")
print(f"test_macro_f1: {analysis['test_macro_f1']:.6f}")
print(f"target_correct: {analysis['target_correct']}")
print(f"additional_correct_needed: {analysis['additional_correct_needed']}")
print("worst_labels:")
for item in analysis["worst_labels"][:12]:
    print(f"  {item['label']}: {item['correct']}/{item['support']} ({item['accuracy']:.3f})")
print("top_confusions:")
for item in analysis["top_confusions"][:12]:
    print(f"  {item['expected_label']} -> {item['predicted_label']}: {item['count']}")
print(f"analysis: {ANALYSIS_PATH}")
print("test_error_analysis_ok: True")
