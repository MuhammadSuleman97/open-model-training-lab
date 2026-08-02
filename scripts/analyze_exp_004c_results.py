#!/usr/bin/env python3
"""Create paired error analysis for Experiment 004c."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
BASELINE_PATH = RESULTS_DIR / "baseline_predictions.jsonl"
EXP_003B_PATH = RESULTS_DIR / "exp_003b_full_predictions.jsonl"
EXP_004C_PATH = RESULTS_DIR / "exp_004c_full_predictions.jsonl"
SUMMARY_PATH = RESULTS_DIR / "exp_004c_full_summary.json"
OUTPUT_PATH = RESULTS_DIR / "exp_004c_paired_analysis.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(rows, key=lambda row: int(row["source_index"]))


def compare(
    previous: list[dict[str, object]],
    current: list[dict[str, object]],
) -> dict[str, object]:
    transitions: Counter[str] = Counter()
    gains: defaultdict[str, int] = defaultdict(int)
    losses: defaultdict[str, int] = defaultdict(int)
    changed_predictions = 0

    for old, new in zip(previous, current):
        old_state = "correct" if old["correct"] else "wrong"
        new_state = "correct" if new["correct"] else "wrong"
        transitions[f"{old_state}_to_{new_state}"] += 1
        changed_predictions += old["prediction"] != new["prediction"]
        label = str(new["expected"])
        if not old["correct"] and new["correct"]:
            gains[label] += 1
        if old["correct"] and not new["correct"]:
            losses[label] += 1

    per_label = [
        {
            "label": label,
            "wrong_to_correct": gains[label],
            "correct_to_wrong": losses[label],
            "net_correct_change": gains[label] - losses[label],
        }
        for label in sorted(set(gains) | set(losses))
    ]
    return {
        "transitions": dict(transitions),
        "changed_predictions": changed_predictions,
        "largest_net_gains": sorted(
            per_label,
            key=lambda row: (
                -int(row["net_correct_change"]),
                str(row["label"]),
            ),
        )[:12],
        "largest_net_losses": sorted(
            per_label,
            key=lambda row: (
                int(row["net_correct_change"]),
                str(row["label"]),
            ),
        )[:12],
    }


baseline = read_jsonl(BASELINE_PATH)
exp_003b = read_jsonl(EXP_003B_PATH)
exp_004c = read_jsonl(EXP_004C_PATH)
summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

if not len(baseline) == len(exp_003b) == len(exp_004c) == 3080:
    raise SystemExit("Experiment 004c analysis failed: row count mismatch.")
for old, middle, current in zip(baseline, exp_003b, exp_004c):
    if not (
        old["source_index"] == middle["source_index"] == current["source_index"]
        and old["expected"] == middle["expected"] == current["expected"]
    ):
        raise SystemExit(
            "Experiment 004c analysis failed: paired rows are misaligned."
        )
if sha256(EXP_004C_PATH) != summary["predictions_sha256"]:
    raise SystemExit(
        "Experiment 004c analysis failed: prediction checksum mismatch."
    )

baseline_comparison = compare(baseline, exp_004c)
exp_003b_comparison = compare(exp_003b, exp_004c)
discordant = baseline_comparison["transitions"]
wrong_to_correct = int(discordant["wrong_to_correct"])
correct_to_wrong = int(discordant["correct_to_wrong"])
discordant_total = wrong_to_correct + correct_to_wrong
exact_p_value = min(
    1.0,
    2
    * sum(
        math.comb(discordant_total, k)
        for k in range(min(wrong_to_correct, correct_to_wrong) + 1)
    )
    / (2**discordant_total),
)
invalid_predictions = Counter(
    str(record["prediction"])
    for record in exp_004c
    if not record["valid_label"]
)

analysis = {
    "name": "exp-004c-paired-error-analysis-v1",
    "test_rows": len(exp_004c),
    "test_sha256": summary["test_sha256"],
    "predictions_sha256": summary["predictions_sha256"],
    "accuracy": summary["accuracy"],
    "macro_f1": summary["macro_f1"],
    "invalid_labels": summary["invalid_labels"],
    "comparison_to_untouched": baseline_comparison,
    "comparison_to_exp_003b": exp_003b_comparison,
    "exact_mcnemar": {
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "discordant_total": discordant_total,
        "two_sided_p_value": exact_p_value,
    },
    "top_invalid_outputs": [
        {"prediction": prediction, "count": count}
        for prediction, count in invalid_predictions.most_common(15)
    ],
}
OUTPUT_PATH.write_text(
    json.dumps(analysis, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — Experiment 004c paired analysis")
print(f"correct: {summary['correct']}/{summary['total']}")
print(
    "baseline_wrong_to_correct: "
    f"{baseline_comparison['transitions']['wrong_to_correct']}"
)
print(
    "baseline_correct_to_wrong: "
    f"{baseline_comparison['transitions']['correct_to_wrong']}"
)
print(f"exact_mcnemar_p_value: {exact_p_value:.6f}")
print(
    "largest_gain: "
    f"{baseline_comparison['largest_net_gains'][0]}"
)
print(
    "largest_loss: "
    f"{baseline_comparison['largest_net_losses'][0]}"
)
print(f"analysis: {OUTPUT_PATH}")
print("exp_004c_analysis_ok: True")
