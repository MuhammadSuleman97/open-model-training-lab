#!/usr/bin/env python3
"""Record precise evidence for the failed Experiment 005 full run."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "exp-005-attention-qkvo-full"
)
LOG_PATH = EXPERIMENT_DIR / "training.log"
RESULT_PATH = EXPERIMENT_DIR / "result.json"
ADAPTER_DIR = PROJECT_ROOT / "adapters" / "exp-005-attention-qkvo-full"

result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
if result.get("status") != "failed_numerical_instability":
    raise SystemExit("Experiment 005 diagnosis failed: status is unexpected.")

reports = [
    {
        "iteration": int(iteration),
        "loss": float(loss),
        "peak_memory_gb": float(memory),
    }
    for iteration, loss, memory in re.findall(
        r"Iter (\d+): Train loss ([^,]+),.*?Peak mem ([0-9.]+) GB",
        LOG_PATH.read_text(encoding="utf-8"),
    )
]
finite = [report for report in reports if math.isfinite(report["loss"])]
nonfinite = [report for report in reports if not math.isfinite(report["loss"])]
if not finite or not nonfinite:
    raise SystemExit(
        "Experiment 005 diagnosis failed: expected loss evidence is missing."
    )
first_nonfinite = nonfinite[0]
if first_nonfinite["iteration"] != 2050:
    raise SystemExit(
        "Experiment 005 diagnosis failed: failure iteration changed."
    )

weight_files = sorted(path.name for path in ADAPTER_DIR.glob("*.safetensors"))
result.update(
    {
        "artifact_usable": False,
        "adapter_saved": bool(weight_files),
        "adapter_weight_files": weight_files,
        "first_nonfinite_training_report": {
            "iteration": first_nonfinite["iteration"],
            "loss": "nan",
            "peak_memory_gb": first_nonfinite["peak_memory_gb"],
        },
        "last_finite_training_report": finite[-1],
        "minimum_finite_training_report": min(
            finite,
            key=lambda report: report["loss"],
        ),
        "peak_memory_gb": max(report["peak_memory_gb"] for report in reports),
        "diagnosis": (
            "Numerical instability after 2,000 finite updates; not unified-"
            "memory exhaustion. No adapter weights were saved."
        ),
    }
)
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — Experiment 005 failure diagnosis")
print("status: failed_numerical_instability")
print(f"last_finite_training_report: {finite[-1]}")
print(
    "first_nonfinite_training_report: "
    f"{result['first_nonfinite_training_report']}"
)
print(f"peak_memory_gb: {result['peak_memory_gb']}")
print(f"adapter_weight_files: {weight_files}")
print("artifact_usable: False")
print(f"result: {RESULT_PATH}")
print("exp_005_failure_diagnosis_ok: True")
