#!/usr/bin/env python3
"""Run the two-update float32 DeBERTa probe on Exp015's pruned train set."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["OPEN_MODEL_TRAINING_CONFIG"] = str(
    PROJECT_ROOT / "configs" / "exp-015-deberta-noise-pruned.json"
)
os.environ["OPEN_MODEL_TRAINING_DATA_MANIFEST"] = str(
    PROJECT_ROOT / "artifacts" / "encoder" / "exp-015" / "data" / "manifest.json"
)
os.environ["OPEN_MODEL_TRAINING_RESULT_DIR"] = str(
    PROJECT_ROOT / "artifacts" / "encoder" / "exp-015-probe"
)
runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_exp_009_probe.py"), run_name="__main__")
