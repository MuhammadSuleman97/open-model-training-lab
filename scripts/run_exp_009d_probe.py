#!/usr/bin/env python3
"""Run DeBERTa's stability probe after casting the model to float32."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["OPEN_MODEL_TRAINING_CONFIG"] = str(
    PROJECT_ROOT / "configs" / "exp-009d-deberta-v3-large-float32.json"
)
os.environ["OPEN_MODEL_TRAINING_RESULT_DIR"] = str(
    PROJECT_ROOT / "artifacts" / "encoder" / "exp-009d-probe"
)
runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_exp_009_probe.py"), run_name="__main__")
