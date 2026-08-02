#!/usr/bin/env python3
"""Locate the next non-finite value with DeBERTa embeddings frozen."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["OPEN_MODEL_TRAINING_CONFIG"] = str(
    PROJECT_ROOT / "configs" / "exp-009c-deberta-v3-large-frozen-embeddings.json"
)
os.environ["OPEN_MODEL_TRAINING_RESULT_DIR"] = str(
    PROJECT_ROOT / "artifacts" / "encoder" / "exp-009c-diagnosis"
)
runpy.run_path(str(PROJECT_ROOT / "scripts" / "diagnose_exp_009_nan.py"), run_name="__main__")
