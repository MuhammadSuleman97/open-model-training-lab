#!/usr/bin/env python3
"""Evaluate Experiment 005b using the shared fixed-pilot evaluator."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["OMTL_PILOT_EXPERIMENT"] = "005b"
runpy.run_path(
    str(Path(__file__).with_name("evaluate_exp_004c_pilot.py")),
    run_name="__main__",
)
