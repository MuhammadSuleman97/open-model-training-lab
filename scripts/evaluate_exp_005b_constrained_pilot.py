#!/usr/bin/env python3
"""Pilot Experiment 005b with canonical BANKING77 label constraints."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["OMTL_PILOT_EXPERIMENT"] = "005b"
os.environ["OMTL_PILOT_CONSTRAINT"] = "canonical_labels"
runpy.run_path(
    str(Path(__file__).with_name("evaluate_exp_004c_pilot.py")),
    run_name="__main__",
)
