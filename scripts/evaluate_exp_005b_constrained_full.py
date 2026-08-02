#!/usr/bin/env python3
"""Evaluate Experiment 005b with canonical label constraints on all tests."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ["OMTL_FULL_EXPERIMENT"] = "005b"
os.environ["OMTL_FULL_CONSTRAINT"] = "canonical_labels"
runpy.run_path(
    str(Path(__file__).with_name("evaluate_exp_004c_full.py")),
    run_name="__main__",
)
