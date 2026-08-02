#!/usr/bin/env python3
"""Verify the isolated PyTorch encoder environment and MPS computation."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import version
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
EXPECTED_VERSIONS = {
    "accelerate": "1.14.0",
    "scikit-learn": "1.9.0",
    "torch": "2.13.0",
    "transformers": "5.14.1",
}

print("Open Model Training Lab — encoder environment check")
print(f"architecture: {platform.machine()}")
print(f"python_version: {platform.python_version()}")
print(f"python_executable: {sys.executable}")

if platform.machine() != "arm64":
    raise SystemExit("Encoder check failed: expected Apple Silicon arm64.")
if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit(
        "Encoder check failed: run with .venv-encoder/bin/python."
    )

for package, expected in EXPECTED_VERSIONS.items():
    observed = version(package)
    print(f"{package}_version: {observed}")
    if observed != expected:
        raise SystemExit(
            f"Encoder check failed: {package} must be {expected}, "
            f"received {observed}."
        )

print(f"mps_built: {torch.backends.mps.is_built()}")
print(f"mps_available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
    raise SystemExit("Encoder check failed: PyTorch MPS is unavailable.")

device = torch.device("mps")
left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
product = left @ right
torch.mps.synchronize()
actual = product.cpu().tolist()
expected = [[19.0, 22.0], [43.0, 50.0]]
calculation_ok = actual == expected

print(f"device: {device}")
print(f"matrix_product: {actual}")
print(f"calculation_ok: {calculation_ok}")
if not calculation_ok:
    raise SystemExit("Encoder check failed: unexpected MPS matrix result.")

print("encoder_environment_ok: True")
