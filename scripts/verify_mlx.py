#!/usr/bin/env python3
"""Prove that MLX can execute a calculation through Metal."""

from __future__ import annotations

from importlib.metadata import version

import mlx.core as mx


print("Open Model Training Lab — MLX compute check")
print(f"mlx_version: {version('mlx')}")

metal_available = mx.metal.is_available()
print(f"metal_available: {metal_available}")
if not metal_available:
    raise SystemExit("MLX compute check failed: no Metal device is available.")

print(f"default_device: {mx.default_device()}")
print(f"device_info: {mx.device_info()}")

left = mx.array([[1.0, 2.0], [3.0, 4.0]])
right = mx.array([[5.0, 6.0], [7.0, 8.0]])
product = left @ right
mx.eval(product)

actual = product.tolist()
expected = [[19.0, 22.0], [43.0, 50.0]]
calculation_ok = actual == expected

print(f"matrix_product: {actual}")
print(f"calculation_ok: {calculation_ok}")

if not calculation_ok:
    raise SystemExit("MLX compute check failed: unexpected matrix result.")
