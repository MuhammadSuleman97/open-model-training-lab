#!/usr/bin/env python3
"""Print a privacy-safe readiness report for the MLX training lab."""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys


def command_output(*args: str) -> str:
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def selected_value(report: str, key: str) -> str:
    prefix = f"{key}:"
    for raw_line in report.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return "unavailable"


def metal_status() -> str:
    text_report = command_output(
        "/usr/sbin/system_profiler",
        "SPDisplaysDataType",
    )
    text_value = selected_value(text_report, "Metal")
    if text_value != "unavailable":
        return text_value

    json_report = command_output(
        "/usr/sbin/system_profiler",
        "-json",
        "SPDisplaysDataType",
    )
    try:
        displays = json.loads(json_report)["SPDisplaysDataType"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return "deferred until MLX compute check"

    if any(item.get("spdisplays_metal") == "spdisplays_supported" for item in displays):
        return "Supported"
    return "deferred until MLX compute check"


hardware = command_output("/usr/sbin/system_profiler", "SPHardwareDataType")
metal = metal_status()

print("Open Model Training Lab — environment check")
print(f"architecture: {platform.machine()}")
print(f"model: {selected_value(hardware, 'Model Name')}")
print(f"model_identifier: {selected_value(hardware, 'Model Identifier')}")
print(f"chip: {selected_value(hardware, 'Chip')}")
print(f"memory: {selected_value(hardware, 'Memory')}")
print(f"metal: {metal}")
print(f"python_version: {platform.python_version()}")
print(f"python_executable: {sys.executable}")
print(f"xcode_developer_dir: {command_output('xcode-select', '-p')}")
print(f"mlx_installed: {importlib.util.find_spec('mlx') is not None}")
print(f"mlx_lm_installed: {importlib.util.find_spec('mlx_lm') is not None}")
print(f"datasets_installed: {importlib.util.find_spec('datasets') is not None}")

apple_silicon_ready = (
    platform.machine() == "arm64"
    and selected_value(hardware, "Chip").startswith("Apple ")
    and metal != "Unsupported"
)
print(f"apple_silicon_ready: {apple_silicon_ready}")

if not apple_silicon_ready:
    raise SystemExit("Environment check failed: Apple Silicon not detected.")
