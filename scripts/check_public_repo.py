#!/usr/bin/env python3
"""Fail if a prospective public commit contains local or oversized artifacts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
REQUIRED_FILES = {
    ".gitignore",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "docs/REPRODUCING.md",
    "docs/RESULTS.md",
    "docs/START_HERE.md",
    "docs/EXPERIMENT_JOURNEY.md",
    "docs/PRESENCE.md",
    "learning/app.js",
    "learning/index.html",
    "learning/styles.css",
    "scripts/README.md",
}
FORBIDDEN_PATH_PARTS = {
    ".DS_Store",
    ".venv",
    "__pycache__",
    "data/cache",
    "models/cache",
    "models/model_manifest.json",
}
FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".onnx",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
}
FORBIDDEN_NAME_PATTERNS = (
    re.compile(r"(?:^|/)evaluation/results/.+_predictions\.jsonl$"),
    re.compile(r"(?:^|/)training\.log$"),
)
SENSITIVE_TEXT_PATTERNS = {
    "absolute macOS user path": re.compile("/" + r"Users/[^/\s]+/"),
    "private key material": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "project source wrapper path": re.compile(
        "referenced-chatgpt-" + "conversation-this-is-untrusted"
    ),
    "private corporate identity": re.compile(
        r"(?:Suleman" + r"DMAGlobal|dma" + r"global\.com\.au)",
        re.IGNORECASE,
    ),
    "GitHub access token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "Hugging Face access token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
}


def public_candidates() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        Path(value.decode("utf-8"))
        for value in result.stdout.split(b"\0")
        if value
    ]


errors: list[str] = []
paths = public_candidates()
path_names = {path.as_posix() for path in paths}

for required in sorted(REQUIRED_FILES - path_names):
    errors.append(f"required public file is missing or ignored: {required}")

total_bytes = 0
for relative_path in paths:
    normalized = relative_path.as_posix()
    absolute_path = PROJECT_ROOT / relative_path
    if not absolute_path.is_file():
        continue
    total_bytes += absolute_path.stat().st_size
    if any(part in normalized for part in FORBIDDEN_PATH_PARTS):
        errors.append(f"local artifact would be committed: {normalized}")
    if relative_path.suffix in FORBIDDEN_SUFFIXES:
        errors.append(f"binary/generated artifact would be committed: {normalized}")
    if any(pattern.search(normalized) for pattern in FORBIDDEN_NAME_PATTERNS):
        errors.append(f"generated output would be committed: {normalized}")
    if absolute_path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
        errors.append(
            f"file exceeds {MAX_PUBLIC_FILE_BYTES} bytes: {normalized}"
        )
    try:
        text = absolute_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for label, pattern in SENSITIVE_TEXT_PATTERNS.items():
        if pattern.search(text):
            errors.append(f"{label} found in {normalized}")

print("Open Model Training Lab — public repository check")
print(f"candidate_files: {len(paths)}")
print(f"candidate_bytes: {total_bytes}")
print(f"max_file_bytes: {MAX_PUBLIC_FILE_BYTES}")
if errors:
    for error in sorted(set(errors)):
        print(f"ERROR: {error}")
    raise SystemExit("public_repository_check_ok: False")
print("public_repository_check_ok: True")
