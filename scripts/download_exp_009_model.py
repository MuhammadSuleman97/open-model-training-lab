#!/usr/bin/env python3
"""Download and verify the pinned Experiment 009 DeBERTa checkpoint."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-009-deberta-v3-large-classifier.json"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009" / "model-cache"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "exp-009" / "model_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit(
        "Experiment 009 model download failed: use .venv-encoder/bin/python."
    )
if MANIFEST_PATH.exists():
    raise SystemExit("Experiment 009 model download stopped: manifest already exists.")

experiment_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model_config = experiment_config["model"]
model_id = model_config["id"]
revision = model_config["revision"]

print("Open Model Training Lab — Experiment 009 model download")
print(f"model_id: {model_id}")
print(f"requested_revision: {revision}")
snapshot_path = Path(
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=str(CACHE_DIR),
        allow_patterns=[
            "config.json",
            "pytorch_model.bin",
            "spm.model",
            "tokenizer_config.json",
        ],
    )
)

required_files = {
    "config.json",
    "pytorch_model.bin",
    "spm.model",
    "tokenizer_config.json",
}
missing_files = sorted(
    filename for filename in required_files if not (snapshot_path / filename).is_file()
)
if missing_files:
    raise SystemExit(
        f"Experiment 009 model download failed: missing files {missing_files}."
    )
unexpected_weights = sorted(path.name for path in snapshot_path.glob("*model*.bin"))
if unexpected_weights != ["pytorch_model.bin"]:
    raise SystemExit(
        "Experiment 009 model download failed: unexpected weight files "
        f"{unexpected_weights}."
    )

architecture = json.loads((snapshot_path / "config.json").read_text(encoding="utf-8"))
expected_architecture = {
    "model_type": "deberta-v2",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
}
for field, expected in expected_architecture.items():
    observed = architecture.get(field)
    if observed != expected:
        raise SystemExit(
            f"Experiment 009 model download failed: expected {field}={expected!r}, "
            f"received {observed!r}."
        )

weight_path = snapshot_path / "pytorch_model.bin"
weight_bytes = weight_path.stat().st_size
manifest = {
    "experiment": experiment_config["experiment"],
    "model_id": model_id,
    "resolved_revision": revision,
    "license": model_config["license"],
    "model_type": architecture["model_type"],
    "hidden_size": architecture["hidden_size"],
    "num_hidden_layers": architecture["num_hidden_layers"],
    "num_attention_heads": architecture["num_attention_heads"],
    "weight_file": weight_path.name,
    "weight_bytes": weight_bytes,
    "weight_gib": round(weight_bytes / (1024**3), 3),
    "weight_sha256": sha256(weight_path),
    "snapshot_path": str(snapshot_path),
}
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"resolved_revision: {revision}")
print(f"model_type: {architecture['model_type']}")
print(f"hidden_size: {architecture['hidden_size']}")
print(f"layers: {architecture['num_hidden_layers']}")
print(f"attention_heads: {architecture['num_attention_heads']}")
print(f"weight_file: {weight_path.name}")
print(f"weight_gib: {weight_bytes / (1024**3):.3f}")
print(f"weight_sha256: {manifest['weight_sha256']}")
print(f"model_manifest: {MANIFEST_PATH}")
print("exp_009_model_download_ok: True")
