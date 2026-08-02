#!/usr/bin/env python3
"""Download and verify the pinned Experiment 006 BERT checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-006-bert-large-classifier.json"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "encoder" / "model-cache"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "model_manifest.json"

if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    raise SystemExit(
        "Encoder model download failed: run with .venv-encoder/bin/python."
    )

experiment_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
model_config = experiment_config["model"]
model_id = model_config["id"]
revision = model_config["revision"]

print("Open Model Training Lab — encoder model download")
print(f"model_id: {model_id}")
print(f"requested_revision: {revision}")

snapshot_path = Path(
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=str(CACHE_DIR),
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ],
    )
)

required_files = {
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
}
missing_files = sorted(
    filename for filename in required_files if not (snapshot_path / filename).is_file()
)
if missing_files:
    raise SystemExit(f"Encoder model download failed: missing files {missing_files}")

available_weight_files = [
    snapshot_path / filename
    for filename in ("model.safetensors", "pytorch_model.bin")
    if (snapshot_path / filename).is_file()
]
if len(available_weight_files) != 1:
    raise SystemExit(
        "Encoder model download failed: expected exactly one supported weight file, "
        f"found {[path.name for path in available_weight_files]}."
    )

architecture = json.loads(
    (snapshot_path / "config.json").read_text(encoding="utf-8")
)
expected_architecture = {
    "model_type": "bert",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
}
for field, expected in expected_architecture.items():
    observed = architecture.get(field)
    if observed != expected:
        raise SystemExit(
            f"Encoder model download failed: expected {field}={expected!r}, "
            f"received {observed!r}."
        )

weight_path = available_weight_files[0]
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
print(f"model_manifest: {MANIFEST_PATH}")
print("encoder_model_download_ok: True")
