#!/usr/bin/env python3
"""Resolve, download, and record the selected MLX model revision."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.json"
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "cache"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"

model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
model_id = model_config["model_id"]
requested_revision = model_config["requested_revision"]

print("Open Model Training Lab — model download")
print(f"model_id: {model_id}")
print(f"requested_revision: {requested_revision}")

model_info = HfApi().model_info(
    model_id,
    revision=requested_revision,
    files_metadata=True,
)
resolved_revision = model_info.sha
if not resolved_revision:
    raise SystemExit("Model download failed: Hugging Face returned no revision.")

print(f"resolved_revision: {resolved_revision}")
snapshot_path = Path(
    snapshot_download(
        repo_id=model_id,
        revision=resolved_revision,
        cache_dir=str(MODEL_CACHE_DIR),
    )
)

required_files = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}
missing_files = sorted(
    filename for filename in required_files if not (snapshot_path / filename).is_file()
)
if missing_files:
    raise SystemExit(f"Model download failed: missing files {missing_files}")

architecture = json.loads(
    (snapshot_path / "config.json").read_text(encoding="utf-8")
)
if architecture.get("model_type") != "qwen3":
    raise SystemExit(
        "Model download failed: expected model_type 'qwen3', received "
        f"{architecture.get('model_type')!r}"
    )
if "quantization" in architecture or "quantization_config" in architecture:
    raise SystemExit("Model download failed: expected an unquantized BF16 model.")

weight_files = sorted(snapshot_path.glob("*.safetensors"))
weight_bytes = sum(path.stat().st_size for path in weight_files)
weight_gib = weight_bytes / (1024**3)

manifest = {
    "model_id": model_id,
    "resolved_revision": resolved_revision,
    "license": model_config["license"],
    "precision": model_config["precision"],
    "quantized": model_config["quantized"],
    "model_type": architecture["model_type"],
    "architectures": architecture.get("architectures", []),
    "weight_files": [path.name for path in weight_files],
    "weight_bytes": weight_bytes,
    "weight_gib": round(weight_gib, 3),
    "snapshot_path": str(snapshot_path),
}
MODEL_MANIFEST_PATH.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"weight_files: {[path.name for path in weight_files]}")
print(f"weight_gib: {weight_gib:.3f}")
print(f"model_manifest: {MODEL_MANIFEST_PATH}")
print("model_download_ok: True")
