#!/usr/bin/env python3
"""Load the first LoRA adapter and run the fixed smoke prediction."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

from mlx_lm import load, stream_generate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "manifest.json"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "banking77" / "raw" / "test.csv"
PROMPT_CONFIG_PATH = PROJECT_ROOT / "configs" / "banking77_prompt.json"
PIPELINE_RESULT_PATH = (
    PROJECT_ROOT / "experiments" / "exp-001-pipeline" / "result.json"
)
BASE_SMOKE_PATH = PROJECT_ROOT / "experiments" / "smoke" / "model_smoke.json"
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "exp-001-pipeline"
ADAPTER_FILE = ADAPTER_PATH / "adapters.safetensors"
RESULT_PATH = (
    PROJECT_ROOT / "experiments" / "exp-001-pipeline" / "adapter_smoke.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
dataset_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
prompt_config = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
pipeline_result = json.loads(PIPELINE_RESULT_PATH.read_text(encoding="utf-8"))
base_smoke = json.loads(BASE_SMOKE_PATH.read_text(encoding="utf-8"))

if pipeline_result["status"] != "complete":
    raise SystemExit("Adapter smoke failed: pipeline result is not complete.")
if sha256(ADAPTER_FILE) != pipeline_result["adapter_sha256"]:
    raise SystemExit("Adapter smoke failed: adapter checksum mismatch.")

labels = dataset_manifest["labels"]
system_prompt = prompt_config["system_template"].format(
    labels=prompt_config["label_separator"].join(labels)
)

with TEST_DATA_PATH.open(encoding="utf-8", newline="") as handle:
    example = next(csv.DictReader(handle))

if (
    example["text"] != base_smoke["request"]
    or example["category"] != base_smoke["expected"]
):
    raise SystemExit("Adapter smoke failed: baseline smoke example changed.")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": example["text"]},
]

print("Open Model Training Lab — LoRA adapter smoke inference")
print(f"model_id: {model_manifest['model_id']}")
print(f"adapter: {ADAPTER_PATH}")
print(f"request: {example['text']}")
print(f"expected: {example['category']}")
print(f"untouched_prediction: {base_smoke['prediction']}")
print("loading_model_and_adapter: True")

started_at = time.perf_counter()
model, tokenizer = load(
    model_manifest["snapshot_path"],
    adapter_path=str(ADAPTER_PATH),
)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=prompt_config["enable_thinking"],
)

raw_output = ""
last_response = None
for response in stream_generate(
    model,
    tokenizer,
    prompt,
    max_tokens=prompt_config["max_output_tokens"],
):
    raw_output += response.text
    last_response = response

if last_response is None:
    raise SystemExit("Adapter smoke failed: the model generated no response.")

prediction = raw_output.strip()
result = {
    "name": "exp-001-pipeline-adapter-smoke",
    "model_id": model_manifest["model_id"],
    "model_revision": model_manifest["resolved_revision"],
    "adapter_sha256": pipeline_result["adapter_sha256"],
    "prompt_version": prompt_config["prompt_version"],
    "dataset_revision": dataset_manifest["source_revision"],
    "split": "test",
    "example_index": 0,
    "request": example["text"],
    "expected": example["category"],
    "untouched_prediction": base_smoke["prediction"],
    "raw_output": raw_output,
    "prediction": prediction,
    "valid_label": prediction in labels,
    "prediction_correct": prediction == example["category"],
    "prompt_tokens": last_response.prompt_tokens,
    "generation_tokens": last_response.generation_tokens,
    "peak_memory_gb": round(last_response.peak_memory, 3),
    "elapsed_seconds_including_load": round(
        time.perf_counter() - started_at,
        3,
    ),
    "finish_reason": last_response.finish_reason,
}
RESULT_PATH.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"raw_output: {raw_output!r}")
print(f"prediction: {prediction}")
print(f"valid_label: {result['valid_label']}")
print(f"prediction_correct: {result['prediction_correct']}")
print(f"peak_memory_gb: {last_response.peak_memory:.3f}")
print(f"result: {RESULT_PATH}")
print("adapter_smoke_ok: True")
