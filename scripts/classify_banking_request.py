#!/usr/bin/env python3
"""Classify one banking request with constrained Experiment 005b."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "manifest.json"
PROMPT_CONFIG_PATH = PROJECT_ROOT / "configs" / "banking77_prompt.json"
EXPERIMENT_RESULT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "exp-005b-attention-qkvo-lr2p5e-7"
    / "result.json"
)
ADAPTER_PATH = (
    PROJECT_ROOT / "adapters" / "exp-005b-attention-qkvo-lr2p5e-7"
)
ADAPTER_FILE = ADAPTER_PATH / "adapters.safetensors"
CONSTRAINED_SUMMARY_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "exp_005b_constrained_full_summary.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(
    description="Classify one request with the trained BANKING77 adapter."
)
parser.add_argument(
    "--text",
    help="Banking request. If omitted, the script asks interactively.",
)
parser.add_argument(
    "--check",
    action="store_true",
    help="Validate artifacts without loading MLX.",
)
args = parser.parse_args()

model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
dataset_manifest = json.loads(
    DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
)
prompt_config = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
experiment_result = json.loads(
    EXPERIMENT_RESULT_PATH.read_text(encoding="utf-8")
)
constrained_summary = json.loads(
    CONSTRAINED_SUMMARY_PATH.read_text(encoding="utf-8")
)
if experiment_result.get("status") != "complete":
    raise SystemExit("Classifier failed: Experiment 005b is incomplete.")
if sha256(ADAPTER_FILE) != experiment_result["adapter_sha256"]:
    raise SystemExit("Classifier failed: adapter checksum mismatch.")
if not Path(model_manifest["snapshot_path"]).is_dir():
    raise SystemExit("Classifier failed: model snapshot is missing.")
if constrained_summary.get("adapter_sha256") != experiment_result["adapter_sha256"]:
    raise SystemExit("Classifier failed: constrained evaluation is stale.")
if constrained_summary.get("constraint_mode") != "canonical_labels":
    raise SystemExit("Classifier failed: canonical constraint was not evaluated.")
if constrained_summary.get("invalid_labels") != 0:
    raise SystemExit("Classifier failed: constrained evaluation produced invalid labels.")

if args.check:
    print("Open Model Training Lab — trained classifier preflight")
    print(f"model_id: {model_manifest['model_id']}")
    print(f"adapter_sha256: {experiment_result['adapter_sha256']}")
    print(f"labels: {len(dataset_manifest['labels'])}")
    print("constraint_mode: canonical_labels")
    print(
        "verified_full_test_accuracy: "
        f"{constrained_summary['accuracy']:.6f}"
    )
    print("verified_full_test_invalid_labels: 0")
    print("trained_classifier_preflight_ok: True")
    raise SystemExit(0)

request = args.text if args.text is not None else input("Banking request: ")
request = request.strip()
if not request:
    raise SystemExit("Classifier stopped: request cannot be empty.")

import mlx.core as mx
from mlx_lm import batch_generate, load

labels = dataset_manifest["labels"]
system_prompt = prompt_config["system_template"].format(
    labels=prompt_config["label_separator"].join(labels)
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": request},
]

print("loading_model_and_adapter: True")
model, tokenizer = load(
    model_manifest["snapshot_path"],
    adapter_path=str(ADAPTER_PATH),
)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=prompt_config["enable_thinking"],
)

label_token_sequences = [
    tuple(tokenizer.encode(label, add_special_tokens=False))
    for label in labels
]
if any(not sequence for sequence in label_token_sequences):
    raise SystemExit("Classifier failed: an allowed label is empty.")
if len(set(label_token_sequences)) != len(labels):
    raise SystemExit("Classifier failed: labels have duplicate token sequences.")
eos_token_ids = tuple(int(token) for token in tokenizer.eos_token_ids)
if not eos_token_ids:
    raise SystemExit("Classifier failed: tokenizer has no EOS token.")


def canonical_label_processor(tokens, logits):
    """Allow only token paths that finish as a canonical BANKING77 label."""

    generated = tuple(int(token) for token in tokens.tolist()[len(prompt) :])
    if generated and generated[-1] in eos_token_ids:
        return logits
    allowed: set[int] = set()
    for sequence in label_token_sequences:
        if sequence[: len(generated)] != generated:
            continue
        if len(generated) == len(sequence):
            allowed.update(eos_token_ids)
        else:
            allowed.add(sequence[len(generated)])
    if not allowed:
        raise RuntimeError(
            "Canonical-label constraint reached an invalid token prefix."
        )
    allowed_ids = mx.array(sorted(allowed), dtype=mx.int32)
    masked = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
    masked[:, allowed_ids] = logits[:, allowed_ids]
    return masked


response = batch_generate(
    model,
    tokenizer,
    [prompt],
    max_tokens=prompt_config["max_output_tokens"],
    prefill_batch_size=1,
    completion_batch_size=1,
    logits_processors=[canonical_label_processor],
)
raw_output = response.texts[0]
prediction = raw_output.strip()

print(f"request: {request}")
print(f"prediction: {prediction}")
print(f"valid_banking77_label: {prediction in labels}")
print("constraint_mode: canonical_labels")
print(f"adapter_sha256: {experiment_result['adapter_sha256']}")
