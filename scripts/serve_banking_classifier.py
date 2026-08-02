#!/usr/bin/env python3
"""Serve constrained Experiment 005b on a localhost JSON API."""

from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


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
MAX_REQUEST_BYTES = 16_384


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(
    description="Serve the trained BANKING77 adapter on localhost."
)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument(
    "--check",
    action="store_true",
    help="Validate artifacts without loading MLX or starting the server.",
)
args = parser.parse_args()

if args.host not in {"127.0.0.1", "localhost", "::1"}:
    raise SystemExit(
        "Local API refused: --host must be a loopback address."
    )
if not 1 <= args.port <= 65535:
    raise SystemExit("Local API refused: --port must be between 1 and 65535.")

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
    raise SystemExit("Local API failed: Experiment 005b is incomplete.")
if experiment_result.get("nonfinite_adapter_values") != 0:
    raise SystemExit("Local API failed: adapter is non-finite.")
if sha256(ADAPTER_FILE) != experiment_result["adapter_sha256"]:
    raise SystemExit("Local API failed: adapter checksum mismatch.")
if not Path(model_manifest["snapshot_path"]).is_dir():
    raise SystemExit("Local API failed: model snapshot is missing.")
if constrained_summary.get("adapter_sha256") != experiment_result["adapter_sha256"]:
    raise SystemExit("Local API failed: constrained evaluation is stale.")
if constrained_summary.get("constraint_mode") != "canonical_labels":
    raise SystemExit("Local API failed: canonical constraint was not evaluated.")
if constrained_summary.get("invalid_labels") != 0:
    raise SystemExit("Local API failed: constrained evaluation produced invalid labels.")

if args.check:
    print("Open Model Training Lab — local API preflight")
    print(f"host: {args.host}")
    print(f"port: {args.port}")
    print(f"model_id: {model_manifest['model_id']}")
    print(f"adapter_sha256: {experiment_result['adapter_sha256']}")
    print(f"labels: {len(dataset_manifest['labels'])}")
    print("constraint_mode: canonical_labels")
    print(f"verified_full_test_accuracy: {constrained_summary['accuracy']:.6f}")
    print("local_api_preflight_ok: True")
    raise SystemExit(0)

import mlx.core as mx
from mlx_lm import batch_generate, load

labels = dataset_manifest["labels"]
system_prompt = prompt_config["system_template"].format(
    labels=prompt_config["label_separator"].join(labels)
)
print("Open Model Training Lab — local banking classifier API", flush=True)
print(f"model_id: {model_manifest['model_id']}", flush=True)
print(f"adapter_sha256: {experiment_result['adapter_sha256']}", flush=True)
print("constraint_mode: canonical_labels", flush=True)
print("loading_model_and_adapter: True", flush=True)
model, tokenizer = load(
    model_manifest["snapshot_path"],
    adapter_path=str(ADAPTER_PATH),
)
label_token_sequences = [
    tuple(tokenizer.encode(label, add_special_tokens=False))
    for label in labels
]
if any(not sequence for sequence in label_token_sequences):
    raise SystemExit("Local API failed: an allowed label is empty.")
if len(set(label_token_sequences)) != len(labels):
    raise SystemExit("Local API failed: labels have duplicate token sequences.")
eos_token_ids = tuple(int(token) for token in tokenizer.eos_token_ids)
if not eos_token_ids:
    raise SystemExit("Local API failed: tokenizer has no EOS token.")


def make_canonical_label_processor(prompt_length: int):
    """Allow only token paths that finish as a canonical BANKING77 label."""

    def process(tokens, logits):
        generated = tuple(
            int(token) for token in tokens.tolist()[prompt_length:]
        )
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

    return process


def classify(request: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=prompt_config["enable_thinking"],
    )
    response = batch_generate(
        model,
        tokenizer,
        [prompt],
        max_tokens=prompt_config["max_output_tokens"],
        prefill_batch_size=1,
        completion_batch_size=1,
        logits_processors=[make_canonical_label_processor(len(prompt))],
    )
    raw_output = response.texts[0]
    prediction = raw_output.strip()
    return {
        "request": request,
        "prediction": prediction,
        "valid_banking77_label": prediction in labels,
        "model_id": model_manifest["model_id"],
        "adapter_sha256": experiment_result["adapter_sha256"],
        "constraint_mode": "canonical_labels",
    }


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "OpenModelTrainingLab/1.0"

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(
                200,
                {
                    "status": "ready",
                    "model_id": model_manifest["model_id"],
                    "adapter_sha256": experiment_result["adapter_sha256"],
                    "constraint_mode": "canonical_labels",
                },
            )
            return
        self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/classify":
            self.send_json(404, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "invalid_content_length"})
            return
        if not 0 < content_length <= MAX_REQUEST_BYTES:
            self.send_json(400, {"error": "invalid_request_size"})
            return
        try:
            payload = json.loads(
                self.rfile.read(content_length).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"error": "invalid_json"})
            return
        request = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(request, str) or not request.strip():
            self.send_json(
                400,
                {"error": "text_must_be_a_nonempty_string"},
            )
            return
        try:
            self.send_json(200, classify(request.strip()))
        except Exception as error:
            print(
                "classification_error: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            self.send_json(
                500,
                {
                    "error": "classification_failed",
                    "detail": type(error).__name__,
                },
            )


server = HTTPServer((args.host, args.port), RequestHandler)
print(f"local_api_ready: http://{args.host}:{args.port}", flush=True)
print("endpoints: GET /health, POST /classify", flush=True)
print("stop_with: Control-C", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nlocal_api_stopped: True", flush=True)
finally:
    server.server_close()
