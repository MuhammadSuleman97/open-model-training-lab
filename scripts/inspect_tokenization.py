#!/usr/bin/env python3
"""Measure SFT sequence lengths before choosing a training limit."""

from __future__ import annotations

import json
import math
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"
PROMPT_CONFIG_PATH = PROJECT_ROOT / "configs" / "banking77_prompt.json"
SFT_MANIFEST_PATH = PROJECT_ROOT / "data" / "banking77" / "sft_manifest.json"
SFT_DIR = PROJECT_ROOT / "data" / "banking77" / "sft"
REPORT_PATH = PROJECT_ROOT / "data" / "banking77" / "tokenization_report.json"

CANDIDATE_LIMITS = [512, 576, 640, 768]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(values: list[int]) -> dict[str, int]:
    return {
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
prompt_config = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
sft_manifest = json.loads(SFT_MANIFEST_PATH.read_text(encoding="utf-8"))
tokenizer = AutoTokenizer.from_pretrained(
    model_manifest["snapshot_path"],
    local_files_only=True,
)

split_reports: dict[str, dict[str, object]] = {}
all_full_lengths: list[int] = []
all_prompt_lengths: list[int] = []
all_completion_lengths: list[int] = []

for split_name in ("train", "valid"):
    records = read_jsonl(SFT_DIR / f"{split_name}.jsonl")
    full_lengths: list[int] = []
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []

    for record in records:
        messages = record["messages"]
        # Match MLX-LM 0.31.3 ChatDataset.process exactly. In particular,
        # return_dict=False produces a token-ID list, and MLX-LM leaves Qwen's
        # enable_thinking template option at its default during SFT.
        full_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        prompt_tokens = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

        if full_tokens[: len(prompt_tokens)] != prompt_tokens:
            raise SystemExit(
                "Tokenization inspection failed: assistant completion is not "
                "a suffix after the masked prompt."
            )

        completion_length = len(full_tokens) - len(prompt_tokens)
        if completion_length <= 0:
            raise SystemExit(
                "Tokenization inspection failed: empty assistant completion."
            )

        full_lengths.append(len(full_tokens))
        prompt_lengths.append(len(prompt_tokens))
        completion_lengths.append(completion_length)

    split_reports[split_name] = {
        "rows": len(records),
        "full_sequence_tokens": summarize(full_lengths),
        "masked_prompt_tokens": summarize(prompt_lengths),
        "assistant_completion_tokens": summarize(completion_lengths),
    }
    all_full_lengths.extend(full_lengths)
    all_prompt_lengths.extend(prompt_lengths)
    all_completion_lengths.extend(completion_lengths)

truncation_counts = {
    str(limit): sum(length > limit for length in all_full_lengths)
    for limit in CANDIDATE_LIMITS
}
recommended_max_seq_length = next(
    (
        limit
        for limit in CANDIDATE_LIMITS
        if truncation_counts[str(limit)] == 0
    ),
    math.ceil(max(all_full_lengths) / 64) * 64,
)

report = {
    "name": "banking77-sft-tokenization-v1",
    "model_id": model_manifest["model_id"],
    "model_revision": model_manifest["resolved_revision"],
    "prompt_version": prompt_config["prompt_version"],
    "training_template_behavior": (
        "Matches MLX-LM 0.31.3 ChatDataset: default Qwen chat-template "
        "options with loss masked through the assistant generation boundary."
    ),
    "sft_train_sha256": sft_manifest["files"]["train.jsonl"]["sha256"],
    "sft_valid_sha256": sft_manifest["files"]["valid.jsonl"]["sha256"],
    "total_rows": len(all_full_lengths),
    "splits": split_reports,
    "combined": {
        "full_sequence_tokens": summarize(all_full_lengths),
        "masked_prompt_tokens": summarize(all_prompt_lengths),
        "assistant_completion_tokens": summarize(all_completion_lengths),
    },
    "candidate_max_seq_length_truncation_counts": truncation_counts,
    "recommended_max_seq_length": recommended_max_seq_length,
    "mask_prompt_safe": True,
}
REPORT_PATH.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("Open Model Training Lab — SFT tokenization inspection")
print(f"total_rows: {len(all_full_lengths)}")
print(f"full_sequence_min: {min(all_full_lengths)}")
print(f"full_sequence_p50: {percentile(all_full_lengths, 0.50)}")
print(f"full_sequence_p95: {percentile(all_full_lengths, 0.95)}")
print(f"full_sequence_p99: {percentile(all_full_lengths, 0.99)}")
print(f"full_sequence_max: {max(all_full_lengths)}")
for limit in CANDIDATE_LIMITS:
    print(
        f"sequences_over_{limit}: "
        f"{truncation_counts[str(limit)]}"
    )
print(
    "assistant_completion_tokens: "
    f"{summarize(all_completion_lengths)}"
)
print(f"recommended_max_seq_length: {recommended_max_seq_length}")
print("mask_prompt_safe: True")
print(f"report: {REPORT_PATH}")
print("tokenization_inspection_ok: True")
