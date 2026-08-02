# Script map

The scripts preserve the chronological research trail. Use this map instead of
reading the directory alphabetically.

## Safety and environment

- `check_public_repo.py` — inspect the exact prospective public commit.
- `verify_environment.py` / `verify_mlx.py` — Qwen/MLX environment and GPU.
- `verify_encoder_environment.py` — BERT/DeBERTa PyTorch MPS environment.
- `serve_learning_lab.py` — run or preflight the interactive guide.

## Data reconstruction

- `inspect_banking77.py` — download and verify pinned original BANKING77.
- `prepare_sft_data.py` — create Qwen chat-format train/validation files.
- `prepare_encoder_data.py` — create raw-text encoder splits.
- `inspect_tokenization.py` — inspect actual training token boundaries.
- `prepare_exp_*_data.py` — deterministic experiment-specific train-only data.

Generated rows are ignored by Git. Public manifests retain counts and
checksums.

## Qwen + MLX/LoRA track

- `download_model.py` / `smoke_inference.py` — obtain and test untouched Qwen.
- `evaluate_full_baseline.py` — establish the full untouched control.
- `run_lora_pipeline.py` — cheap end-to-end adapter proof.
- `run_exp_002.py` through `run_exp_005b.py` — chronological LoRA experiments.
- `evaluate_exp_*` — checksum-bound pilot and full evaluations.
- `classify_banking_request.py` — one-request local inference.
- `serve_banking_classifier.py` — loopback-only JSON demonstration API.

## BERT/DeBERTa encoder track

- `download_encoder_model.py` / `smoke_encoder_model.py` — BERT base setup.
- `run_exp_006.py` — BERT-Large full fine-tuning.
- `run_exp_007.py` / `run_exp_008.py` — rejected BERT refinements.
- `download_exp_009_model.py` / `smoke_exp_009_model.py` — DeBERTa setup.
- `diagnose_exp_009_nan.py` / `diagnose_exp_009c_nan.py` — optimizer failure
  localization.
- `run_exp_009d.py` — stable float32 full fine-tuning.
- `run_exp_010.py` / `run_exp_011.py` — retained DeBERTa refinements.
- `run_exp_012_label_audit.py` — train-only five-fold label-quality audit.
- `run_exp_013.py` / `run_exp_014.py` — rejected hard-negative refinements.
- `run_exp_015.py` — rejected noise-pruned retraining.

## Naming convention

- `prepare_...` creates deterministic local inputs.
- `smoke_...` proves a forward pass.
- `..._probe` performs a tiny backward pass and saves no model.
- `run_...` performs training.
- `finalize_...` packages a validation-selected checkpoint.
- `evaluate_...` measures an untouched split.
- `analyze_...` explains transitions after metrics are fixed.
- `diagnose_...` localizes a recorded failure.

## Before running anything expensive

1. Read the experiment config.
2. Verify `test_rows_loaded: 0` for development work.
3. Run the matching probe.
4. Confirm the output directory is empty or intentionally resumable.
5. Record the parent checksum and promotion rule.

Historical scripts deliberately refuse unsafe overwrites. Do not delete those
guards merely to make a command rerun.
