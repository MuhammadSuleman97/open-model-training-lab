# Reproducing the experiments

This lab was designed for Apple Silicon and records exact package, model and
dataset revisions. It contains a Qwen/MLX track and a BERT/DeBERTa/PyTorch MPS
track. Reproduction still means repeating an experiment, not assuming
byte-identical floating-point output across every machine.

## 1. Inspect and install

Use Python 3.11 on a Mac with Apple Silicon. The recorded full-data runs peaked
near 5.2 GB of MLX memory at batch size 1; allow additional memory for macOS and
other applications.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r environment/requirements.lock
.venv/bin/python scripts/verify_environment.py
.venv/bin/python scripts/verify_mlx.py
```

`verify_mlx.py` performs a real GPU calculation. An import alone does not prove
that MLX can use Metal.

## 2. Reconstruct the data

```bash
.venv/bin/python scripts/inspect_banking77.py
.venv/bin/python scripts/prepare_sft_data.py
.venv/bin/python scripts/prepare_evaluation.py
```

The downloader uses the pinned original PolyAI repository and verifies SHA-256
checksums. It intentionally rejects the incomplete mirror that omitted 14
records.

The resulting partitions are:

```text
Original train: 10,003
  ├── SFT train:      9,233  (optimizer updates)
  └── Validation:       770  (10 per intent)

Original test:       3,080  (final held-out evaluation)
```

## 3. Download and inspect the model

```bash
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/smoke_inference.py
.venv/bin/python scripts/inspect_tokenization.py
```

`download_model.py` resolves the configured revision, downloads approximately
3.2 GiB of BF16 weights, and writes the machine-specific ignored file
`models/model_manifest.json`.

## 4. Establish the baseline

```bash
.venv/bin/python scripts/evaluate_pilot.py
.venv/bin/python scripts/evaluate_full_baseline.py
```

Do this before fine-tuning. Training loss cannot substitute for an untouched
held-out control.

## 5. Validate the training pipeline

```bash
.venv/bin/python scripts/run_lora_pipeline.py
.venv/bin/python scripts/smoke_adapter_inference.py
```

The small run proves model loading, prompt masking, backpropagation, adapter
saving and adapter reloading. It is not an accuracy experiment.

## 6. Run the stable full-data configuration

The historical `run_exp_005b.py` intentionally verifies its failed parent run
before performing the stable retry. A fresh clone can train the recorded stable
configuration directly through MLX-LM:

```bash
.venv/bin/mlx_lm.lora \
  --config configs/exp-005b-attention-qkvo-lr2p5e-7.yaml
```

On the recorded M2 Max this took about 96 minutes. Training output belongs in
the ignored adapter and experiment directories. Do not commit `.safetensors`
files to normal Git history.

## 7. Evaluate carefully

The historical evaluators bind results to checksums so stale predictions cannot
be mixed with a new adapter. If your adapter checksum differs, use a new
experiment identifier and update the copied evaluator's expected result record
rather than overwriting the published experiment.

Evaluate a small balanced pilot first, then the full test. Compare:

- exact-label accuracy;
- macro F1;
- invalid-label rate;
- per-intent accuracy; and
- paired prediction transitions against the baseline.

Do not tune repeatedly against the 3,080-row test set. Use validation data for
model selection and preserve the test as the final exam.

## 8. Reproduce the encoder track

Use the separate environment so PyTorch/Transformers changes do not disturb the
recorded MLX stack:

```bash
python3.11 -m venv .venv-encoder
.venv-encoder/bin/python -m pip install --upgrade pip
.venv-encoder/bin/python -m pip install -r environment/encoder-requirements.lock
.venv-encoder/bin/python scripts/verify_encoder_environment.py
.venv-encoder/bin/python scripts/prepare_encoder_data.py
```

The encoder data script reconstructs the same 9,233 training rows and 770
validation rows while keeping the original 3,080-row test sealed.

Before full training, download the pinned model and run the matching two-update
probe. For the DeBERTa track:

```bash
.venv-encoder/bin/python scripts/download_exp_009_model.py
.venv-encoder/bin/python scripts/smoke_exp_009_model.py
.venv-encoder/bin/python scripts/run_exp_009d_probe.py
```

The float32 cast in Exp009d is intentional. The pinned checkpoint's float16
parameters became non-finite after the first AdamW update on the recorded MPS
setup even though forward losses were finite.

The complete Exp009d training run takes roughly 75 minutes on the recorded M2
Max. Exp011 is a continuation of Exp010 rather than a standalone fresh-clone
command; follow [EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md) and inspect each
parent checksum before reproducing the full chain.

Do not run the reporting-only test evaluator while selecting a model. Validation
must choose the checkpoint first.

## 9. Serve locally

After the recorded adapter exists at its configured path:

```bash
.venv/bin/python scripts/classify_banking_request.py \
  --text "The cash machine charged me an extra fee."

.venv/bin/python scripts/serve_banking_classifier.py
```

The server binds only to a loopback address and is intentionally
single-threaded because MLX Metal generation failed when called from worker
threads during this project.

## Artifact safety

Before a commit:

```bash
python3 scripts/check_public_repo.py
```

The check rejects model/adaptor binaries, caches, row-level predictions,
machine-specific absolute paths, likely secret material and files larger than
the public source limit.
