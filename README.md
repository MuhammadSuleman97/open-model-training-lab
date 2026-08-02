# Open Model Training Lab

**One banking-intent task, three open models, fifteen controlled experiments,
and every important failure preserved.**

This is a beginner-friendly, reproducible account of learning model
fine-tuning on an Apple M2 Max with 32 GB unified memory. It starts with Qwen +
LoRA, discovers why a generative LLM is a poor fit for closed-set
classification, moves to BERT and DeBERTa, and finishes with a **94.12%**
reporting-only BANKING77 test result.

The project does not hide the failed runs. NaN weights, rejected refinements,
over-pruned data and packaging bugs are part of the learning material.

> **Result, not marketing:** 94.12% is a strong local learning result, not a
> production banking system and not the 95% target originally set for the lab.

## The result in 30 seconds

| Era | System | Evaluation split | Accuracy | What it taught us |
|---|---|---|---:|---|
| A | Untouched Qwen3 1.7B | Test | 47.86% | Establish a baseline first |
| A | Qwen + LoRA + constrained decoding | Test | 51.10% | SFT, adapters and output contracts |
| B | BERT-Large classifier | Validation | 91.56% | Architecture fit matters enormously |
| C | DeBERTa-v3-large | Validation | 92.60% | float32 fixed MPS optimizer instability |
| C | DeBERTa upper-layer refinement | Validation | **92.99%** | Current validation champion |
| C | Same selected refinement | Reporting-only test | **94.12%** | 2,899 of 3,080 held-out requests correct |
| C | Noise-pruned retraining | Validation | 90.78% | Model disagreement is not proof of bad data |

Validation selected the models. Test results were opened only for reporting and
were excluded from every later experiment decision.

## Start here—no model download required

The interactive Flight Recorder teaches the entire project in plain English:

```bash
python3 scripts/serve_learning_lab.py
```

Open <http://127.0.0.1:8090/>. It includes:

- a Qwen → BERT → DeBERTa architecture map;
- an interactive checkpoint comparison;
- the complete experiment timeline and failure diagnoses;
- train/validation/test and data-leakage explanations;
- a searchable beginner glossary;
- interview flashcards and a knowledge quiz; and
- the Exp015 data-quality result, including why it was rejected.

Your browser stores reading and quiz progress locally. The guide sends no data
anywhere.

## What was actually trained?

```text
Banking request
      │
      ├── Qwen era: generate the exact label text token by token
      │               └── LoRA adapters; original Qwen weights frozen
      │
      └── Encoder era: produce 77 class scores directly
                      ├── BERT-Large full fine-tuning
                      └── DeBERTa full + upper-layer refinement
```

All models already understood language from pretraining. This lab taught them
the BANKING77 taxonomy and the exact decision behaviour needed for 77 banking
support intents.

## The scientific story

1. **Measure before changing weights.** Untouched Qwen scored 47.86%.
2. **Learn the complete LLM workflow.** LoRA, stable retries, evaluation,
   constrained decoding, CLI and local API were implemented with MLX.
3. **Recognize an architecture mismatch.** Generating labels was unnecessary
   for a fixed 77-class problem.
4. **Move to a purpose-built classifier.** BERT-Large reached 91.56%
   validation accuracy.
5. **Reject attractive ideas when evidence says no.** Class weighting, label
   smoothing and checkpoint ensembling all failed to improve BERT.
6. **Diagnose instead of guessing.** DeBERTa's float16 parameters became NaN
   after the optimizer update on MPS; casting the model to float32 fixed it.
7. **Protect the final exam.** DeBERTa experiments were selected on the fixed
   validation set, never by test performance.
8. **Treat data cleaning as a hypothesis.** Removing 1,078 suspicious rows
   reduced validation accuracy, so the child was rejected.

See [the complete experiment journey](docs/EXPERIMENT_JOURNEY.md) and
[recorded metrics](docs/RESULTS.md).

## Repository map

```text
.
├── README.md                 # Result and fastest reader path
├── learning/                 # Self-contained interactive course
├── docs/
│   ├── START_HERE.md         # Guided paths for learners and reviewers
│   ├── EXPERIMENT_JOURNEY.md # Exp001–015: question, result, decision
│   ├── RESULTS.md            # Metrics with validation/test labels
│   ├── REPRODUCING.md        # Rebuild both MLX and encoder tracks
│   └── PRESENCE.md           # GitHub/X/Hugging Face launch workflow
├── scripts/
│   └── README.md             # Map of preparation/training/evaluation scripts
├── configs/                  # Immutable experiment configurations
├── environment/              # Direct and fully locked dependencies
├── evaluation/               # Compact public summaries only
├── experiments/              # Small Qwen run records, including failures
├── data/                     # Source manifests; dataset rows stay local
└── TRAINING_STATE.md         # Detailed chronological research notebook
```

The 99 scripts are retained for evidence and reproducibility. New readers
should not read them alphabetically—use [scripts/README.md](scripts/README.md).

## Reproduce it on Apple Silicon

Two isolated environments prevent the MLX and PyTorch stacks from interfering:

```bash
# Generative Qwen + MLX/LoRA track
python3.11 -m venv .venv
.venv/bin/python -m pip install -r environment/requirements.lock
.venv/bin/python scripts/verify_mlx.py

# Encoder BERT/DeBERTa + PyTorch MPS track
python3.11 -m venv .venv-encoder
.venv-encoder/bin/python -m pip install -r environment/encoder-requirements.lock
.venv-encoder/bin/python scripts/verify_encoder_environment.py
```

Do not begin a long training run from this snippet. Follow
[REPRODUCING.md](docs/REPRODUCING.md), which reconstructs the pinned dataset,
establishes the baseline and runs a small backward-pass probe first.

## Public artifact boundary

Normal Git history contains source code, configuration, checksums, compact
metrics and explanations. It deliberately excludes:

- 78+ GB of local checkpoints and trainer state;
- downloaded Qwen, BERT and DeBERTa weights;
- LoRA `.safetensors` adapters;
- copied BANKING77 rows and generated training files;
- Python environments and Hugging Face caches;
- row-level predictions and training logs; and
- machine-specific manifests and absolute local paths.

Verify the exact prospective commit before publishing:

```bash
python3 scripts/check_public_repo.py
python3 -m compileall -q scripts
node --check learning/app.js
python3 scripts/serve_learning_lab.py --check
```

## Limitations

- BANKING77 is an educational intent taxonomy, not live bank traffic.
- The 94.12% model still misclassified 181 of 3,080 test requests.
- Several intents have ambiguous semantic boundaries.
- The local API is single-user, loopback-only demonstration code.
- No model weights are distributed in this repository.
- Results are specific to the recorded revisions, splits and Apple Silicon
  environment; exact floating-point reproduction is not guaranteed.

## Licence and attribution

Original code and learning materials are MIT licensed. Third-party artifacts
retain their own licences:

- Qwen3: Apache 2.0;
- BERT-Large: Apache 2.0;
- DeBERTa-v3-large: MIT; and
- BANKING77: CC BY 4.0, using the pinned PolyAI source recorded in
  [DATASET_SOURCES.md](data/DATASET_SOURCES.md).

Downloaded model weights and dataset rows are not redistributed here.

## Contributing

Reproductions on other Apple Silicon machines, corrections, accessibility
improvements and carefully controlled experiments are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

If the lab helps you, share both the result that worked and the hypothesis that
failed. That is the point of the project.
