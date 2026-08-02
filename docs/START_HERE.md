# Start here

This repository is a research notebook, a runnable codebase and a beginner
course. Use the path that matches why you opened it.

## I am new to model training

1. Run `python3 scripts/serve_learning_lab.py`.
2. Complete **First principles**, **Three model eras** and **Data discipline**.
3. Use the workbench to compare Qwen, BERT and DeBERTa.
4. Read the failure cards before reading training code.
5. Answer the interview flashcards aloud and complete the quiz.

You should finish able to explain:

- why the baseline comes first;
- why train, validation and test have different jobs;
- what LoRA changes and what remains frozen;
- why encoder classification fit this task better than generation;
- why a low loss does not prove high accuracy;
- why float32 fixed the DeBERTa MPS failure; and
- why Exp015 was a good experiment even though it lost.

## I want the engineering evidence

Read in this order:

1. [Experiment journey](EXPERIMENT_JOURNEY.md)
2. [Recorded results](RESULTS.md)
3. [Reproduction protocol](REPRODUCING.md)
4. [Script map](../scripts/README.md)
5. [Chronological state](../TRAINING_STATE.md)

The state file is intentionally detailed. It is the lab notebook, not the
landing page.

## I want to reproduce one experiment

Do not start with the champion. First prove your machine and data pipeline:

```text
environment check
      ↓
pinned data reconstruction
      ↓
untouched baseline
      ↓
two-update backward probe
      ↓
full training
      ↓
validation selection
      ↓
one final test report
```

Follow [REPRODUCING.md](REPRODUCING.md). Never overwrite an existing run or
use test results to choose a configuration.

## I am reviewing this for an interview

The compact answer is:

> I built a reproducible BANKING77 training lab on an M2 Max. I first adapted
> Qwen3 1.7B with LoRA and learned the complete generative SFT workflow,
> including numerical failure diagnosis and constrained decoding. Because the
> task was fixed-label classification, I then benchmarked BERT and DeBERTa
> encoders. I kept test data out of model selection, fixed DeBERTa's MPS
> optimizer instability by training in float32, and reached 92.99% validation
> and 94.12% reporting-only test accuracy. I also rejected several plausible
> refinements when they regressed validation performance.

Be ready to explain why switching architecture was an evidence-based decision,
not random model shopping.

## What is intentionally absent

The public source does not contain downloaded weights, checkpoints, adapters,
dataset rows, row-level predictions, virtual environments or local paths. Run
`python3 scripts/check_public_repo.py` to verify the boundary.
