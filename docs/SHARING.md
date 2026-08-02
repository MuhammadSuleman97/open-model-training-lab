# Launch copy

Canonical repository target:
`https://github.com/MuhammadSuleman97/open-model-training-lab`

Verified interactive guide:
`https://muhammadsuleman97.github.io/open-model-training-lab/`

## GitHub repository description

> Train Qwen, BERT and DeBERTa on BANKING77 from an M2 Max—15 controlled experiments, failures included, plus an interactive beginner guide.

## GitHub topics

`apple-silicon`, `banking77`, `bert`, `deberta`, `fine-tuning`, `llm-evaluation`,
`lora`, `machine-learning`, `mlx`, `mps`, `open-models`, `pytorch`, `qwen3`,
`reproducible-research`, `transformers`

## X launch post

> I set out to learn open-model training on my M2 Max—and aimed for 95% on BANKING77.
>
> 15 controlled experiments later:
> • Qwen + LoRA: 51.1%
> • BERT: 91.6% validation
> • DeBERTa: 94.12% test
> • “cleaner” data experiment: failed
>
> I published the code, failures and interactive beginner guide:
> https://muhammadsuleman97.github.io/open-model-training-lab/

## X thread

### 1/5

> I wanted to understand model training—not just run a notebook. So I built a reproducible BANKING77 lab on an M2 Max and kept every important failure. Qwen → BERT → DeBERTa, 15 experiments, one honest result. 🧵

### 2/5

> Qwen3 1.7B taught me the complete generative workflow: baseline first, chat-format SFT, LoRA, NaN diagnosis, held-out evaluation, constrained decoding and a local API. It improved 47.86% → 51.10%, but generation was the wrong shape for a 77-label task.

### 3/5

> Switching to an encoder classifier was the breakthrough. BERT reached 91.56% validation. DeBERTa initially became NaN after its first MPS optimizer update; tracing the first bad tensor showed the float16 update path was unstable. Casting to float32 fixed it.

### 4/5

> The selected DeBERTa refinement reached 92.99% validation and 94.12% on the reporting-only test. Then I tried pruning 1,078 suspicious training labels. The run was perfectly stable—and fell to 90.78%. Healthy training ≠ a successful hypothesis.

### 5/5

> The repository includes configs, checksums, compact metrics, rejected experiments, an interactive course, glossary, quiz and interview practice. It excludes 78+ GB of local weights/data/checkpoints. Reproduce it or tell me what I should test next:
> https://muhammadsuleman97.github.io/open-model-training-lab/

## Suggested launch image text

```text
ONE TASK · THREE MODELS · FIFTEEN EXPERIMENTS

Qwen + LoRA        51.10%
BERT-Large         91.56% validation
DeBERTa-v3-large   94.12% test

FAILURES INCLUDED
M2 MAX · 32 GB · BANKING77
```

## DEV article title and opening

**Title:** I trained Qwen, BERT and DeBERTa on an M2 Max—here are the failures the notebook tutorials skip

> My first model finished training and still classified fewer than half the
> evaluation examples correctly. That was the beginning of the useful part.
> This is the story of how a generative Qwen experiment became a 15-experiment
> lesson in architecture fit, numerical precision, evaluation discipline and
> label noise.

Disclose that AI tools assisted code and documentation review, while the
experiments were run locally and their outputs were retained and verified.

## Show HN title

> Show HN: A failure-inclusive open-model training course built on an M2 Max

Post only after the live guide works without setup. Stay in the discussion and
answer technical questions with exact evidence.

## Never claim

- that 94.12% reached the 95% target;
- that test data selected Exp011 or later experiments;
- that the classifier is production-ready;
- that out-of-fold disagreement proves a label is wrong;
- that all runs succeeded; or
- that model weights or BANKING77 rows are included in GitHub.
