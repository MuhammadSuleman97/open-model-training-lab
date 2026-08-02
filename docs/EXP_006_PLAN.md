# Experiment 006 — BERT sequence classifier

## Question

Can an architecture designed for closed-set text classification reach at least
90% accuracy on BANKING77 on the same M2 Max, using the same public source and
held-out test split? The broader program will continue toward the user's 95%
launch target if this literature-backed BERT baseline falls short.

The metric is **accuracy**, not “efficiency.” Efficiency will be reported
separately through training time, peak memory, model size and inference speed.

## Why change architecture?

The Qwen experiment framed classification as text generation:

```text
request + list of 77 labels → generate the tokens of one label
```

That design introduced avoidable work and failure modes:

- the prompt was roughly 440 tokens because it listed the taxonomy;
- the model had to generate a multi-token label exactly;
- aliases and prose could become invalid outputs;
- constrained decoding fixed syntax but not semantic confusion; and
- stable q/k/v/o LoRA required a very small learning rate and only modestly
  changed the pretrained generator.

Experiment 006 instead uses:

```text
short request → bidirectional BERT encoder → 77 logits → argmax class
```

Every output is inherently one of the 77 classes. The loss directly optimizes
class probability rather than next-token probability.

## Evidence for the target

The original BANKING77 paper reports 93.66% full-data accuracy for its
fine-tuned BERT-Large classifier and 93.36% for USE+ConveRT. Therefore 90% is a
reasonable target for this dataset, although a different implementation,
split, seed, framework and device mean it is not guaranteed.

Primary sources:

- Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*:
  <https://aclanthology.org/anthology-files/pdf/nlp4convai/2020.nlp4convai-1.5.pdf>
- PyTorch MPS backend:
  <https://docs.pytorch.org/docs/stable/notes/mps.html>
- Hugging Face Trainer:
  <https://huggingface.co/docs/transformers/en/trainer>
- Pinned BERT-Large model:
  <https://huggingface.co/google-bert/bert-large-cased>

## Fixed design

- Model: `google-bert/bert-large-cased`
- Revision: `a25a4b00fd23b14ba7b902af2b756931b6677ba9`
- Licence: Apache 2.0
- Task head: 77-way sequence classification
- Device: PyTorch MPS on Apple Silicon
- Train: the existing 9,233 records
- Validation: the existing balanced 770 records
- Test: the canonical 3,080-row BANKING77 test split
- Seed: 3411
- Initial maximum length: 64, subject only to pre-training tokenization
  inspection
- Initial learning rate: `2e-5`
- Epoch ceiling: 5
- Selection metric: validation accuracy, with macro F1 also reported

PyTorch lives in `.venv-encoder`, separate from the completed `.venv` MLX
environment.

## Gates

1. **Environment gate:** PyTorch reports MPS available and completes a tensor
   operation on the GPU.
2. **Data gate:** labels map deterministically to IDs 0–76; no train/validation
   or train/test text overlap exists.
3. **Tokenization gate:** the selected maximum length is justified from all
   train/validation/test records before training.
4. **Pipeline gate:** a small pilot finishes with finite loss, saves a model and
   reloads it.
5. **Validation gate:** the full run must reach at least 88% validation accuracy
   before the test evaluator is unlocked.
6. **Experiment 006 gate:** at least 90% exact test accuracy, demonstrating the
   expected architecture correction.
7. **Public-launch target:** at least 95% on a justified follow-up experiment.
   The project will not tune repeatedly against test labels to force this
   number.

Every result will remain recorded even if it misses its gate. A BERT result
between 90% and 95% triggers error analysis and a documented Experiment 007
hypothesis—likely a stronger modern or conversational sentence encoder—not
silent metric shopping.

## Leakage rule

Hyperparameters and checkpoint selection use only training and validation data.
The test evaluator runs once after the configuration is frozen. The repository
will also disclose that the same standard test set was previously used for the
Qwen experiments and is a public benchmark, so this is not a private blind
evaluation.

## Comparison to publish

If successful, the public story becomes a controlled architecture lesson:

```text
1.7B causal generator + LoRA       51.10%
340M bidirectional classifier      Experiment 006 gate ≥ 90%
follow-up specialist               public-launch target ≥ 95%
```

The claim will be that matching the architecture and objective to the task
matters—not that smaller models always beat larger models.
