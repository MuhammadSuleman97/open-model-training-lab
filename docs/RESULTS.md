# Recorded results

Scores are labelled by split. Qwen comparisons use the canonical 3,080-row
test set. Encoder experiments were selected using the fixed 770-row validation
set; test was opened only for a final reporting result and was excluded from
all later experiment decisions.

## Cross-architecture summary

| Experiment | Architecture | Split | Correct | Accuracy | Macro F1 | Status |
|---|---|---|---:|---:|---:|---|
| Qwen baseline | Qwen3 1.7B generator | Test | 1,474/3,080 | 0.478571 | 0.468243 | Control |
| 005b constrained | Qwen + LoRA | Test | 1,574/3,080 | 0.511039 | 0.496203 | Qwen champion |
| 006 | BERT-Large classifier | Validation | 705/770 | 0.915584 | 0.914837 | BERT champion |
| 009d | DeBERTa-v3-large | Validation | 713/770 | 0.925974 | 0.926117 | Superseded |
| 009d | Same selected model | Test report | 2,896/3,080 | 0.940260 | 0.940117 | Fixed report |
| 010 | DeBERTa continuation | Validation | 715/770 | 0.928571 | 0.928234 | Superseded |
| 011 | DeBERTa upper layers | Validation | **716/770** | **0.929870** | **0.929701** | Champion |
| 011 | Same selected model | Test report | **2,899/3,080** | **0.941234** | **0.941173** | Final report |
| 013 | Hard negatives | Validation | 715/770 | 0.928571 | 0.928234 | Rejected |
| 014 | Broader hard negatives | Validation | 715/770 | 0.928571 | 0.928234 | Rejected |
| 015 | Noise-pruned retrain | Validation | 699/770 | 0.907792 | 0.907391 | Rejected |

The lab's original 95% target was not reached. Exp011 missed it by 27 test
answers. It remains an educational result, not a production banking model.

## Qwen/LoRA comparison

| Configuration | Trainable values | Correct | Accuracy | Macro F1 | Invalid |
|---|---:|---:|---:|---:|---:|
| Untouched Qwen3 1.7B | 0 | 1,474 | 0.478571 | 0.468243 | 176 |
| 003b: q/v LoRA, 1,925 rows | 917,504 | 1,459 | 0.473701 | 0.479970 | 177 |
| 004c: q/v LoRA, 9,233 rows | 917,504 | 1,522 | 0.494156 | 0.497983 | 159 |
| 005b: q/k/v/o LoRA, 9,233 rows | 1,835,008 | 1,552 | 0.503896 | 0.499911 | 173 |
| 005b + canonical decoding | 1,835,008 | **1,574** | **0.511039** | 0.496203 | **0** |

The best Qwen task accuracy is only 51.10%. This phase demonstrates a complete
generative SFT method and measurable improvement, not production readiness.

## Encoder model selection

BERT changed the formulation from next-token generation to direct 77-class
scoring, raising validation accuracy by more than forty percentage points.
Class weighting, label smoothing and epoch ensembling did not improve the
BERT parent and were rejected.

DeBERTa initially failed numerically: its float16 checkpoint produced finite
forward losses, but pretrained parameters became non-finite after the first
AdamW update on MPS. Casting all parameters to float32 produced a stable
five-epoch run and 0.925974 validation accuracy.

Two controlled continuations improved validation:

- Exp010 train-only targeted oversampling: +2 correct versus Exp009d;
- Exp011 upper-layer-only refinement: +1 correct versus Exp010, with no
  parent-correct prediction harmed.

Exp011 was selected before its reporting-only test was opened.

## Data-quality experiments

Exp012 used five-fold out-of-fold predictions over only the 9,233 training rows
and identified 1,186 label disagreements. They were treated as review signals,
not automatic corrections.

- Exp013 used 162 high-probability rivals in a hard-negative objective and
  harmed one validation answer.
- Exp014 expanded coverage to 282 rivals and produced the same regression.
- Exp015 removed 1,078 rows where the out-of-fold model disagreed and assigned
  the supplied label less than 25% probability. Stable retraining fell to
  0.907792 validation accuracy—17 correct answers behind Exp011.

No labels were rewritten and no test rows were loaded by Exp012–015. These
results show that model disagreement is not sufficient evidence for automatic
deletion or relabelling.

## The matched constrained-decoding experiment

The original unconstrained evaluation used inference batch size 8 while the
constrained evaluator used batch size 1. Because batch shape can alter floating
point behavior, a second unconstrained batch-size-1 evaluation isolated the
decoding change.

| Batch-size-1 condition | Correct | Accuracy | Macro F1 | Invalid |
|---|---:|---:|---:|---:|
| Unconstrained 005b | 1,545 | 0.501623 | 0.499571 | 181 |
| Canonical-constrained 005b | 1,574 | 0.511039 | 0.496203 | 0 |

Paired transitions:

- both correct: 1,545;
- constrained only correct: 29;
- unconstrained only correct: 0;
- both wrong: 1,506;
- predictions changed: 181; and
- already-valid predictions changed: 0.

The constraint changed exactly the 181 invalid outputs. Twenty-nine became the
correct canonical label; 152 became valid but remained incorrect. The exact
two-sided McNemar p-value for the matched correctness transitions was
`3.72529029846e-09`.

The small macro-F1 reduction illustrates why one metric is not enough. The
constraint improves exact accuracy and guarantees a valid output contract, but
does not uniformly improve every class.

## Numerical failures that affected the design

### Experiment 003

The 1,925-row q/v run used learning rate `1e-5`. Training became non-finite by
the iteration-950 report; all 917,504 saved adapter values were non-finite. Peak
memory was only 5.029 GB, so the failure was numerical instability rather than
unified-memory exhaustion. Retrying at `2.5e-6` completed stably as 003b.

### Experiment 004

Batch size 7 reached a `NaN` train-loss report at iteration 50 despite fitting
within memory. A batch-size-1 probe was stable, and the full q/v run completed
at learning rate `5e-7` as 004c.

### Experiment 005

Expanding LoRA targets from q/v to q/k/v/o doubled trainable values. The first
full run was healthy through iteration 2,000 and became non-finite at 2,050.
Halving the learning rate to `2.5e-7` produced the stable 005b adapter.

These artifacts were retained as failed experiment records rather than treated
as successful checkpoints.

## Reproducibility identifiers

- Model: `Qwen/Qwen3-1.7B-MLX-bf16`
- Model revision: `720c04346ea2b095c801ebbd545c109230964cd4`
- Dataset repository: `PolyAI-LDN/task-specific-datasets`
- Dataset revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- SFT training records: 9,233
- Validation records: 770
- Test records: 3,080
- Stable 005b learning rate: `2.5e-7`
- Stable 005b adapter values: 1,835,008
- Recorded adapter SHA-256:
  `b6454e88a364ddae305243618d52d23ac83da5273c5a347707dda52882c8a7b1`
- BERT model: `google-bert/bert-large-cased`
- BERT revision: `a25a4b00fd23b14ba7b902af2b756931b6677ba9`
- DeBERTa model: `microsoft/deberta-v3-large`
- DeBERTa revision: `64a8c8eab3e352a784c658aef62be1662607476f`
- Exp011 selected weight SHA-256:
  `5c94f7bbeef65ce5d71d4be3c23d8cf88bbd92330408d69a9d04c4e2a42186bd`

Machine-readable summaries live in `evaluation/results/`; experiment metadata
lives in `experiments/*/result.json`. Compact encoder metrics live in
[`evaluation/encoder_results.json`](../evaluation/encoder_results.json); local
checkpoints and trainer outputs remain ignored.
