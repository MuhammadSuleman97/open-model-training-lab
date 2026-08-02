# Experiment journey

Every experiment had a question and a promotion rule. “Completed” means the
pipeline worked; it does not automatically mean the model improved.

## Era A — Learn generative SFT with Qwen

| Experiment | Question | Evidence | Decision |
|---|---|---|---|
| Baseline | What can untouched Qwen do? | 47.8571% test; 176 invalid labels | Scientific control |
| 001 | Does the complete LoRA pipeline work? | Ten updates, adapter save/reload and smoke inference passed | Pipeline proven, not an accuracy claim |
| 002 | Does 539-row balanced training help? | Pilot accuracy 39.61%; macro F1 improved slightly | Scale data before judging |
| 003 | Can 1,925 rows train at the original rate? | Every adapter value became non-finite | Reject corrupted artifact |
| 003b | Does a lower rate stabilize it? | Stable; 47.37% full test | Keep evidence, not champion |
| 004 | Can batch size 7 accelerate full-data training? | NaN by iteration 50 | Reject configuration |
| 004c | Can batch size 1 and 5e-7 finish? | Stable; 49.42% test | First full-data improvement |
| 005 | Does q/k/v/o LoRA help at 5e-7? | NaN at iteration 2,050 | Retry more conservatively |
| 005b | Does half the learning rate stabilize q/k/v/o? | 50.39% test | Best unconstrained Qwen |
| Constraint | Can inference guarantee official labels? | 51.10% test; invalid outputs 181 → 0 | Promote Qwen system |

The Qwen era established the whole workflow: data preparation, token masking,
LoRA, numerical diagnosis, evaluation, controlled A/B comparison, CLI and API.
It also exposed an architecture mismatch: a generator had to spell an exact
class label instead of directly selecting one of 77 classes.

## Era B — Match the architecture with BERT

| Experiment | Question | Validation accuracy | Decision |
|---|---|---:|---|
| 006 | How does a purpose-built BERT-Large classifier perform? | **91.5584%** | BERT champion |
| 007 | Does class-balanced loss repair weak classes? | 91.1688% | Rejected; 3 fewer correct |
| 008 | Does label smoothing repair overconfident errors? | 91.2987% | Rejected; target errors unchanged |
| Ensemble | Do epoch-4 and epoch-5 probabilities complement each other? | 91.4286% | Rejected |

This era taught the most important architectural lesson: a model designed to
encode text and score fixed classes was far better suited to BANKING77 than
open-ended token generation.

## Era C — Diagnose and refine DeBERTa

| Experiment | Question | Validation accuracy | Decision |
|---|---|---:|---|
| 009a | Can DeBERTa train with the initial MPS setup? | NaN after update | Diagnose |
| 009b | Is the learning rate simply too high? | NaN even at 2e-6 | Not a simple LR issue |
| 009c | Is the word embedding the bad tensor? | Next tensor became NaN | Freezing one tensor insufficient |
| 009d | Does casting all parameters to float32 fix updates? | **92.5974%** | Stable architecture champion |
| 010 | Does train-only targeted oversampling help? | **92.8571%** | Promoted on validation |
| 011 | Can upper-layer-only continuation improve gently? | **92.9870%** | Current validation champion |
| 012 | Which train rows look suspicious out of fold? | 1,186 disagreements | Audit signal only |
| 013 | Does a hard-negative margin fix strong rivals? | 92.8571% | Rejected; harmed one answer |
| 014 | Does broader hard-negative coverage help? | 92.8571% | Rejected; same regression |
| 015 | Does removing 1,078 suspicious rows improve training? | 90.7792% | Rejected; 17 fewer correct |

Exp011 was selected only by validation evidence. Its subsequent test report was
2,899/3,080 correct: **94.1234% accuracy** and **0.941173 macro F1**. That test
result was never used to select Exp012–015.

## Failure taxonomy

The project preserves four different kinds of failure:

- **Pipeline failure:** code or packaging stopped despite intact artifacts.
- **Numerical failure:** parameters or loss became NaN/Inf.
- **Scientific failure:** training was healthy, but validation regressed.
- **Product limitation:** a metric improved while the system remained unsafe
  for real banking decisions.

Distinguishing these prevents unnecessary reruns and dishonest conclusions.

## Final retained model

Exp011 remains the validation champion:

- parent: Exp010 DeBERTa-v3-large;
- trainable encoder layers: 20–23 plus classification components;
- frozen encoder layers: 0–19;
- trainable parameters: 51,513,421;
- validation: 716/770, 0.929870 accuracy, 0.929701 macro F1;
- reporting-only test: 2,899/3,080, 0.941234 accuracy; and
- status: educational champion, not production deployment.
