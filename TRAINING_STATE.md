# Open Model Training Lab — State

Last updated: 2026-08-03 (Australia/Sydney)
Status: Complete; retained as a reproducible learning lab

## Current phase

Phase 1 — Supervised fine-tuning fundamentals (complete)

Phase 2 — Controlled model-improvement experiments (complete)

Phase 3 — Learning consolidation and interview preparation (complete)

Phase 4 — Open-source publication preparation (complete)

Phase 5 — Architecture-appropriate accuracy improvement (complete)

## Current step

Step 107 — Preserve the champion artifacts and archive the completed lab.

The local workspace was conservatively cleaned after GitHub, GitHub Pages,
X and Hugging Face publication. Its disk footprint fell from approximately
83 GB to 1.7 GB. Rebuildable Python environments, downloaded base-model
caches, rejected and duplicated checkpoints, interrupted runs, training logs,
generated prediction rows and temporary files were removed. The canonical
Experiment 011 DeBERTa selected model remains local at 1.6 GB, and its
`model.safetensors` SHA-256 was reverified as
`5c94f7bbeef65ce5d71d4be3c23d8cf88bbd92330408d69a9d04c4e2a42186bd`.
The promoted Experiment 005b Qwen LoRA adapter also remains local. Git history
passed `git fsck`, and `main` remained synchronized with `origin/main` before
this state update.

The public source package has been restructured after Exp015. The root README
now tells the complete Qwen → BERT → DeBERTa story, clearly labels validation
versus reporting-only test metrics, leads with the 94.1234% result without
claiming the missed 95% target, and points readers into three depths: interactive
guide, experiment journey and reproduction protocol. New reader-facing files
include `docs/START_HERE.md`, `docs/EXPERIMENT_JOURNEY.md`,
`scripts/README.md`, `docs/PRESENCE.md`, refreshed launch copy, a GitHub profile
README template and a Hugging Face static Space card. Compact encoder results
are now published as `evaluation/encoder_results.json` while all local encoder
weights and trainer artifacts remain ignored.

The repository now contains a GitHub Pages workflow that publishes only the
static `learning/` directory. The presence plan prioritizes GitHub as canonical
evidence, X as ongoing distribution, Hugging Face as the ML-native runnable
surface, DEV for the long-form article, and optional Show HN after the guide is
live. Reddit is deliberately deferred until the user has meaningful community
participation; LinkedIn is excluded by preference.

The prospective public commit contains 198 files and approximately 1.17 MB.
The publication checker confirms that 78+ GB of local artifacts, downloaded
weights, adapters, generated dataset rows, environments, logs and row-level
predictions remain ignored. It also rejects common checkpoint suffixes, access
token formats, absolute macOS user paths, the private corporate Git identity
and the local source-wrapper path. Python compilation, JavaScript syntax,
workflow YAML parsing, public JSON parsing, local Markdown links, guide
preflight and the public-artifact check all pass.

GitHub publication is complete under the renamed account `msulemans`. The
sanitized repository is public at
`https://github.com/msulemans/open-model-training-lab`; the initial
public commit uses Muhammad Suleman and a GitHub noreply address. Repository
topics, Discussions and the homepage are configured, and the public validation
workflow passes. The interactive guide is deployed through GitHub Actions at
`https://msulemans.github.io/open-model-training-lab/` and returns HTTP
200 without authentication.

The account-level Pages configuration previously pointed to the unused custom
domain `iamsuleman.me`, causing every project page to redirect to an unresolved
host. The user confirmed that the domain is no longer owned. Its Pages custom
domain was removed, HTTPS was enabled on the restored `github.io` URLs, and the
learning guide was redeployed successfully.

The public profile README repository created before the username change has a
concise README that links to the repository and
interactive guide. Both profile URLs return HTTP 200 without authentication.
The renamed GitHub account is `msulemans`; its bio and website
are empty. Updating those two fields through the API requires the additional
GitHub CLI `user` scope; the current token has repository and workflow access
but GitHub correctly rejected the profile mutation without that scope.

The interactive learning guide has now been expanded into a three-model flight
recorder. It explains, in beginner language, the complete Qwen generative-LoRA
era, the BERT-Large encoder pivot, the DeBERTa-v3-large numerical diagnosis and
champion refinements, and the completed Exp015 data-quality hypothesis. Its
workbench distinguishes validation scores from reporting-only test scores; its
timeline records why every experiment was run; and its failure lab now includes
rejected hypotheses as well as software and numerical incidents. New glossary,
flashcard and quiz material covers encoder classifiers, logits, classifier
heads, full versus partial fine-tuning, MPS, float32, out-of-fold auditing,
label noise, hard negatives, noise pruning and promotion rules. This learning
guide update changes no training configuration, dataset, checkpoint or active
experiment artifact.

The user correctly identified training as the split that updates LoRA,
validation as the split used to inspect and tune experiments, and test as the
untouched final exam. The lesson is paused before the next ML topic while the
lab is prepared for open-source publication. The repository now has a
results-first README, MIT licence, contribution/security guidance, exact
reproduction notes, CI, a public-artifact checker and ready-to-edit X launch
copy. The prospective public source is 114 files and approximately 610 KB;
model weights, adapters, copied data, logs, caches, environments and row-level
predictions are confirmed ignored. The public identity is now Muhammad Suleman,
GitHub user `msulemans`, with the recommended repository name
`open-model-training-lab`. Publication is now deliberately paused. The new
goal is at least 95% BANKING77 accuracy through an architecture-appropriate
sequence classifier. The original BANKING77 paper reports 93.66% for full-data
BERT-Tuned, so Experiment 006 will fine-tune pinned BERT-Large with a 77-class
head using PyTorch MPS in an isolated environment. Experiment 006 has a 90%
gate; if it lands between 90% and 95%, a separately justified Experiment 007
will pursue the launch target without test-set metric shopping. The isolated
environment installed successfully and is now locked. PyTorch 2.13.0 has also
completed a real matrix calculation through the M2 Max GPU using MPS. The
pinned 24-layer BERT-Large checkpoint has been downloaded and verified; its
1.247 GiB legacy PyTorch weight artifact is recorded in a local manifest.
The checkpoint and its newly initialized 77-class head have completed a finite
one-request forward pass on MPS using approximately 2.008 GiB of driver memory.
The raw-text encoder dataset is also prepared with 9,233 training, 770
validation and 3,080 sealed test records, with zero exact-text overlap across
all splits. Development token inspection found 32 of 10,003 records above 64
tokens and a maximum of 98, so the ceiling is now 128 with dynamic padding.
The corrected two-update backward probe then produced loss 4.413, finite
gradients, a 10.321 GiB peak and an 11.39 samples/second rate over the longest
records, confirming the full training path and loss normalization.
Experiment 006 then completed all five epochs in 41 minutes. Validation
accuracy progressed 72.21%, 87.14%, 90.39%, 91.56% and 91.56%; the predefined
accuracy-only tie policy selects the first 91.56% checkpoint at epoch 4. The
test set remained sealed. Final packaging stopped after training because an
overly broad `*.bin` check also counted `training_args.bin`. Trainer's internal
best-model restore additionally bypassed BERT's legacy LayerNorm key conversion;
the intact epoch-4 and epoch-5 checkpoints remain available.
The supported checkpoint-loading path subsequently reproduced epoch 4's
validation loss 0.371551, accuracy 0.915584 and macro F1 0.914837. The selected
model then passed an exact save/reload prediction check and is recorded by
SHA-256. Experiment 006 is finalized; it passed the 90% gate but not the 95%
launch target, while the test set remains sealed.
Validation error analysis confirmed 705 correct and 65 errors. The weakest
intent was `topping_up_by_card` at 5/10, and the largest confusion was
`transfer_not_received_by_recipient` versus
`balance_not_updated_after_bank_transfer` at three examples. Several errors
are high-confidence but linguistically ambiguous under the supplied label,
which makes annotation/taxonomy ambiguity part of the remaining gap.
Epoch-4/epoch-5 probability averaging then reduced validation accuracy to
0.914286. The checkpoints disagreed on only ten records, repaired four unique
errors each, and had an oracle-union ceiling of 0.920779. This rejects the
ensemble hypothesis. The initial plan made DeBERTaV3-Large Experiment 007, and
SentencePiece 0.2.2 was installed for its tokenizer. The user correctly
challenged that switch as premature for a model-optimization learning project.
DeBERTa is therefore deferred to a later architecture benchmark. Experiment
007 now continues from the roundtrip-verified BERT model and isolates
class-balanced loss as its first refinement. The exact encoder environment
remains locked and the test set remains sealed.
The train-only refinement manifest now contains 77 finite inverse-square-root
weights normalized to mean 1.0, ranging from 0.789350 to 2.100324. Two weak
validation classes, `card_acceptance` and `card_swallowed`, receive weights
1.500231 and 1.470521 respectively, supporting the refinement hypothesis.
The two-update weighted-loss probe passed with loss 0.101737, finite gradients,
9.728 GiB peak memory and no validation/test access or saved weights.
The two-epoch refinement completed in 14.48 minutes. Both epochs scored
0.911688 validation accuracy and 0.911438 macro F1, three fewer correct answers
than the 0.915584 parent. The strict promotion rule rejected the child, so the
roundtrip-verified Experiment 006 model remains canonical and the test remains
sealed.
Transition analysis showed ten changed predictions: three parent errors were
fixed, but six correct parent predictions were harmed. Upweighted classes had
zero fixes and three harms, confirming that the proposed mechanism failed.
Experiment 007 is closed as rejected. Experiment 008 returns to the unchanged
canonical Experiment 006 model and isolates label smoothing 0.05, motivated by
the thirteen validation errors made above 90% confidence.
The two-update Experiment 008 probe passed with finite loss 0.462308, finite
gradients, a 9.541 GiB peak and no validation/test access or saved weights.
The full two-epoch label-smoothed refinement completed stably in 14.70 minutes.
Epochs one and two reached 0.911688 and 0.912987 validation accuracy; the best
child remained two correct answers behind the 0.915584 parent. The strict rule
rejected the child, Experiment 006 remains canonical, and the test stays sealed.
Transition analysis found thirteen changed predictions: four parent errors were
fixed but six correct parent predictions were harmed. Crucially, label smoothing
fixed zero of the thirteen high-confidence parent errors it was designed to
address and harmed no high-confidence correct prediction. The mechanism did not
solve its target problem, so Experiment 008 is closed as rejected.
The BERT champion has now survived epoch ensembling and two controlled
same-model continuations. An architecture benchmark is therefore justified.
Experiment 009 will compare pinned MIT-licensed DeBERTa-v3-large using the same
train/validation splits and sealed test, while Experiment 006 remains canonical.
The exact 24-layer checkpoint downloaded successfully at the pinned revision.
Its 0.814 GiB base weight file passed structural checks and is recorded with
SHA-256 `dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5`.
The first smoke-test attempt stopped safely before model loading because the
isolated encoder environment had SentencePiece but not Protobuf, which
Transformers requires to extract this checkpoint's tokenizer. The subsequent
TikToken error was only a failed fallback, not the root cause. No result file
or model output was produced, and no model/data state changed.
Pinned Protobuf 7.35.1 is now installed and verified inside `.venv-encoder`.
The retry smoke test passed: the fresh 77-class head produced finite `[1, 77]`
logits on MPS with 1.040 GiB peak driver memory. The model's missing classifier
and pooler tensors are expected downstream-head initialization, while the
pretrained DeBERTa tensors loaded successfully. Transformers also emitted a
non-blocking generic tokenizer-regex warning; tokenization and execution were
valid.
The first DeBERTa backward probe then failed safely at update 2: loss changed
from 4.395 to `nan` at learning rate `2e-5`. No validation/test data or weights
were saved. We preserve this as Experiment 009's numerical-stability finding
and isolate a ten-times-lower learning rate in Experiment 009b.
The 2e-6 retry also became `nan` on its second update, so the issue is not
resolved by a simple ten-times learning-rate reduction. No weights or held-out
data were touched. The next diagnosis instruments gradients, parameters and
forward losses after each microbatch using the native slow SentencePiece
tokenizer.
The diagnosis localized the first non-finite values to the pretrained
`deberta.embeddings.word_embeddings.weight` immediately after optimizer step 1;
all eight forward losses before that update were finite. Experiment 009c freezes
only this lexical embedding matrix and leaves the remaining encoder parameters
and new classifier head trainable. Experiment 009c still became `nan` on its
second update, so freezing embeddings is insufficient. We will locate the next
non-finite tensor before attempting another workaround.
The second diagnosis moved the first bad tensor to
`deberta.embeddings.LayerNorm.weight`. Direct inspection of the pinned
checkpoint found all 402 pretrained tensors are `float16`. This identifies
half-precision AdamW updates on MPS as the common cause. Experiment 009d casts
the whole model to float32, unfreezes every parameter, and repeats the original
`2e-5` probe.
The float32 probe passed: losses were 4.395 and 4.311, both updates remained
finite, all parameters were trainable, and peak MPS memory was 16.249 GiB. The
full five-epoch run then completed safely within the 32 GB machine. Validation
accuracy progressed 67.40%, 85.58%, 90.26%, 91.69% and 92.60%; epoch 5 is the
validation winner at 0.925974, eight answers above the BERT champion's 705/770.
The test set remains sealed. The training wrapper failed only while writing its
metadata because the config lacked an `evaluation` block; the checkpoints and
trainer state were intact. The config block and validation-only finalizer have
now repaired and verified the run. The selected model has validation loss
0.350554, accuracy 0.925974 and macro F1 0.926117. Its saved model round trip
is exact, and its weight SHA-256 is
`59b0d419bd5c6d992123a08ba880829cd6585a99b3f598d4d3a5a884d052a8d6`.
It was the validation champion, but promotion waited for the sealed test. The
one-time final exam is now complete: 2,896/3,080 correct, 0.940260 accuracy,
0.940117 macro F1 and 0.281448 loss. The test score is 0.014286 above
validation. The 95% target requires 2,926 correct, so 30 additional correct
answers are needed. The test is now an observed final benchmark and must not be
used to select another experiment; any refinement must use training and
validation only, followed by a clearly labelled comparison against this fixed
test result.

## Next action

The GitHub profile-scope update and pinning remain optional follow-up work.
The static guide is now also published as the public Hugging Face Space
`https://huggingface.co/spaces/msulemans/open-model-training-lab`. Its commit
uploaded only `README.md`, `index.html`, `styles.css` and `app.js`; no model
weights or dataset rows were uploaded. Next, publish the prepared DEV article
and optionally submit the interactive guide to Show HN. Do not distribute
local model weights until their licences, model card and intended-use
limitations have been reviewed explicitly.

Experiment 015 is complete and rejected. Its five validation accuracies were
`0.529870`, `0.853247`, `0.896104`, `0.906494`, and `0.907792`; epoch 5 was
best at 699/770 correct with macro F1 `0.907391`. This is 14 fewer correct than
the original full-data DeBERTa Exp009d (713/770) and 17 fewer than the Exp011
champion (716/770). The strict validation change versus Exp011 is `-0.022078`.
Training itself was stable: five epochs completed in 3,687.391 seconds with
17.775 GiB peak driver memory. No validation rows entered training, no labels
were rewritten, no test rows were loaded, and no test evaluation occurred.
The pruning hypothesis is therefore rejected without opening test. Exp011
remains the canonical validation champion. Do not finalize, promote, or test
the Exp015 child. The next model-improvement hypothesis has not yet been
selected.

## Historical notes — Exp013 onward

The Exp013 full continuation completed safely but regressed validation from
0.929870 to 0.928571 (macro F1 0.928234); the strict promotion rule rejects
it. The checkpoint is now finalized and round-trip verified with weight SHA
`14026ca51286e6c166b37531ca70710984cfb64777d6cb0fd111826ca2cfa5c2`.
Its validation transition analysis found one parent-correct
`contactless_not_working` example harmed and zero fixes. Exp011 remains the
champion. The read-only error report found 54 errors, 32 high-confidence
errors (confidence at least 0.9), and only 2 low-margin errors. The dominant
boundaries are exchange charge versus card exchange rate, transfer decline
versus failure, pending transfer versus pending card payment, and identity
verification. The next experiment must be justified by this validation-only
evidence; the sealed test remains permanently excluded from selection.

## Historical development notes

Validation-only analysis is complete. It found 713 correct and 57 errors,
matching the finalized validation metrics exactly. The weakest labels are
`topping_up_by_card` (6/10), `contactless_not_working` (7/10), and
`pending_top_up` (7/10). The leading confusion pairs are
`card_payment_wrong_exchange_rate -> exchange_charge`,
`topping_up_by_card -> top_up_reverted`, and
`declined_transfer -> failed_transfer`. These are development observations;
the 0.940260 test score is not being used to select the next experiment.
The next experiment will use only original training rows, preserve the same
770-row validation split, and start with a small stability/data-preparation
probe before any long run. The deterministic preparation script duplicated
2,124 targeted training rows once, producing 11,357 rows across 18 labels. It
used no synthetic text and no validation/test rows. The expanded train
checksum is `e981ec0a01169b53def3779027d74936fab19574d03d42e134af57620d3735f7`.

The holdout correction is complete: Experiment 010 was selected only by the
existing validation split, and its scripts never read the observed test files.
The one-epoch continuation completed with 0.928571 validation accuracy, up
from 0.925974 (two additional correct validation predictions). The child has
now been independently finalized and round-trip verified. Transition analysis
found 5 child-only fixes, 3 parent-only correct predictions, 710 unchanged
correct predictions and only 9 changed predictions. The child is therefore a
valid validation champion. However, the targeted weak labels did not improve:
`topping_up_by_card`, `contactless_not_working`, and `pending_top_up` stayed at
6/10, 7/10 and 7/10. The net gain came from `verify_my_identity`,
`card_payment_not_recognised` and `compromised_card`. The oversampling
hypothesis is useful but not proven for its intended classes.

No command is required for this bookkeeping step. Do not evaluate the test
set; the 0.940260 result remains the fixed Experiment 009d benchmark and is
not transferable to Experiment 010.

The next experiment is now defined from validation-only evidence. Experiment
011 keeps Experiment 010 as its parent, duplicates only train rows from the
known confusion-boundary labels, and freezes DeBERTa layers 0-19 while
training layers 20-23, the pooler and classifier at a conservative learning
rate. Its preparation produced 11,496 train rows, 2,263 duplicated rows and
20 target labels. The train checksum is
`01200ad8491a79d7a2f8256bd86be906fc916d3cbc6fd0360079a4abcfbebe48`.
Validation and test rows loaded during preparation: zero.

The probe script is ready, but the current Codex shell cannot access MPS, so
the probe must be run from the user's verified `.venv-encoder` on the M2 Max.
This is an environment limitation, not a training failure. Do not start the
full Experiment 011 run until the probe reports finite loss and
`exp_011_probe_ok: True`. The first user-side probe also stopped safely before
training because DeBERTa exposes its pooler as root-level `pooler.*`, not
`deberta.pooler.*`. The freeze rule has been corrected; no probe result or
weights were saved from that attempt. The corrected probe then passed with
loss `0.021563`, 51,513,421 trainable parameters, 383,627,264 frozen
parameters, 4.078 GiB peak memory, validation/test rows loaded 0, and no
weights saved. The one-epoch full continuation then completed successfully
with train loss `0.067731`, validation loss `0.360661`, validation accuracy
`0.929870`, validation macro F1 `0.929701`, one additional correct validation
prediction over Exp010, 5.329 GiB peak memory and 257.534 seconds elapsed.
Test rows loaded and test evaluation remain zero/false. Validation-only
finalization passed: the selected model round trip is exact and its weight
SHA-256 is
`5c94f7bbeef65ce5d71d4be3c23d8cf88bbd92330408d69a9d04c4e2a42186bd`.
The transition analysis then found exactly one changed validation prediction:
one parent error was fixed, no parent-correct prediction was harmed, and the
child reached 0.929870 accuracy / 0.929701 macro F1. Exp011 is therefore the
validation champion. A reporting-only test evaluator is now ready; its output
must not be used to choose another experiment. The reporting-only test
evaluation is now complete: 2,899/3,080 correct, test accuracy `0.941234`,
macro F1 `0.941173`, and loss `0.284209`. This is 3 more correct test
examples than fixed Exp009d (`0.940260`), a `+0.000974` accuracy change
(`+0.0974` percentage points). The 95% target requires 2,926/3,080 correct,
so 27 more would be needed. `launch_target_reached_on_test` is false. This
result is reporting-only and must never be used to select, tune, or justify a
future experiment. The user chose to continue with a data-quality approach.
Exp012 has now completed a five-fold out-of-fold audit using only the 9,233
training rows. The lightweight word-TF-IDF/logistic-regression audit reached
0.871548 OOF accuracy and found 1,186 training-row disagreements; these are
review candidates, not automatic relabel decisions. The audit loaded zero
validation rows and zero test rows. The source checksum is
`eef5f18881164886425835e3ff0ca4a751d692780cadf16b833eab62369ec59c`.
Review of the strongest disagreements found mostly semantic boundary cases,
not safe typo-level corrections: payment-not-recognised versus direct-debit,
pending-transfer versus transfer-timing, disposable-card access versus
limits, and cash-withdrawal versus card-payment exchange rates. We will not
delete or relabel these rows automatically. The next refinement will retain
the original labels and add a train-only hard-negative objective so the model
learns to separate each true label from its strongest rival.
Exp013 data preparation is complete: it retains all 9,233 original training
rows, activates 162 train-only rival labels whose OOF rival probability is at
least 0.5, rewrites zero labels, and records train checksum
`9fb9f502d91fa5ae6c88ded35a5a0a109032f773b14efd53cef104ebcada0f0c`.
Validation rows loaded and test rows loaded remain zero.
The Exp013 two-update MPS probe then passed with finite total and pairwise
losses, 51,513,421 trainable parameters, 383,627,264 frozen parameters, a
3.224 GiB peak, and zero validation/test rows. The full run is now guarded by
the same split checks and a custom finite-loss stop. The one-epoch run then
completed with total train loss 0.080457, mean cross-entropy 0.060291, mean
pairwise loss 0.080855, 162 active hard-negative rows, and a 5.329 GiB peak.
Validation accuracy was 0.928571 versus the Exp011 parent at 0.929870, so the
child is rejected without reading the test set. Finalization and round-trip
verification produced weight SHA
`14026ca51286e6c166b37531ca70710984cfb64777d6cb0fd111826ca2cfa5c2`.
Transition analysis found one changed prediction:
`contactless_not_working -> change_pin`; it was a parent-only correct answer,
so Exp013 fixed zero validation errors and harmed one correct prediction.
The Exp011 champion error report then reproduced 716/770 validation accuracy
and 0.929701 macro F1. Its weakest labels were `topping_up_by_card` (6/10)
and `pending_top_up` (7/10). Representative high-confidence errors include
`card_payment_wrong_exchange_rate -> exchange_charge`,
`pending_transfer -> pending_card_payment`, and
`unable_to_verify_identity -> verify_my_identity`. Several texts are
semantically ambiguous under the BANKING77 taxonomy, so the next refinement
must improve boundaries without treating validation text as training data.
Exp014 is therefore defined as a controlled coverage test: it keeps the Exp013
margin, weight, learning rate and trainable layers unchanged, but lowers the
train-only OOF rival threshold from 0.5 to 0.4. Preparation and the finite-loss
probe must pass before any full run; validation and test remain excluded from
data preparation and probing. Preparation is complete: 9,233 source training
rows, 282 active rivals, original labels retained, checksum
`b4fc8eccb3ccd7b5f8e088ba6dcd2e836c13205cbd2de84ffb24f3ede315263e`, and zero
validation/test rows. The two-update MPS probe passed with finite loss,
51,513,421 trainable parameters, 383,627,264 frozen parameters, and a 3.185
GiB peak. The full run then completed safely with all 282 rival rows seen, but
validation accuracy fell from the Exp011 parent's 0.929870 to 0.928571. This is
the same one-example regression as Exp013, so the broader hard-negative
hypothesis is rejected and Exp011 remains the validation champion. Test rows
loaded and test evaluations remained zero.
Published BANKING77 label-quality work reports more than 1,400 suspected noisy
training labels and a 4.5% supervised-classifier improvement after removing
flagged examples. This aligns closely with the lab's independent Exp012
train-only audit, which found 1,186 OOF disagreements. Exp015 therefore tests
data quality directly: it removes, but never relabels, rows selected by the
pre-registered Exp012 rule of OOF disagreement plus given-label probability
below 0.25. Preparation retained 8,155 of 9,233 training rows, removed 1,078
(11.68%), kept all 77 labels with at least 16 examples each, and produced train
checksum `7f449e5f3551f9e195e31ab0f0b4e8c81a32b6a44739a97eecbfedf04722e357`.
Validation and test rows loaded remain zero. Exp015 will retrain from the
original DeBERTa-v3-large checkpoint using Exp009d's seed, optimizer, learning
rate, epoch count and 10% warmup proportion so the isolated experimental
change is the filtered training data. A two-update
finite-loss MPS probe passed with loss 4.359903, finite gradients, a 16.249 GiB
peak, no saved weights and zero validation/test rows. The full five-epoch run
is now ready and will select checkpoints using only the unchanged 770-row
validation split.

## Verified hardware

- Machine: MacBook Pro (`Mac14,5`)
- Chip: Apple M2 Max
- CPU: 12 cores (8 performance, 4 efficiency)
- GPU: 30 cores
- Unified memory: 32 GB
- Architecture: `arm64`
- Metal: supported

## Verified software

- macOS: 27.0, build `26A5388g`
- Lab Python: 3.11.9 in an isolated `.venv`
- Lab Python executable: `.venv/bin/python`
- Lab pip: 26.2
- Xcode developer directory: `/Applications/Xcode.app/Contents/Developer`
- Git: 2.50.1
- `uv`: not installed
- `mlx`: 0.32.0
- `mlx-lm`: 0.31.3
- `datasets`: 5.0.1

## Project choices

- Primary framework: MLX / MLX-LM on Apple Silicon.
- Initial learning task: public-dataset intent classification.
- First dataset: BANKING77 via pinned revision
  `57ec275d8078af65b7731c2a98be812d844a6d6b` of the original
  `PolyAI-LDN/task-specific-datasets` repository.
- Initial model: `Qwen/Qwen3-1.7B-MLX-bf16`.
- Initial model revision:
  `720c04346ea2b095c801ebbd545c109230964cd4`.
- Model mode for classification: thinking disabled.
- Initial precision: BF16 LoRA, not quantized, so quantisation is not a
  confounding variable in the first experiment.
- Baseline workflow: a deterministic balanced 154-example pilot first, then the
  complete 3,080-example test evaluation after the evaluator is validated.
- Evaluation scoring uses exact canonical labels. Near-miss aliases such as
  `transfer_failed` for `failed_transfer` remain invalid rather than being
  silently normalized.
- Validation split: 10 examples per intent selected only from the original
  training split with seed 3407.
- Training loss will mask the system/user prompt through the assistant
  generation boundary. With MLX-LM 0.31.3 and Qwen3, the learned assistant
  completion contains Qwen's empty thinking wrapper followed by the intent
  label; inference remains explicitly thinking-disabled.
- Training approach: establish a baseline, then use LoRA.
- State tracking: this file is the canonical checkpoint.
- Interaction style: one step at a time; remain on a failed step until resolved.

## Completed

- [x] Expanded the self-contained interactive learning guide across Qwen,
  BERT-Large and DeBERTa-v3-large, including architecture decisions, rejected
  experiments, numerical diagnoses, Exp015 data-quality work, terminology,
  quiz and interview practice.

- [x] Created the local project structure.
- [x] Created the canonical state tracker.
- [x] Inspected the host architecture, chip, memory, GPU, and Metal support.
- [x] Inspected the active Python, Xcode, and Git environment.
- [x] Confirmed that MLX training dependencies are not yet installed.
- [x] Created an isolated Python 3.11.9 virtual environment.
- [x] Made the Metal report resilient to missing `system_profiler` display data.
- [x] User reproduced the environment verification.
- [x] Installed and recorded MLX training dependencies.
- [x] Saved the exact resolved environment to `requirements.lock`.
- [x] Ran a minimal MLX compute check through the M2 Max GPU.
- [x] Selected the initial model and recorded its immutable revision.
- [x] Downloaded and validated the selected model.
- [x] Downloaded and inspected canonical BANKING77.
- [x] Ran and recorded one untouched-model smoke prediction.
- [x] Prepared a deterministic balanced 154-example evaluation pilot.
- [x] Evaluated the untouched model on the balanced pilot.
- [x] Established and recorded the complete untouched-model baseline.
- [x] Prepared leakage-safe SFT train and validation files.
- [x] Inspected all SFT token lengths using the exact MLX-LM chat processing
  behavior.
- [x] Ran the first LoRA pipeline-validation experiment.
- [x] Loaded the saved LoRA adapter and ran a fixed smoke prediction.
- [x] Prepared the balanced 539-example training experiment.
- [x] Trained Experiment 002 for one pass over 539 balanced examples.
- [x] Evaluated Experiment 002 on the fixed balanced test pilot.
- [x] Prepared the nested 1,925-example training experiment.
- [x] Completed stable Experiment 003b training.
- [x] Recorded the failed high-learning-rate run as unusable.
- [x] Evaluated stable Experiment 003b on the fixed balanced test pilot.
- [x] Evaluated stable Experiment 003b on the complete test set.
- [x] Probed batch size 7 for full-data training.
- [x] Trained a stable full-data experiment on all 9,233 SFT records.
- [x] Recorded the failed batch-size-7 full run.
- [x] Probed batch size 1 and learning rate `5e-7`.
- [x] Trained Experiment 004c for one full-data epoch.
- [x] Evaluated Experiment 004c on the fixed balanced pilot.
- [x] Evaluated Experiment 004c on the complete test set.
- [x] Completed paired error analysis against the untouched model and
  Experiment 003b.
- [x] Ran an interactive request through the trained adapter.
- [x] Restarted the classifier API with single-threaded Metal generation.
- [x] Called `POST /classify` and received a JSON prediction.
- [x] Stopped the localhost API and released its model resources.
- [x] Probed q/k/v/o LoRA attention targets for Experiment 005.
- [x] Recorded Experiment 005 full run as failed and unusable.
- [x] Trained stable Experiment 005b for one full-data epoch.
- [x] Evaluated Experiment 005b on the fixed balanced pilot.
- [x] Evaluated Experiment 005b on the complete test set.
- [x] Compared Experiment 005b pairwise with Experiment 004c.
- [x] Designed the canonical-label adherence experiment.
- [x] Evaluated canonical-constrained Experiment 005b on the fixed pilot.
- [x] Evaluated canonical-constrained Experiment 005b on the complete test set.
- [x] Evaluated unconstrained Experiment 005b at batch size 1 on the pilot.
- [x] Evaluated unconstrained Experiment 005b at batch size 1 on the full test.
- [x] Completed an exact paired comparison of matched batch-size-1 outputs.
- [x] Promoted constrained Experiment 005b into the one-request classifier.
- [x] Verified the promoted constrained one-request classifier from Terminal.
- [x] Aligned the localhost API with constrained Experiment 005b.
- [x] Started the constrained Experiment 005b API from Terminal.
- [x] Called the promoted API and verified its complete constrained response.
- [x] Stopped the promoted API cleanly and released port 8080.
- [x] Opened and served the interactive learning guide successfully.
- [x] Completed the Section 01 first-principles teaching exercise.
- [x] Distinguished pretraining, full fine-tuning, and LoRA.
- [x] Distinguished model artifacts from the MLX runtime.
- [x] Understood the untouched baseline and its before/after role.
- [x] Distinguished training loss from held-out exact-label accuracy.
- [x] Distinguished training, validation, and test split responsibilities.
- [x] Created the public README, result narrative, reproduction guide and
  sharing copy.
- [x] Added MIT licensing, citation, contribution and security guidance.
- [x] Added CI and a local public-artifact safety check.
- [x] Confirmed large/generated artifacts and local model paths are ignored.
- [x] Passed Python, JavaScript, JSON, YAML, link and public-repository checks.
- [x] Designed Experiment 006 with a pinned BERT-Large classifier and leakage
  controls.
- [x] Created and installed the isolated PyTorch encoder environment.
- [x] Locked exact encoder package versions.
- [x] Verified PyTorch MPS availability and a real GPU matrix calculation.
- [x] Downloaded and verified the pinned BERT-Large checkpoint.
- [x] Loaded BERT-Large with a new 77-class head and ran a finite MPS pass.
- [x] Prepared raw-text encoder splits and verified zero text overlap.
- [x] Raised max length from 64 to 128 after development-only token inspection.
- [x] Passed the corrected full-backward MPS probe with finite normalized loss.
- [x] Completed five epochs of Experiment 006 training with test still sealed.
- [x] Finalized and roundtrip-verified the selected Experiment 006 model.
- [x] Completed validation-only error and confusion analysis.
- [x] Rejected the checkpoint ensemble using validation-only evidence.
- [x] Installed and locked SentencePiece for a deferred architecture benchmark.
- [x] Reframed Experiment 007 as refinement of the trained BERT model.
- [x] Prepared and verified train-only class weights for Experiment 007.
- [x] Passed the weighted-loss backward probe from the selected BERT model.
- [x] Completed and rejected class-balanced refinement after validation regression.
- [x] Confirmed class balancing produced zero upweighted-class fixes and three harms.
- [x] Designed a single-value label-smoothing refinement from canonical BERT.
- [x] Passed the label-smoothing backward probe from the selected BERT model.
- [x] Completed and rejected label-smoothed refinement after validation regression.
- [x] Confirmed label smoothing fixed zero targeted high-confidence errors.
- [x] Justified a controlled architecture benchmark after same-model refinements.
- [x] Downloaded and verified the pinned DeBERTa-v3-large checkpoint.
- [x] Installed and verified pinned Protobuf for the DeBERTa tokenizer.
- [x] Passed the DeBERTa untrained MPS smoke test.
- [x] Recorded the first DeBERTa numerical-instability probe at `2e-5`.
- [x] Confirmed the instability persists at `2e-6`.
- [x] Localized the first non-finite value to the word-embedding optimizer update.
- [x] Confirmed freezing the word embeddings alone does not prevent the NaN.
- [x] Confirmed all 402 DeBERTa checkpoint tensors are float16.
- [x] Passed the float32 DeBERTa backward probe with all parameters trainable.
- [x] Completed five float32 DeBERTa epochs with the test set sealed.
- [x] Recorded the epoch-5 validation winner at 0.925974.
- [x] Added the missing evaluation metadata block after result capture failed.
- [x] Added and ran the validation-only finalizer.
- [x] Verified the selected DeBERTa model save/reload round trip and checksum.
- [x] Evaluated the finalized DeBERTa model once on the sealed 3,080-row test.
- [x] Recorded 2,896/3,080 test accuracy (0.940260) and 184 errors.
- [x] Created reproducible DeBERTa test error analysis.
- [x] Analyzed DeBERTa validation errors without using the observed test score.
- [x] Prepared Experiment 010 train-only targeted data (11,357 rows).
- [x] Resolved the post-test holdout protocol; Experiment 010 excludes test
  data and uses validation-only selection.
- [x] Passed the Experiment 010 two-update continuation probe.
- [x] Completed one epoch of Experiment 010 targeted continuation.
- [x] Finalized and round-trip verified the Experiment 010 child.
- [x] Analyzed Experiment 010 validation transitions before promotion.
- [x] Promoted Experiment 010 as the validation champion only.
- [x] Prepared Experiment 011 train-only upper-layer refinement data.
- [x] Passed the Experiment 011 finite-loss upper-layer probe.
- [x] Completed Experiment 011 full validation-only continuation.
- [x] Finalized and round-trip verified the Experiment 011 checkpoint.
- [x] Analyzed Experiment 011 validation transitions before promotion.
- [x] Run the final reporting-only Experiment 011 test evaluation.
- [x] Recorded the sealed Exp011 benchmark: 2,899/3,080, 0.941234 accuracy,
  0.941173 macro F1; test result is excluded from future selection.
- [x] Ran the Exp012 five-fold train-only label-quality audit.
- [x] Reviewed Exp012 candidates and confirmed semantic boundary ambiguity;
  no automatic deletion or relabeling was performed.
- [x] Prepared the Exp013 train-only hard-negative records; all original
  labels retained and 162 rival labels activated.
- [x] Passed the Exp013 finite-loss MPS hard-negative probe; no weights saved.
- [x] Completed the full Exp013 validation-only continuation; it regressed to
  0.928571 and is not promoted.
- [x] Finalized and round-trip verified the rejected Exp013 checkpoint.
- [x] Analyzed the Exp013 validation transition; zero fixes and one harm.
- [x] Ran the Exp011 validation error report; 54 errors and 32 high-confidence
  errors were recorded without test access.
- [x] Designed Exp014 as a broader train-only hard-negative coverage test;
  no validation/test rows are used.
- [x] Prepared Exp014 data with 282 active rivals and original labels intact.
- [x] Passed the Exp014 finite-loss broader hard-negative probe; no weights
  saved.
- [x] Completed the full Exp014 validation-only continuation; it regressed to
  0.928571 and is rejected without test access.
- [x] Designed Exp015 as a controlled train-only label-noise filtering test.
- [x] Prepared 8,155 Exp015 training rows after removing 1,078 confidently
  suspicious rows; no labels rewritten and no held-out rows loaded.
- [x] Passed the Exp015 two-update float32 DeBERTa MPS probe; finite loss,
  no saved weights and no held-out rows loaded.
- [x] Completed the full Exp015 validation-only DeBERTa retraining.
- [x] Rejected Exp015 at 0.907792 validation accuracy versus Exp011 at
  0.929870; retained Exp011 and kept test locked.
- [x] Rewrote the public README around the complete three-model result.
- [x] Added beginner, reviewer and reproduction navigation.
- [x] Added an Exp001–015 experiment journey and script map.
- [x] Added compact machine-readable encoder results.
- [x] Added GitHub/X/Hugging Face/DEV presence and launch materials.
- [x] Added a GitHub Pages workflow for the static learning guide.
- [x] Strengthened the public checker for model binaries, access tokens and
  private corporate identity.
- [x] Revalidated the 1.17 MB prospective public source package.
- [x] Authenticate GitHub CLI and create the sanitized public repository.
- [x] Configure repository metadata, Pages and logged-out access.
- [x] Create the public profile README repository.
- [x] Publish the X launch thread after the live guide URL was verified.
- [x] Publish the static learning guide as a Hugging Face Space.
- [ ] Add the optional GitHub profile bio and pin the repository after the
  username rename is settled.

## Experiments

- `baseline-smoke-v1`: complete
- `baseline-balanced-pilot-v1`: complete
- `baseline-full-v1`: complete
- `exp-001-pipeline`: complete, including adapter-loading smoke
- `exp-002-balanced-539`: complete, including pilot evaluation
- `exp-003-balanced-1925`: failed due to numerical instability; diagnosis
  complete; adapter unusable
- `exp-003b-balanced-1925-lr2p5e-6`: stable training complete; pilot
  and full evaluation complete
- `exp-004-full-data-batch7-probe`: complete; not an accuracy experiment
- `exp-004-full-data`: failed early with non-finite loss; no adapter saved
- `exp-004b-full-data-batch1-probe`: complete; stability probe only
- `exp-004c-full-data-batch1-lr5e-7`: stable training complete; pilot
  and full evaluation complete
- `exp-005-attention-qkvo-full`: failed at iteration 2,050 due to numerical
  instability; no adapter saved
- `exp-005b-attention-qkvo-lr2p5e-7`: stable training plus unconstrained,
  batch-matched, and canonical-constrained full evaluations complete;
  constrained inference is the confirmed deployment candidate
- `exp-006-bert-large-classifier`: five-epoch training and validation-only
  selection complete; selected epoch 4 at 0.915584 validation accuracy;
  canonical model roundtrip verified; test remains sealed
- `exp-007-bert-class-balanced-refinement`: complete and rejected at 0.911688
  validation accuracy; test remains sealed
- `exp-008-bert-label-smoothing-refinement`: complete and rejected at 0.912987
  validation accuracy; transition analysis complete; test remains sealed
- `exp-009-deberta-v3-large-classifier`: model-download preparation complete;
  backward probe failed at update 2 with non-finite loss; no weights saved;
  test remains sealed
- `exp-009b-deberta-v3-large-stability`: lower-learning-rate probe prepared;
  training not started; test remains sealed
- `exp-009d-deberta-v3-large-float32`: five-epoch training complete; epoch-5
  validation winner 0.925974; finalized and roundtrip-verified; test evaluation
  complete at 0.940260; targeted refinement decision pending
- `exp-010-deberta-targeted-oversampling`: train-only data prepared (11,357
  rows, 18 labels); probe passed; one-epoch continuation reached 0.928571
  validation accuracy; finalized, roundtrip-verified and promoted as the
  validation champion; test excluded
- `exp-011-deberta-upper-layer-refinement`: train-only upper-layer refinement
  reached 0.929870 validation accuracy; finalized, roundtrip-verified and
  promoted as the validation champion; final reporting-only test reached
  0.941234 accuracy (2,899/3,080); test is now permanently excluded from
  future experiment selection
- `exp-012-label-quality-audit`: five-fold out-of-fold audit over 9,233
  training rows; 1,186 disagreements reported; validation and test excluded;
  candidate review complete; hard-negative refinement planned
- `exp-013-deberta-hard-negative-refinement`: train-only data prepared from
  the validation champion using 162 rival labels; finite-loss probe passed;
  full validation-only continuation completed at 0.928571 and rejected;
  finalized and round-trip verified; transition analysis found zero fixes and
  one harm; closed; test excluded
- `exp-014-deberta-broader-hard-negative`: configuration and guarded
  train-only preparation/probe scripts added; 282 active rivals prepared;
  finite-loss probe passed; full continuation regressed to 0.928571 and was
  rejected; test excluded
- `exp-015-deberta-noise-pruned`: 1,078 train-only rows selected by the
  pre-registered Exp012 disagreement/probability rule were removed without
  relabeling; 8,155 rows and all 77 labels remain; finite-loss MPS probe
  passed; full validation-only retraining pending; test excluded

## Issues and fixes

- Experiment 006 completed training, but Trainer's internal best-checkpoint
  restore bypassed BERT's legacy LayerNorm `gamma/beta` to `weight/bias`
  conversion and warned about 98 mismatched keys. Direct verification proved
  the normal `from_pretrained()` path maps the saved tensors exactly. The
  separate finalizer therefore reloads the preselected epoch-4 checkpoint via
  that supported path, reproduces validation metrics and verifies a saved-model
  round trip before creating the canonical result.
- Experiment 006 packaging originally required one file matching either
  `*.safetensors` or `*.bin`; this accidentally counted both `model.safetensors`
  and `training_args.bin`. The finalizer now accepts only the exact model weight
  filenames.

- The first Experiment 006 backward probe was finite but reported loss 17.65,
  almost exactly four times the expected random 77-class loss. Source inspection
  showed that `BertForSequenceClassification` accepts generic keyword arguments
  but its legacy mean cross-entropy ignores Trainer's `num_items_in_batch`.
  Trainer therefore misdetected the model as accumulation-aware and skipped its
  own division by four. Gradient clipping kept the probe safe, no weights were
  saved, and the corrected probe explicitly sets
  `model_accepts_loss_kwargs = False` as required by Trainer's contract.

- The first Experiment 006 checkpoint download retrieved only the 650 KB
  configuration/tokenizer snapshot and then stopped safely because the pinned
  legacy BERT repository has `pytorch_model.bin`, not `model.safetensors`.
  The downloader now accepts the repository's actual PyTorch weight filename,
  still requires exactly one supported weight artifact, and will reuse the
  files already cached during the retry.

- The first constrained-pilot attempt stopped before producing any prediction
  because MLX 0.32's `ArrayAt` helper has no JAX-style `.set()` method. The
  masking processor now uses MLX indexed replacement, matching the installed
  MLX-LM implementation. The empty predictions checkpoint is safe to resume.
- The second constrained-pilot attempt also stopped before producing a
  prediction because MLX-LM computes one token ahead before recognizing the
  current EOS stop token. The processor now passes through that discarded
  post-EOS calculation instead of treating EOS as an invalid label prefix.

- Direct `sysctl` CPU and memory queries were denied inside the Codex workspace
  sandbox. The same facts were verified through macOS `system_profiler`.
- Inside the Codex shell, Homebrew precedes pyenv shims on `PATH`, so a plain
  `python3` resolved to Python 3.14.4 after entering the repo. The lab virtual
  environment was created explicitly from the installed pyenv Python 3.11.9,
  and project commands will use `.venv/bin/python` to remain deterministic.
- The user's first environment-check run reported `metal: unavailable` even
  though the Apple M2 Max and `arm64` were detected. This was a brittle
  `system_profiler` parsing/checking issue, not evidence that the machine lacks
  Metal. The later MLX compute check proved Metal access.
- MLX cannot access a Metal device from Codex's headless/sandboxed execution
  session. This is expected for that environment, so Step 03 must be run from
  the user's normal Terminal session.
- The DeBERTa validation-error analysis also requires the user's normal
  Terminal: the Codex shell reports MPS unavailable even though the user's
  earlier encoder training and final test evaluation successfully used MPS.
  The script stopped before loading validation data or writing an analysis.
- Holdout protocol correction: Experiment 010 preparation read only the
  original training file and validation-analysis metadata (`test_rows_loaded:
  0`); it did not train on test data. Because the DeBERTa test score was
  already observed, it remains a fixed benchmark and cannot select or
  substantiate this continuation. Experiment 010 proceeds only with the
  existing validation split; no test file is read by its probe or future
  selection scripts.
- The first Step 03 run stopped before touching Metal because the top-level
  `mlx` package does not expose a `__version__` attribute. Version reporting now
  uses Python package metadata instead; the corrected rerun passed.
- MLX 0.32.0 warns that `mx.metal.device_info()` is deprecated. The verification
  script now uses `mx.device_info()`.
- The first BANKING77 mirror inspection observed 9,993 train and 3,076 test
  rows, versus the documented 10,003/3,080. Direct comparison found the mirror
  has no extra records but omits 10 train and 4 test queries from the original.
  The lab switched to the commit-pinned canonical PolyAI source with checksums.
- The first tokenization inspection incorrectly treated the Transformers 5
  `BatchEncoding` result as a token-ID list, so its suffix check compared result
  fields rather than token IDs. The script now requests a plain token-ID list;
  the underlying SFT records and Qwen chat template were correct.
- A source check of MLX-LM 0.31.3 confirmed its chat dataset leaves Qwen's
  training template options at their defaults and masks through the assistant
  generation boundary. The tokenization report was aligned to that exact
  behavior; the maximum full sequence and recommended 576-token limit are
  unchanged.
- After the pipeline adapter was saved, later invocations stopped rather than
  overwriting it. This is the runner's intended artifact-protection behavior,
  not a training failure.
- The first Experiment 004c attempt was interrupted with Control-C during its
  initial validation pass, before any training update or adapter weight was
  produced. MLX-LM had already created an adapter configuration and the runner
  had created its log, so overwrite protection correctly rejected immediate
  reruns. The partial attempt was archived, and the runner now terminates its
  complete MLX process group and archives partial artifacts on Control-C.
- Experiment 009d completed all five epochs and reached 0.925974 validation
  accuracy at epoch 5, but the runner raised `KeyError: 'evaluation'` after
  saving the checkpoints while constructing result metadata. The run did not
  lose weights or touch the sealed test set. The missing config block was
  added and `finalize_exp_009d.py` now performs validation-only selection,
  supported-path reload and exact prediction round-trip checks.

## Results

- MLX version: 0.32.0
- Metal available: true
- Default MLX device: GPU 0
- MLX device: Apple M2 Max (`applegpu_g14s`)
- Reported Metal memory: 32 GiB
- Recommended working-set limit: approximately 24.96 GiB
- Test matrix product: `[[19.0, 22.0], [43.0, 50.0]]`
- Test calculation correct: true
- BANKING77 source revision:
  `57ec275d8078af65b7731c2a98be812d844a6d6b`
- BANKING77 train rows: 10,003, all text-unique
- BANKING77 test rows: 3,080, all text-unique
- BANKING77 intent labels: 77
- BANKING77 train/test text overlap: 0
- BANKING77 source checksums: verified
- Model revision:
  `720c04346ea2b095c801ebbd545c109230964cd4`
- Model weight file: `model.safetensors`, 3.205 GiB
- Model architecture: `Qwen3ForCausalLM`
- Model precision: BF16, unquantized
- Smoke-test prompt size before the example response: 441 tokens
- Smoke prediction: `lost_or_stolen_card`
- Smoke expected label: `card_arrival`
- Smoke output was a valid label: true
- Smoke prediction was correct: false
- Smoke peak memory: 3.803 GB
- Smoke prompt processing: 246.730 tokens/second
- Smoke generation: 22.843 tokens/second
- Pilot selection seed: 3407
- Pilot examples: 154, with 2 from every one of the 77 intents
- Pilot checksum:
  `9b1381574ec9783e6075a2cd8f79bd6e4b936e668c3cfc1bc6e54aa447ea5000`
- Pilot correct predictions: 62 of 154
- Pilot accuracy: 0.402597
- Pilot macro F1: 0.344888
- Pilot invalid labels: 6 of 154
- Pilot invalid-label rate: 0.038961
- Pilot peak memory: 4.574 GB
- Pilot elapsed time: 40.889 seconds
- Pilot predictions checksum:
  `bbb497c31ffbebc1714c3b6483a7496f22a76b66e0aa064ca2319736a52d884f`
- Full baseline correct predictions: 1,474 of 3,080
- Full baseline accuracy: 0.478571
- Full baseline macro F1: 0.468243
- Full baseline invalid labels: 176 of 3,080
- Full baseline invalid-label rate: 0.057143
- Full baseline peak memory: 4.773 GB
- Full baseline elapsed time: 808.261 seconds
- Full baseline predictions checksum:
  `964c890ef7bad615f7037c2f5f79a1670463276cd1043aa94c9a93fa39b87681`
- SFT train rows: 9,233
- SFT validation rows: 770, exactly 10 per intent
- SFT train/validation text overlap: 0
- SFT train checksum:
  `65f0b2d327a48c64532cabfbd2622a4aef34ebbcf170add9d8664b8fa645c959`
- SFT validation checksum:
  `2c59614c9497d09e3eaddf1f7294f1cd6e5528463f33b669c52140d6ee5e2338`
- SFT full-sequence token length: minimum 440, median 451, p95 475,
  p99 490, maximum 537
- SFT sequences longer than 512 tokens: 11
- SFT sequences longer than 576 tokens: 0
- MLX-LM masked assistant completion: 8 to 15 tokens
- Selected maximum sequence length: 576, with no truncation
- Pipeline LoRA trainable parameters: 0.918 million of 1,720.575 million
  (`0.053%`)
- Pipeline validation loss: 7.345 before training and 0.751 at iteration 10,
  each measured on only 5 validation batches
- Pipeline final reported train loss: 0.825
- Pipeline peak memory: 4.852 GB
- Pipeline elapsed time: 14.361 seconds
- Pipeline adapter size: 3,676,917 bytes
- Pipeline adapter checksum:
  `647187eccde4d6e872f04bc66358ee18a84a891627ee6f26d73944f819c07361`
- Pipeline adapter smoke prediction: `card_delivery_estimate`
- Untouched smoke prediction on the same request: `lost_or_stolen_card`
- Expected smoke label: `card_arrival`
- Pipeline adapter smoke output was a valid BANKING77 label: true
- Pipeline adapter smoke prediction was correct: false
- Pipeline adapter smoke peak memory: 3.994 GB
- Experiment 002 selection seed: 3408
- Experiment 002 training rows: 539, exactly 7 per intent
- Experiment 002 validation rows: 154, exactly 2 per intent
- Experiment 002 train/validation text overlap: 0
- Experiment 002 training checksum:
  `5234027d1cbe8a06a3e4545f810837eebe489b2efa78e8ff446f56e77775b701`
- Experiment 002 validation checksum:
  `4e5fca4c4b39c129f925d5f2957745081f6b81b38fd65a2473a4ce97b6b0d72d`
- Experiment 002 validation loss: 7.011 before training, 0.268 at iteration
  270, and 0.277 at iteration 539
- Experiment 002 final reported train loss: 0.253
- Experiment 002 peak memory: 4.968 GB
- Experiment 002 elapsed time: 464.347 seconds
- Experiment 002 adapter checksum:
  `cd2111999465a49d060f512d3eefd01534d940dba8a817ece826d53a2faaf764`
- Experiment 002 pilot correct predictions: 61 of 154
- Experiment 002 pilot accuracy: 0.396104 versus 0.402597 untouched
  (`-0.006494`, one fewer correct example)
- Experiment 002 pilot macro F1: 0.368099 versus 0.344888 untouched
  (`+0.023212`)
- Experiment 002 pilot invalid labels: 4 of 154 versus 6 untouched
- Experiment 002 pilot invalid-label rate: 0.025974 versus 0.038961 untouched
- Experiment 002 pilot transitions: 16 wrong-to-correct, 17
  correct-to-wrong, 45 correct-to-correct, and 76 wrong-to-wrong
- Experiment 002 changed 84 of the 154 pilot predictions
- Experiment 003 training rows: 1,925, exactly 25 per intent
- Experiment 003 includes all 539 Experiment 002 training rows
- Experiment 003 validation rows: the same 154 records as Experiment 002
- Experiment 003 train/validation text overlap: 0
- Experiment 003 training checksum:
  `9277ac00d508719cfc7d3bd72c78ca10f0aac7b9a0c3e464932bb0fb75f70aba`
- Experiment 003 validation checksum:
  `4e5fca4c4b39c129f925d5f2957745081f6b81b38fd65a2473a4ce97b6b0d72d`
- Experiment 003 high-learning-rate run: loss rose from 0.285 at iteration 100
  to 12.259 at iteration 900, then was `NaN` at the iteration-950 report
- Experiment 003 validation loss: 7.011 initially, then `NaN` at iterations
  963 and 1,925
- Experiment 003 failed-run peak memory: 5.029 GB, ruling out unified-memory
  exhaustion as the cause
- Experiment 003 failed adapter: all 64 tensors and all 917,504 values are
  non-finite; the artifact must not be used
- Experiment 003b validation loss: 7.011 initially, 0.396 at iteration 963,
  and 0.375 at iteration 1,925
- Experiment 003b final reported train loss: 0.236
- Experiment 003b peak memory: 5.031 GB
- Experiment 003b elapsed time: 1,239.901 seconds
- Experiment 003b adapter: all 64 tensors and 917,504 values are finite
- Experiment 003b adapter checksum:
  `5357a9c472a24ede76497653d12bb338ddf3795817e1be06a5b98e01068b3983`
- Experiment 003b pilot correct predictions: 63 of 154
- Experiment 003b pilot accuracy: 0.409091 versus 0.402597 untouched and
  0.396104 for Experiment 002
- Experiment 003b pilot macro F1: 0.384519 versus 0.344888 untouched and
  0.368099 for Experiment 002
- Experiment 003b pilot invalid labels: 11 of 154
- Experiment 003b pilot invalid-label rate: 0.071429
- Experiment 003b versus untouched pilot transitions: 14 wrong-to-correct,
  13 correct-to-wrong, 49 correct-to-correct, and 78 wrong-to-wrong
- Experiment 003b full-test correct predictions: 1,459 of 3,080
- Experiment 003b full-test accuracy: 0.473701 versus 0.478571 untouched
  (`-0.004870`, 15 fewer correct answers)
- Experiment 003b full-test macro F1: 0.479970 versus 0.468243 untouched
  (`+0.011727`)
- Experiment 003b full-test invalid labels: 177 versus 176 untouched
- Experiment 003b full-test transitions: 270 wrong-to-correct, 285
  correct-to-wrong, 1,189 correct-to-correct, and 1,336 wrong-to-wrong
- Experiment 003b changed 1,338 of 3,080 full-test predictions
- Largest per-intent gains included `getting_virtual_card` (+24 correct),
  `card_payment_fee_charged` (+22), and `unable_to_verify_identity` (+21)
- Largest per-intent losses included `verify_my_identity` (-15 correct),
  `verify_top_up` (-13), and `transaction_charged_twice` (-12)
- Experiment 004 probe batch size: 7
- Experiment 004 probe validation loss: 6.811 initially and 5.456 at
  iteration 10
- Experiment 004 probe final reported train loss: 5.228
- Experiment 004 probe peak memory: 12.534 GB
- Experiment 004 probe adapter: all 917,504 values are finite
- Experiment 004 full run initial validation loss: 6.956
- Experiment 004 full run first training report: `NaN` at iteration 50
- Experiment 004 full run peak memory: 12.556 GB
- Experiment 004 safety guard stopped training before an adapter was saved
- Experiment 004b probe completed all 50 batch-size-1 updates at `5e-7`
- Experiment 004b validation loss: 9.705 initially and 6.988 at iteration 50;
  each measurement used only one validation record and is a stability signal,
  not an accuracy result
- Experiment 004b final reported train loss: 7.427
- Experiment 004b peak memory: 4.951 GB
- Experiment 004b adapter: all 917,504 values are finite
- Experiment 004b adapter checksum:
  `a636e8ddafbf01d085503663a4ab71360f7e76567bcfff2cdeddf94b38ee48cb`
- Experiment 004c validation loss: 6.959 initially, 0.328 at iteration 4,617,
  and 0.298 at iteration 9,233
- Experiment 004c final reported train loss: 0.278
- Experiment 004c peak memory: 5.031 GB
- Experiment 004c elapsed time: 5,216.728 seconds
- Experiment 004c adapter: all 917,504 values are finite
- Experiment 004c adapter checksum:
  `f6527a0dd0c554d878dc6af3d8b82bfac22edf957f219fc359eccd1688a41059`
- Experiment 004c pilot correct predictions: 69 of 154
- Experiment 004c pilot accuracy: 0.448052 versus 0.402597 untouched
  (`+0.045455`, seven more correct examples)
- Experiment 004c pilot macro F1: 0.417256 versus 0.344888 untouched
  (`+0.072368`)
- Experiment 004c pilot invalid labels: 7 of 154 versus 6 untouched
- Experiment 004c pilot accuracy improved by 0.038961 over Experiment 003b
- Experiment 004c full-test correct predictions: 1,522 of 3,080
- Experiment 004c full-test accuracy: 0.494156 versus 0.478571 untouched
  (`+0.015584`, 48 more net-correct answers)
- Experiment 004c full-test macro F1: 0.497983 versus 0.468243 untouched
  (`+0.029740`)
- Experiment 004c full-test invalid labels: 159 versus 176 untouched
- Experiment 004c full-test accuracy improved by 0.020455 over Experiment 003b
- Experiment 004c versus untouched transitions: 238 wrong-to-correct, 190
  correct-to-wrong, 1,284 correct-to-correct, and 1,368 wrong-to-wrong
- Exact paired McNemar two-sided p-value: 0.022985
- Largest net intent gain versus untouched:
  `card_payment_fee_charged` (+29 correct)
- Largest net intent loss versus untouched:
  `why_verify_identity` (-15 correct)
- Interactive request: `The cash machine charged me an extra fee.`
- Interactive prediction: `cash_withdrawal_charge` (valid and correct)
- Local API ready at `http://127.0.0.1:8080`
- Local API response prediction: `cash_withdrawal_charge`
- Local API response adapter checksum:
  `f6527a0dd0c554d878dc6af3d8b82bfac22edf957f219fc359eccd1688a41059`
- Corrected local API requests: 2 successful HTTP 200 responses
- Local API shutdown: clean
- Experiment 005 probe validation loss: 9.705 initially and 5.225 at
  iteration 50; each used one validation record
- Experiment 005 probe final reported train loss: 5.203
- Experiment 005 probe peak memory: 5.058 GB
- Experiment 005 probe adapter: all 1,835,008 values are finite
- Experiment 005 probe adapter checksum:
  `1780ba08774b2293dc619cc7d4ebb07e9ebe3e511ed27bda8b01e8e25e9b20a2`
- Experiment 005 full run last finite report: loss 0.228 at iteration 2,000
- Experiment 005 full run first non-finite report: `NaN` at iteration 2,050
- Experiment 005 full run peak memory: 5.165 GB
- Experiment 005 full run saved no adapter weights and is unusable
- Experiment 005b validation loss: 6.959 initially, 0.498 at iteration 4,617,
  and 0.494 at iteration 9,233
- Experiment 005b final reported train loss: 0.501
- Experiment 005b peak memory: 5.180 GB
- Experiment 005b elapsed time: 5,768.549 seconds
- Experiment 005b adapter: all 1,835,008 values are finite
- Experiment 005b adapter checksum:
  `b6454e88a364ddae305243618d52d23ac83da5273c5a347707dda52882c8a7b1`
- Experiment 005b pilot correct predictions: 68 of 154
- Experiment 005b pilot accuracy: 0.441558 versus 0.402597 untouched
  (`+0.038961`) and 0.448052 for Experiment 004c (`-0.006494`)
- Experiment 005b pilot macro F1: 0.401781 versus 0.344888 untouched
  (`+0.056893`) and 0.417256 for Experiment 004c (`-0.015475`)
- Experiment 005b pilot invalid labels: 7 of 154, equal to Experiment 004c
- Experiment 005b versus Experiment 004c pilot transitions: 63 both correct,
  6 correct only for 004c, 5 correct only for 005b, and 80 both wrong
- Experiment 005b full-test correct predictions: 1,552 of 3,080
- Experiment 005b full-test accuracy: 0.503896 versus 0.478571 untouched
  (`+0.025325`, 78 more correct) and 0.494156 for Experiment 004c
  (`+0.009740`, 30 more correct)
- Experiment 005b full-test macro F1: 0.499911 versus 0.468243 untouched
  (`+0.031669`) and 0.497983 for Experiment 004c (`+0.001929`)
- Experiment 005b full-test invalid labels: 173 versus 176 untouched and 159
  for Experiment 004c
- Experiment 005b versus Experiment 004c full-test transitions: 1,398 both
  correct, 154 correct only for 005b, 124 correct only for 004c, and 1,404
  both wrong
- Experiment 005b versus Experiment 004c exact paired McNemar two-sided
  p-value: 0.081795
- Experiment 005b and Experiment 004c changed predictions on 664 of 3,080
  examples
- All 173 Experiment 005b invalid outputs were non-empty. The most common were
  plausible taxonomy aliases: `get_virtual_card` (32), `top_up_by_card` (31),
  `refund_not_showing_up` (29), `getting_physical_card` (14), and
  `declined_direct_debit_payment` (13).
- Constrained Experiment 005b pilot correct predictions: 69 of 154 versus 68
  unconstrained (`+0.006494` accuracy)
- Constrained Experiment 005b pilot macro F1: 0.397626 versus 0.401781
  unconstrained (`-0.004155`)
- Constrained Experiment 005b pilot invalid labels: 0 versus 7 unconstrained
- Constrained inference changed 12 of 154 pilot predictions. It preserved all
  68 unconstrained correct answers and repaired one invalid output to the
  correct label; six other invalid outputs became valid but stayed incorrect.
- Constrained Experiment 005b pilot elapsed time: 55.083 seconds
- Constrained Experiment 005b full-test correct predictions: 1,574 of 3,080
- Constrained Experiment 005b full-test accuracy: 0.511039 versus 0.503896
  unconstrained (`+0.007143`, 22 more correct)
- Constrained Experiment 005b full-test macro F1: 0.496203 versus 0.499911
  unconstrained (`-0.003708`)
- Constrained Experiment 005b full-test invalid labels: 0 versus 173
  unconstrained
- Constrained versus unconstrained 005b transitions: 1,534 both correct, 40
  correct only when constrained, 18 correct only when unconstrained, and 1,488
  both wrong; exact paired McNemar two-sided p-value 0.005355
- Constrained versus unconstrained 005b changed 248 predictions. Because these
  evaluations used batch sizes 1 and 8 respectively, the comparison is
  confounded until a batch-size-1 unconstrained control is available.
- Constrained Experiment 005b full-test elapsed time: 1,012.428 seconds
- Unconstrained batch-size-1 005b pilot correct predictions: 67 of 154 versus
  69 constrained
- Unconstrained batch-size-1 005b pilot macro F1: 0.390267 versus 0.397626
  constrained
- Matched pilot transitions: 67 both correct, 2 correct only when constrained,
  0 correct only when unconstrained, and 85 both wrong
- Matched pilot constraint changes: seven predictions, exactly the seven
  unconstrained invalid outputs; two became correct and five stayed wrong
- Unconstrained batch-size-1 pilot elapsed time: 51.379 seconds
- Unconstrained batch-size-1 005b full-test correct predictions: 1,545 of
  3,080
- Unconstrained batch-size-1 005b full-test accuracy: 0.501623
- Unconstrained batch-size-1 005b full-test macro F1: 0.499571
- Unconstrained batch-size-1 005b full-test invalid labels: 181 of 3,080
  (0.058766)
- Unconstrained batch-size-1 005b full-test elapsed time: 1,035.906 seconds
- Matched full-test constrained accuracy: 0.511039, an increase of 0.009416
  and 29 correct answers over unconstrained batch-size-1 inference
- Matched full-test constrained macro F1: 0.496203 versus 0.499571
  unconstrained (`-0.003368`)
- Matched full-test transitions: 1,545 both correct, 29 correct only when
  constrained, zero correct only when unconstrained, and 1,506 both wrong
- The matched constraint changed exactly 181 predictions, all of which were
  invalid unconstrained outputs. It preserved all 2,899 valid predictions.
- Of the 181 invalid unconstrained outputs, 29 became correct canonical labels
  and 152 became valid but remained wrong.
- Matched full-test exact paired McNemar two-sided p-value:
  `3.72529029846e-09`
- Decision: use Experiment 005b with canonical-label constrained decoding as
  the default classifier. It has the best matched accuracy and guarantees the
  77-label output contract.
- Promoted API request: `The cash machine charged me an extra fee.`
- Promoted API prediction: `cash_withdrawal_charge`
- Promoted API valid canonical label: true
- Promoted API constraint mode: `canonical_labels`
- Promoted API adapter checksum:
  `b6454e88a364ddae305243618d52d23ac83da5273c5a347707dda52882c8a7b1`
- Promoted API end-to-end status: passed
- Promoted API shutdown status: clean (`local_api_stopped: True`)
- First end-to-end training and deployment lifecycle: complete

### DeBERTa-v3-large benchmark

- Experiment 009d model revision:
  `64a8c8eab3e352a784c658aef62be1662607476f`
- Experiment 009d precision: float32 on MPS
- Experiment 009d training rows: 9,233
- Experiment 009d validation rows: 770
- Experiment 009d epochs: 5
- Experiment 009d learning rate: `2e-5`
- Experiment 009d validation accuracy by epoch: 0.674026, 0.855844,
  0.902597, 0.916883, 0.925974
- Experiment 009d validation macro F1 at epoch 5: 0.9261
- Experiment 009d best checkpoint: epoch 5 (`checkpoint-1445`)
- Experiment 009d strict improvement versus BERT epoch-4 champion:
  `+0.010390` (8 additional correct answers)
- Experiment 009d test evaluation: complete once, after validation selection
- Experiment 009d test rows: 3,080
- Experiment 009d test loss: 0.281448
- Experiment 009d test accuracy: 0.940260 (2,896/3,080)
- Experiment 009d test macro F1: 0.940117
- Experiment 009d validation-to-test accuracy delta: `+0.014286`
- Experiment 009d 95% target: 2,926/3,080; 30 additional correct answers
  would be required
- Experiment 009d largest test gaps: `pending_transfer` 29/40,
  `declined_transfer` 31/40, `topping_up_by_card` 32/40
- Experiment 009d largest test confusion:
  `pending_transfer -> transfer_not_received_by_recipient` (7)
- Experiment 009d test result and error analysis are recorded; test metrics
  must not be used to select future refinements
- Experiment 009d validation error analysis: 713/770 correct, 57 errors
- Experiment 009d weakest validation labels: `topping_up_by_card` 6/10,
  `contactless_not_working` 7/10, `pending_top_up` 7/10
- Experiment 009d leading validation confusions:
  `card_payment_wrong_exchange_rate -> exchange_charge` (2),
  `topping_up_by_card -> top_up_reverted` (2),
  `declined_transfer -> failed_transfer` (2)
- Experiment 009d selected validation loss: 0.350554
- Experiment 009d selected validation macro F1: 0.926117
- Experiment 009d selected model weight SHA-256:
  `59b0d419bd5c6d992123a08ba880829cd6585a99b3f598d4d3a5a884d052a8d6`
- Experiment 009d finalization: complete; exact save/reload round trip verified
- Experiment 009d sealed-test evaluation: complete; future selection remains
  validation-only
- Experiment 010 targeted data: 11,357 train rows, 18 labels, 2,124 added
  duplicates; train SHA-256
  `e981ec0a01169b53def3779027d74936fab19574d03d42e134af57620d3735f7`
- Experiment 010 continuation probe: loss 0.025904, 14.148 GiB peak,
  validation/test rows loaded 0, weights saved false
- Experiment 010 full continuation: complete for one epoch; test access
  prohibited
- Experiment 010 one-epoch continuation: train loss 0.079788, validation loss
  0.359768, validation accuracy 0.928571, validation macro F1 0.928234
- Experiment 010 strict validation improvement versus Experiment 009d:
  `+0.002597` (2 additional correct validation predictions)
- Experiment 010 continuation peak memory: 16.898 GiB; elapsed 986.654 seconds
- Experiment 010 training result: test rows loaded 0; test evaluated false
- Experiment 010 finalization: exact save/reload validation round trip passed;
  selected weight SHA-256
  `ffb1d52259ac2d3f4f0c65c29de5b369595f3fcdc157ad3d9f3eb43cd6b85a82`
- Experiment 010 validation transitions: 5 child-only correct, 3
  parent-only correct, 710 both correct, 52 both wrong, 9 changed
  predictions; test access prohibited
- Experiment 010 targeted weak labels unchanged: `topping_up_by_card` 6/10,
  `contactless_not_working` 7/10, `pending_top_up` 7/10
- Experiment 010 validation champion promotion: accepted for development only;
  its test performance is unknown and must not be inferred from Experiment
  009d's 0.940260 benchmark

## Experiment design notes

- A proposed 26-per-intent Experiment 003 would have required 2,002 records,
  but the smallest intent has only 25 available training records after the
  validation holdout. Experiment 003 therefore uses 25 per intent (1,925 total)
  to remain balanced, unique, and duplication-free.
