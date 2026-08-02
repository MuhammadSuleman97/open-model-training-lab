# Experiment 009 — DeBERTa-v3-large architecture benchmark

## Why a new architecture is now justified

Experiment 006's BERT-Large model remains the canonical champion at 91.5584%
validation accuracy. We first attempted to improve that exact trained model:

- epoch-4/epoch-5 probability ensembling scored 91.4286%;
- class-balanced continuation scored 91.1688%;
- label-smoothed continuation scored 91.2987%.

Those controlled failures show that the tested post-training refinements do not
close the 27-answer validation gap to 95%. Experiment 009 therefore changes the
encoder architecture while holding the task, dataset splits, output labels and
evaluation policy constant.

## Fair-comparison rules

- Candidate: `microsoft/deberta-v3-large` at an exact pinned revision.
- Licence: MIT.
- Training: the same 9,233 BANKING77 training records.
- Selection: the same 770-record validation split.
- Benchmark: strictly beat Experiment 006's 91.5584% validation accuracy.
- Program target: at least 95% validation accuracy.
- Test: keep all 3,080 records sealed during development.
- Experiment 006 remains canonical unless the candidate wins.

Downloading a pretrained encoder is not claiming that our BERT work failed.
It creates a controlled architecture comparison after same-model optimization
was properly attempted and measured.

The untrained 77-class head smoke test passed on MPS with finite `[1, 77]`
logits and a 1.040 GiB peak driver allocation. The classifier and pooler
weights were intentionally newly initialized; the pretrained encoder loaded.

## Stability finding

The first two-update backward probe used the BERT benchmark's learning rate
`2e-5`. Its first loss was `4.395`, but the second update became `NaN`; no
weights were saved. This is preserved as a failed stability probe. Experiment
009b retried the identical probe at `2e-6`, but the second update also became
`NaN`. Before trying another hyperparameter, the next diagnosis instruments the
first bad gradient, parameter or forward-loss location.

The diagnosis found all forward losses finite and the first non-finite values
immediately after optimizer step 1 in
`deberta.embeddings.word_embeddings.weight`. Experiment 009c therefore freezes
only that pretrained lexical embedding matrix as a targeted stability probe;
the rest of DeBERTa and the new classifier head remain trainable.
That probe also became `NaN` on update 2. The next instrumented run keeps the
same freeze and identifies the next first-bad stage before we choose a fix.

The second diagnosis moved the first bad tensor to
`deberta.embeddings.LayerNorm.weight`. Direct checkpoint inspection found all
402 pretrained tensors are `float16`, identifying half-precision AdamW updates
on MPS as the common cause. Experiment 009d casts the model to float32,
unfreezes all parameters and repeats the two-update probe at `2e-5`.

The float32 probe passed with finite losses `4.395` and `4.311`, confirming the
precision fix. Full five-epoch training is now authorized with validation-only
selection and a non-finite-loss safety stop.

## Training result

The full float32 run completed all five epochs without a non-finite loss. The
validation trajectory was:

| Epoch | Validation accuracy |
|---:|---:|
| 1 | 0.674026 |
| 2 | 0.855844 |
| 3 | 0.902597 |
| 4 | 0.916883 |
| 5 | **0.925974** |

Epoch 5 is the validation winner: 713/770 correct, 8 more than the BERT
champion's 705/770, a strict improvement of `+0.010390`. It is still below the
program launch target of 95%, and the 3,080-record test split remains sealed.

The training wrapper raised `KeyError: 'evaluation'` only while writing its
result metadata after the checkpoints had already been saved. The missing
configuration block was repaired. `scripts/finalize_exp_009d.py` then selected
epoch 5, reproduced validation loss `0.350554`, accuracy `0.925974` and macro
F1 `0.926117`, saved the model through the supported Transformers loading
path, reloaded it, and verified exact validation predictions. The selected
weight SHA-256 is
`59b0d419bd5c6d992123a08ba880829cd6585a99b3f598d4d3a5a884d052a8d6`.
No retraining is required.

The one-time sealed-test evaluation uses `scripts/evaluate_exp_009d_test.py`.
It verifies the 3,080-row test checksum before loading any test text, refuses
to overwrite its result, and records aggregate and per-intent metrics.

## Final test result

The one-time test evaluation completed successfully:

- Test loss: `0.281448`
- Test accuracy: `0.940260` (2,896/3,080)
- Test macro F1: `0.940117`
- Validation-to-test accuracy delta: `+0.014286`
- 95% target: 2,926/3,080, so 30 more correct predictions are needed

The largest observed test gaps were `pending_transfer` (29/40),
`declined_transfer` (31/40), and `topping_up_by_card` (32/40). The largest
confusion was `pending_transfer` → `transfer_not_received_by_recipient` (7).
These observations are recorded in `test_error_analysis.json`; they are not
selection data for future training. The next analysis must use validation only.

## Validation-only error analysis

The selected model was re-evaluated on the 770-row validation split without
loading test data. It reproduced 713 correct predictions and 57 errors
(accuracy `0.925974`, macro F1 `0.926117`). The weakest labels were:

- `topping_up_by_card`: 6/10
- `contactless_not_working`: 7/10
- `pending_top_up`: 7/10

The leading validation confusions were
`card_payment_wrong_exchange_rate` → `exchange_charge` (2),
`topping_up_by_card` → `top_up_reverted` (2), and
`declined_transfer` → `failed_transfer` (2). These validation-only findings
justify preparing a targeted train-only refinement, but they do not justify
using the observed test score to choose hyperparameters.

## Next experiment: train-only targeted oversampling

Experiment 010 will begin with a data-preparation manifest, not a long
training run. It selects the eight weakest validation labels plus the expected
and predicted labels from the seven most common validation confusion pairs.
Every selected row comes from the original 9,233-record training split; each
selected row is duplicated once, while the validation split remains unchanged.
There is no synthetic text and no test access. The first command is:

```bash
.venv-encoder/bin/python scripts/prepare_exp_010_targeted_data.py
```

After the manifest is checked, we will run a small numerical probe, then a
low-learning-rate continuation from the finalized DeBERTa model. Promotion
will require a strict validation improvement; the observed 0.940260 test score
will remain a fixed comparison only.

The two-update Experiment 010 continuation probe passed with training loss
`0.025904` and a 14.148 GiB MPS peak. It loaded 11,357 train rows, loaded no
validation or test rows, and saved no weights. The next command is the
validation-only one-epoch continuation; do not evaluate the test set again.

That one-epoch continuation completed without numerical instability. It
reached validation loss `0.359768`, accuracy `0.928571` and macro F1 `0.928234`,
an improvement of two correct validation predictions over Experiment 009d.
The run loaded zero test rows. `scripts/finalize_exp_010.py` must now verify
the child checkpoint and save/reload round trip before any promotion decision.
That finalization passed; the selected child weight SHA-256 is
`ffb1d52259ac2d3f4f0c65c29de5b369595f3fcdc157ad3d9f3eb43cd6b85a82`.
The next validation-only step compares parent/child transitions to identify
fixes and harms. No test file is read.

The transition analysis completed with 5 child-only correct predictions, 3
parent-only correct predictions, 710 unchanged correct predictions, 52 cases
wrong in both models, and only 9 changed predictions. Experiment 010 is
accepted as the validation champion for development. Its targeted weak labels
did not change (`topping_up_by_card` 6/10, `contactless_not_working` 7/10,
`pending_top_up` 7/10), so the oversampling mechanism did not solve its primary
target. The observed Experiment 009d test score remains unrelated and is not
assigned to Experiment 010.

## Holdout pause

The Experiment 010 manifest confirmed `test_rows_loaded: 0`; no test examples
were copied into its training data, and no Experiment 010 training has run.
However, the test score was already observed before this proposed continuation.
Therefore the current test set cannot be used to select or substantiate a later
95% claim. Experiment 010 proceeds only with the existing validation split; its
probe and selection scripts do not read any test file. The 0.940260 result
remains a fixed benchmark for comparison.

## Download result

The pinned 24-layer base checkpoint downloaded successfully. Its `0.814 GiB`
weight file has SHA-256
`dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5`.

## Tokenizer compatibility issue

The first smoke attempt stopped before model loading because the isolated
encoder environment had SentencePiece but lacked Protobuf, which Transformers
needs to extract this checkpoint's tokenizer. The TikToken message was a
secondary fallback failure. Protobuf `7.35.1` is now pinned as an explicit
encoder-environment dependency and has been installed successfully; the
downloaded checkpoint remains valid.
