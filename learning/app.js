const models = [
  {
    id: "base",
    tab: "Untouched Qwen",
    kicker: "Control · no training",
    name: "The baseline",
    story: "We measured the original model before touching any weights. This gives every later result a trustworthy reference point.",
    decision: "Keep as the scientific control.",
    accuracy: 47.8571,
    f1: 46.8243,
    invalid: 176,
    facts: [["Correct", "1,474 / 3,080"], ["Trainable values", "0"], ["Peak memory", "4.773 GB"], ["Role", "Control"]],
  },
  {
    id: "004c",
    tab: "004c · q/v",
    kicker: "Full data · stable LoRA",
    name: "The efficient adapter",
    story: "One epoch over 9,233 records with LoRA on q and v projections. Stable, compact, and our first statistically supported improvement over the base model.",
    decision: "Retain as the smaller, format-steady candidate.",
    accuracy: 49.4156,
    f1: 49.7983,
    invalid: 159,
    facts: [["Correct", "1,522 / 3,080"], ["Trainable values", "917,504"], ["Learning rate", "5e-7"], ["Adapter", "3.68 MB"]],
  },
  {
    id: "005b",
    tab: "005b · q/k/v/o",
    kicker: "Larger adapter · safer LR",
    name: "The accuracy leader",
    story: "We doubled the adapter capacity by targeting q, k, v, and o. The original learning rate became unstable, so the stable retry used half the rate.",
    decision: "Best unconstrained full-test accuracy observed.",
    accuracy: 50.3896,
    f1: 49.9911,
    invalid: 173,
    facts: [["Correct", "1,552 / 3,080"], ["Trainable values", "1,835,008"], ["Learning rate", "2.5e-7"], ["Adapter", "7.35 MB"]],
  },
  {
    id: "constrained",
    tab: "005b + guardrails",
    kicker: "Same weights · controlled decoding",
    name: "The canonical decoder",
    story: "No retraining. At each token, invalid label paths are masked. The model must finish as one of the 77 published labels.",
    decision: "Confirmed default: best matched accuracy and zero invalid labels.",
    accuracy: 51.1039,
    f1: 49.6203,
    invalid: 0,
    facts: [["Correct", "1,574 / 3,080"], ["Invalid labels", "0"], ["Training", "None added"], ["Inference batch", "1"]],
  },
  {
    id: "bert006",
    tab: "BERT · 006",
    kicker: "Encoder pivot · full fine-tune",
    name: "The architecture breakthrough",
    scope: "Validation set · 770 questions",
    story: "BERT-Large reads the whole request and sends one representation into a 77-class head. Matching the architecture to classification moved us from roughly 51% to 91.56%.",
    decision: "Retained after class weighting, label smoothing, and checkpoint ensembling all failed to improve it.",
    accuracy: 91.5584,
    f1: 91.4837,
    invalid: 0,
    benchmark: 91.5584,
    accuracyNote: "705 of 770 validation questions correct.",
    invalidNote: "A 77-class head always returns one of 77 positions.",
    facts: [["Training", "Full model"], ["Selected epoch", "4 of 5"], ["Peak memory", "10.321 GiB probe"], ["Test use", "None"]],
  },
  {
    id: "deberta009d",
    tab: "DeBERTa · 009d",
    kicker: "Stronger encoder · float32",
    name: "The stable DeBERTa benchmark",
    scope: "Validation selected · test reported once",
    story: "DeBERTa-v3-large improved the encoder design, but its half-precision checkpoint became NaN on MPS. Casting every parameter to float32 made five-epoch full fine-tuning stable.",
    decision: "Promoted over BERT using validation only; its one-time test report reached 94.03%.",
    accuracy: 92.5974,
    f1: 92.6117,
    invalid: 0,
    benchmark: 91.5584,
    accuracyNote: "713 of 770 validation questions correct; 94.03% on the fixed test report.",
    invalidNote: "Classification head: no free-form label generation.",
    facts: [["Training", "Full model"], ["Precision", "float32"], ["Selected epoch", "5 of 5"], ["Test report", "94.026%"]],
  },
  {
    id: "deberta011",
    tab: "DeBERTa · 011",
    kicker: "Upper-layer refinement · champion",
    name: "The current validated champion",
    scope: "Validation champion · test is locked",
    story: "Starting from Exp010, we froze layers 0–19 and trained only layers 20–23 plus the classification components. This made one additional validation answer correct without harming another.",
    decision: "Current champion: 92.99% validation and a reporting-only 94.12% test result.",
    accuracy: 92.9870,
    f1: 92.9701,
    invalid: 0,
    benchmark: 92.5974,
    accuracyNote: "716 of 770 validation questions correct.",
    invalidNote: "The final test is locked and cannot choose later experiments.",
    facts: [["Trainable", "51.5M"], ["Frozen", "383.6M"], ["Validation", "92.987%"], ["Test report", "94.123%"]],
  },
  {
    id: "deberta015",
    tab: "Exp015 · rejected",
    kicker: "Data-quality experiment · completed",
    name: "The over-pruned retraining",
    scope: "Validation result · test never loaded",
    story: "A five-fold out-of-fold audit identified suspicious train labels. Exp015 removed 1,078 high-confidence disagreements, kept every original label, and retrained DeBERTa from its original checkpoint for five stable epochs.",
    decision: "Rejected: 90.78% validation did not beat Exp011's fixed 92.99%. Exp011 remains champion and the test stays locked.",
    accuracy: 90.7792,
    f1: 90.7391,
    invalid: 0,
    benchmark: 92.9870,
    accuracyNote: "699 of 770 correct—17 fewer than Exp011's 716.",
    invalidNote: "No rows were relabelled and no test rows were loaded.",
    facts: [["Removed", "1,078 rows"], ["Best epoch", "5 of 5"], ["Validation", "90.779%"], ["Decision", "Rejected"]],
  },
];

const stages = [
  {
    title: "Verify the machine",
    tag: "Environment",
    summary: "Before training anything, we proved that the Mac, Python environment, MLX, and Metal GPU path were real and reproducible.",
    did: "Created an isolated Python 3.11.9 environment, installed pinned MLX packages, and multiplied two matrices on the M2 Max GPU.",
    why: "If the compute stack is wrong, every later failure becomes ambiguous. Environment verification removes that uncertainty first.",
    command: ".venv/bin/python scripts/verify_mlx.py",
    lesson: "A successful import is not a GPU test. Perform an actual calculation and inspect the selected device.",
  },
  {
    title: "Lock trustworthy data",
    tag: "Dataset",
    summary: "We rejected an incomplete dataset mirror and downloaded the commit-pinned original BANKING77 files.",
    did: "Verified 10,003 train rows, 3,080 test rows, 77 labels, zero text overlap, and file checksums.",
    why: "Training on a drifting or incomplete mirror would make our results impossible to reproduce or compare.",
    command: ".venv/bin/python scripts/inspect_banking77.py",
    lesson: "A familiar dataset name is not enough. Record the source revision, counts, schema, and checksums.",
  },
  {
    title: "Measure before training",
    tag: "Baseline",
    summary: "The untouched model completed the full 3,080-question test before we changed any weights.",
    did: "Recorded 47.8571% accuracy, 0.468243 macro F1, and 176 invalid labels.",
    why: "Without a baseline, a low training loss can feel successful even when the model became worse.",
    command: ".venv/bin/python scripts/evaluate_full_baseline.py",
    lesson: "The baseline is the control group of a machine-learning experiment.",
  },
  {
    title: "Prepare SFT examples",
    tag: "Data engineering",
    summary: "We converted each question and intent into Qwen chat messages and reserved a balanced validation split.",
    did: "Created 9,233 training and 770 validation records, then inspected every tokenized length.",
    why: "The model trains on tokens, not visible strings. Template boundaries and sequence lengths must be verified before spending compute.",
    command: ".venv/bin/python scripts/inspect_tokenization.py",
    lesson: "Data formatting is part of the model. A correct CSV can still become incorrect training tokens.",
  },
  {
    title: "Prove the pipeline",
    tag: "Experiment 001",
    summary: "A ten-update LoRA run checked the complete pipeline without pretending to be an accuracy experiment.",
    did: "Loaded Qwen, attached LoRA, trained, saved the adapter, reloaded it, and generated a prediction.",
    why: "A cheap pipeline test catches structural failures before a two-hour training run.",
    command: ".venv/bin/python scripts/run_lora_pipeline.py",
    lesson: "Separate ‘does the pipeline work?’ from ‘does the model perform well?’",
  },
  {
    title: "Scale data carefully",
    tag: "Experiments 002 → 003b",
    summary: "We moved from 539 balanced examples to 1,925, keeping evaluation fixed. A high learning rate destroyed one run; a lower rate stayed finite.",
    did: "Diagnosed all 917,504 failed adapter values as non-finite, then retried at 2.5e-6.",
    why: "Changing data size changes the number of optimizer updates. A rate stable for a short run can explode over a longer run.",
    command: ".venv/bin/python scripts/diagnose_exp_003_failure.py",
    lesson: "NaN is evidence. Check when it started, whether memory was exhausted, and whether the saved artifact is usable.",
  },
  {
    title: "Train on all data",
    tag: "Experiment 004c",
    summary: "After a batch-size-7 run failed, batch size 1 and a very small learning rate completed one full epoch stably.",
    did: "Trained 917,504 q/v LoRA values for 9,233 updates and reached 49.4156% full-test accuracy.",
    why: "The full dataset teaches more linguistic variation, but stable optimization matters more than rushing with a large batch.",
    command: ".venv/bin/python scripts/run_exp_004c.py",
    lesson: "A probe validates safety. The full test—not the training log—validates usefulness.",
  },
  {
    title: "Increase adapter capacity",
    tag: "Experiment 005b",
    summary: "Adding k and o projections doubled trainable adapter values. The first full run reached NaN; halving the learning rate completed stably.",
    did: "Trained 1.835M values and reached 50.3896% accuracy—30 more correct answers than 004c.",
    why: "More trainable capacity can represent more task-specific behavior, but also changes optimization stability.",
    command: ".venv/bin/python scripts/run_exp_005b.py",
    lesson: "Our comparison changed both targets and learning rate, so it identifies a better configuration—not the isolated causal effect of k/o.",
  },
  {
    title: "Control the output space",
    tag: "Constrained decoding",
    summary: "Many ‘invalid’ outputs were sensible aliases. We constrained token generation to paths that finish as a canonical BANKING77 label.",
    did: "Reduced invalid labels from 173 to zero without touching model weights. The best observed accuracy became 51.1039%.",
    why: "A production classifier needs a closed output contract, not merely semantically plausible text.",
    command: ".venv/bin/python scripts/evaluate_exp_005b_constrained_full.py",
    lesson: "Inference algorithms are part of the system. Better behavior does not always require more training.",
  },
  {
    title: "Prove the constraint effect",
    tag: "Completed · Step 41",
    summary: "We repeated unconstrained and constrained inference at the same batch size over all 3,080 test records.",
    did: "The constraint preserved every valid prediction, converted 29 invalid outputs into correct labels, and converted the other 152 invalid outputs into valid but still-wrong labels.",
    why: "If two factors change together, we cannot honestly attribute the result to either one.",
    command: ".venv/bin/python scripts/classify_banking_request.py --check",
    lesson: "A controlled A/B test turned an encouraging result into defensible evidence: 51.1039% versus 50.1623%, with an exact paired p-value of 3.73e-9.",
  },
  {
    title: "Deploy and close the loop",
    tag: "Completed · Steps 42–44",
    summary: "We promoted the proven adapter and constraint into both a one-request CLI and a localhost JSON API.",
    did: "Sent a real HTTP request, received cash_withdrawal_charge with the exact adapter checksum and constraint mode, then stopped the server cleanly.",
    why: "An evaluated artifact is not yet a usable system. Deployment checks that preprocessing, model loading, decoding, and the external response contract still agree.",
    command: ".venv/bin/python scripts/serve_banking_classifier.py",
    lesson: "The production unit is the whole inference pipeline—not just the adapter file or its test score.",
  },
  {
    title: "Pivot to an encoder",
    tag: "Experiment 006 · BERT-Large",
    summary: "Qwen taught us the workflow, but generating label text was a poor match for a closed 77-way decision. We moved to a dedicated sequence classifier.",
    did: "Added a new 77-class head to BERT-Large and fine-tuned all model parameters for five epochs on MPS. The best validation checkpoint reached 91.5584%.",
    why: "Architecture fit can matter more than repeatedly tuning an ill-suited model. BERT directly scores the 77 choices instead of composing an answer token by token.",
    command: ".venv-encoder/bin/python scripts/run_exp_006.py",
    lesson: "Switching architecture was itself a learned conclusion from the Qwen experiment—not random model hopping.",
  },
  {
    title: "Reject plausible BERT refinements",
    tag: "Experiments 007–008",
    summary: "Class-weighted loss and label smoothing sounded reasonable, but both reduced validation accuracy.",
    did: "Compared each child against the unchanged Exp006 parent, counted repaired and harmed answers, and rejected both children.",
    why: "A mechanism can be theoretically sensible and still fail on real data. Promotion rules protect the champion from wishful thinking.",
    command: ".venv-encoder/bin/python scripts/analyze_exp_008_validation.py",
    lesson: "A failed controlled experiment is useful knowledge when its hypothesis and rejection rule were written first.",
  },
  {
    title: "Diagnose DeBERTa NaNs",
    tag: "Experiments 009a–009d",
    summary: "DeBERTa produced finite forward losses, but parameters became non-finite immediately after the first MPS optimizer update.",
    did: "Lowered the learning rate, instrumented microbatches, froze suspected tensors, inspected checkpoint dtypes, then cast the full model from float16 to float32.",
    why: "The failure was numerical precision—not bad labels or insufficient memory. float32 uses more memory but gives safer update range and precision.",
    command: ".venv-encoder/bin/python scripts/run_exp_009d_probe.py",
    lesson: "Locate the first bad operation before changing several settings. The float32 probe turned a guess into a diagnosis.",
  },
  {
    title: "Train the stronger encoder",
    tag: "Experiment 009d",
    summary: "Five stable float32 epochs made DeBERTa the validation champion and produced a 94.026% reporting-only test result.",
    did: "Selected epoch 5 using validation accuracy, recovered intact checkpoints after a metadata KeyError, verified save/reload, and only then opened the sealed test once.",
    why: "Checkpoint selection and final testing answer different questions. Validation chooses; test reports generalization after choices are finished.",
    command: ".venv-encoder/bin/python scripts/finalize_exp_009d.py",
    lesson: "A wrapper crash after training does not mean the model failed. Inspect durable artifacts before rerunning expensive work.",
  },
  {
    title: "Refine without touching test",
    tag: "Experiments 010–011",
    summary: "Train-only oversampling improved validation, then upper-layer-only training added one more correct answer.",
    did: "Exp010 reached 92.8571% validation. Exp011 froze layers 0–19, trained 51.5M upper-layer parameters, and reached 92.9870%.",
    why: "Small, controlled continuation can preserve learned language features while gently changing the decision boundary.",
    command: ".venv-encoder/bin/python scripts/run_exp_011.py",
    lesson: "The 94.123% test result is a report, not permission to tune against test errors. Exp011 remains champion by validation evidence.",
  },
  {
    title: "Audit label quality",
    tag: "Experiment 012",
    summary: "A five-fold out-of-fold classifier examined every training row without judging a row using a model that trained on it.",
    did: "Found 1,186 disagreements among 9,233 train rows. No validation or test rows were loaded, and no labels were silently rewritten.",
    why: "When the model is confidently wrong, either the decision boundary is weak or the supplied label may be noisy. Both deserve investigation.",
    command: ".venv-encoder/bin/python scripts/analyze_exp_011_errors.py",
    lesson: "Out-of-fold predictions are an audit signal, not ground truth. Disagreement alone is not enough to relabel data.",
  },
  {
    title: "Test hard-negative learning",
    tag: "Experiments 013–014 · rejected",
    summary: "We explicitly pushed true labels above their strongest rival for 162, then 282, ambiguous training rows. Both attempts lost one validation answer.",
    did: "Kept original labels, added a pairwise margin loss, measured the exact changed prediction, and rejected both children.",
    why: "More targeted loss does not guarantee better generalization—especially when many candidate rows are genuinely ambiguous.",
    command: ".venv-encoder/bin/python scripts/analyze_exp_013_validation.py",
    lesson: "Do not promote a complicated method merely because it sounds advanced. It must beat the simple champion on unchanged validation data.",
  },
  {
    title: "Test data-quality pruning",
    tag: "Experiment 015 · rejected",
    summary: "We removed only strong train-only noise candidates and retrained DeBERTa from the same original checkpoint. The run was stable, but accuracy fell.",
    did: "Removed 1,078 rows where the out-of-fold model disagreed and assigned the given label probability below 0.25; retained 8,155 rows and all 77 labels. Five epochs peaked at 90.7792% validation accuracy.",
    why: "This isolates one question: can less—but cleaner—training data outperform more noisy data? No synthetic text, relabelling, validation leakage, or test tuning is involved.",
    command: "artifacts/encoder/exp-015/training_result.json",
    lesson: "This rule removed useful signal along with suspected noise: 699/770 correct versus Exp011's 716/770. A clean run can still disprove its hypothesis.",
  },
];

const failures = [
  { type: "system", title: "Metal looked unavailable", symptom: "The first environment script said Metal was unavailable on an M2 Max.", cause: "A brittle system_profiler check confused missing display data with missing Metal support.", fix: "Deferred the verdict to an actual MLX matrix calculation, which proved GPU access." },
  { type: "code", title: "MLX version attribute crashed", symptom: "mlx.__version__ raised AttributeError before the compute check.", cause: "The top-level MLX package does not expose that attribute.", fix: "Read the installed package version through Python package metadata." },
  { type: "data", title: "The dataset mirror was incomplete", symptom: "The mirror contained 9,993/3,076 rows instead of 10,003/3,080.", cause: "Fourteen records were missing from the maintained mirror.", fix: "Switched to the pinned original PolyAI source and verified checksums." },
  { type: "code", title: "Tokenization suffix check failed", symptom: "The script claimed the assistant answer was not a suffix of the prompt.", cause: "Transformers 5 returned a BatchEncoding, but the code treated it as a list of token IDs.", fix: "Requested return_dict=False and matched MLX-LM chat processing exactly." },
  { type: "numeric", title: "Experiment 003 became NaN", symptom: "Loss rose sharply and every saved adapter value became non-finite.", cause: "The learning rate was unstable over the longer 1,925-update run—not a memory shortage.", fix: "Marked the artifact unusable and retried at a lower learning rate as 003b." },
  { type: "numeric", title: "Batch size 7 failed immediately", symptom: "Experiment 004 reported NaN at iteration 50 and used 12.556 GB peak memory.", cause: "The batch configuration changed optimization behavior; available memory alone did not guarantee stability.", fix: "Probed batch size 1, then trained 004c stably at 5e-7." },
  { type: "system", title: "Control-C blocked the rerun", symptom: "An interrupted run left an adapter_config and log, so overwrite protection stopped the next command.", cause: "The runner correctly refused to overwrite partial experiment evidence.", fix: "Archived partial artifacts and taught the runner to auto-archive after future interruptions." },
  { type: "numeric", title: "Larger q/k/v/o adapter diverged", symptom: "Experiment 005 was healthy through iteration 2,000, then reached NaN at 2,050.", cause: "The same learning rate was too aggressive for the expanded adapter over a full epoch.", fix: "Halved the rate to 2.5e-7; 005b completed all 9,233 updates." },
  { type: "system", title: "The local API returned RuntimeError", symptom: "The CLI worked, but POST /classify returned HTTP 500.", cause: "ThreadingHTTPServer invoked MLX Metal generation from worker threads.", fix: "Used a single-threaded HTTPServer; two subsequent requests returned HTTP 200." },
  { type: "code", title: "Constraint mask used the wrong MLX API", symptom: "ArrayAt had no .set() method.", cause: "The code assumed a JAX-style setter that MLX 0.32 does not provide.", fix: "Used MLX indexed replacement, matching the installed library." },
  { type: "code", title: "EOS looked like an invalid prefix", symptom: "The constraint failed after finishing a valid label.", cause: "MLX-LM computes one token ahead before it recognizes the current EOS stop token.", fix: "Allowed the discarded post-EOS calculation to pass through." },
  { type: "code", title: "Gradient accumulation inflated BERT loss", symptom: "The first two-update BERT probe reported loss near 17.65 instead of the expected random-class loss near 4.34.", cause: "The custom loss path was multiplied even though the current Trainer already normalizes gradient accumulation.", fix: "Removed the duplicate scaling; the corrected probe produced finite loss 4.413." },
  { type: "code", title: "BERT packaging rejected a valid run", symptom: "Five epochs completed, but the runner expected exactly one .bin weight artifact and stopped.", cause: "Its file pattern counted training_args.bin as a second model weight file.", fix: "Loaded the intact epoch-4 checkpoint through the supported path, packaged it, and verified an exact save/reload round trip." },
  { type: "numeric", title: "DeBERTa became NaN after one update", symptom: "Forward losses were finite, then pretrained parameters became non-finite immediately after AdamW stepped on MPS.", cause: "The downloaded checkpoint tensors were float16; full-parameter optimizer updates were numerically unsafe in that precision on this setup.", fix: "Cast the complete model to float32 before training. The probe and all five epochs then stayed finite." },
  { type: "code", title: "DeBERTa finished training, then metadata crashed", symptom: "After 75 minutes and all five epochs, the wrapper raised KeyError: evaluation.", cause: "The reporting code expected a configuration block that the training configuration did not contain.", fix: "Preserved the checkpoints, added the metadata block, and used a finalizer to select and verify the completed model without retraining." },
  { type: "code", title: "Upper-layer freeze guard named the pooler incorrectly", symptom: "Experiment 011's probe stopped because it could not find deberta.pooler parameters.", cause: "Transformers exposes this model's pooler at the root-level pooler prefix.", fix: "Corrected the expected prefix, reran the cheap probe, and only then started full training." },
  { type: "experiment", title: "Class weighting and label smoothing regressed", symptom: "BERT refinements 007 and 008 each scored below the 91.5584% parent.", cause: "The techniques changed confidence and class emphasis but did not repair their targeted validation errors.", fix: "Rejected both children under the predeclared rule and kept Exp006 unchanged." },
  { type: "experiment", title: "Hard negatives harmed one answer", symptom: "Experiments 013 and 014 both changed one correct contactless_not_working prediction into change_pin.", cause: "The rival-label objective over-corrected an already ambiguous boundary.", fix: "Rejected both children and kept Exp011 as champion; the failure motivated a cleaner-data hypothesis instead." },
  { type: "experiment", title: "Noise pruning removed useful signal", symptom: "Exp015 completed stably but reached only 90.7792% validation accuracy, 17 correct answers behind Exp011.", cause: "Out-of-fold disagreement was a useful suspicion signal but not reliable enough to delete 11.68% of the training set at the chosen threshold.", fix: "Rejected the child without test evaluation, retained Exp011, and recorded that model confidence cannot substitute for verified label corrections." },
];

const flashcards = [
  ["What was the goal of this project?", "To specialize an open Qwen3 1.7B model for BANKING77 intent classification, then prove improvement with held-out evaluation on an M2 Max using MLX and LoRA."],
  ["Why did you establish a baseline before fine-tuning?", "The untouched baseline is the control. Without it, falling training loss cannot tell us whether the completed model became more useful on unseen data."],
  ["Why LoRA instead of full fine-tuning?", "LoRA trains small low-rank adapter matrices while freezing the 1.7B base weights. It reduced memory and storage enough to run controlled experiments locally."],
  ["What caused the NaN runs, and how did you respond?", "They were numerical optimization failures, not simple out-of-memory errors. We found the first non-finite iteration, verified adapter values, rejected corrupted artifacts, and retried with safer batch and learning-rate settings."],
  ["What is the difference between validation loss and test accuracy?", "Validation loss measures token prediction quality during experiment development. Test accuracy measures exact correct labels on untouched questions. Loss guides training; the test decides usefulness."],
  ["Why did the 154-example pilot and 3,080-example test disagree?", "Two examples per intent create high sampling noise. The pilot is a cheap gate; the complete test gives a more stable estimate."],
  ["What did constrained decoding accomplish?", "It masked token paths that could not finish as one of the 77 canonical labels. It reduced invalid outputs to zero without changing the trained weights."],
  ["What did the batch-size-1 control prove?", "At the same batch size, constrained decoding preserved all 1,545 valid correct predictions, repaired 29 invalid outputs, and eliminated all 181 invalid labels. The gain is therefore caused by the constraint, not batch shape."],
  ["Why did BERT outperform Qwen so dramatically?", "BANKING77 is closed-set classification. BERT encodes the complete question and directly scores 77 classes, while Qwen had to generate an exact label string token by token. The architecture matched the task better."],
  ["What is a classifier head?", "A small final neural layer that receives the encoder representation and outputs one raw score, or logit, for each of the 77 intents. The highest-scoring position becomes the prediction."],
  ["Why did we try DeBERTa after optimizing BERT?", "We first tested three BERT refinements and none beat the parent. That evidence justified an architecture benchmark. DeBERTa offers a stronger representation of token content and position for classification."],
  ["Why did float32 fix DeBERTa on MPS?", "The checkpoint arrived in float16. Its forward pass worked, but optimizer updates overflowed into non-finite values. float32 used more memory while providing safer numerical range and precision for full fine-tuning."],
  ["What does freezing layers 0–19 mean?", "Those parameters still participate in inference but receive no gradient updates. Experiment 011 trained only upper encoder layers 20–23 and the classification components, preserving lower-level language knowledge."],
  ["What is an out-of-fold prediction?", "Each training row is predicted by a model trained on the other folds, never on that row. This creates a less self-confirming signal for spotting difficult or potentially noisy examples."],
  ["Why didn't we automatically correct suspicious labels?", "A model disagreement is not proof that the published label is wrong. Many BANKING77 boundaries are semantically ambiguous, so automatic relabelling could replace human labels with model mistakes."],
  ["What exactly is Exp015 testing?", "Whether retraining from the original DeBERTa checkpoint on 8,155 cleaner-looking rows beats training on all 9,233 rows. The only intended variable is train-data pruning; validation decides and test stays locked."],
  ["What did Exp015 teach us?", "The run was technically healthy but the hypothesis failed. Removing 1,078 suspicious rows reduced validation accuracy from 92.987% to 90.779%, so the audit score was not strong enough to justify deletion at that threshold."],
];

const quizQuestions = [
  { q: "Why keep the test set completely separate from training?", options: ["To make downloads smaller", "To measure generalization without answer leakage", "Because MLX cannot train on CSV", "To reduce GPU temperature"], answer: 1, why: "The test set is the final exam. If its answers influence training or tuning, the score no longer estimates performance on unseen data." },
  { q: "A run ends with very low training loss. What can you conclude?", options: ["It is definitely the best model", "It cannot produce invalid labels", "It optimized the training objective; test evaluation is still required", "The learning rate was perfect"], answer: 2, why: "Loss proves optimization progress, not generalization or product usefulness." },
  { q: "Why was the failed Experiment 003 adapter unusable?", options: ["It was too small", "Every adapter value was non-finite", "It used public data", "Its checksum was too long"], answer: 1, why: "NaN/Inf weights cannot support reliable inference. We preserved the evidence and rejected the artifact." },
  { q: "What does LoRA freeze?", options: ["The dataset", "The test metrics", "The original model weights", "The Mac GPU"], answer: 2, why: "The original Qwen weights remain unchanged while small attached matrices are trained." },
  { q: "Why can a 154-example pilot mislead us?", options: ["It has no labels", "Two examples per intent create high variance", "It always uses CPU", "Macro F1 cannot be computed"], answer: 1, why: "One changed answer moves pilot accuracy by about 0.65 percentage points." },
  { q: "What was the main benefit of constrained decoding?", options: ["It doubled model parameters", "It made training faster", "It guaranteed outputs belong to the 77-label vocabulary", "It replaced the test set"], answer: 2, why: "The constraint acts during inference and closes the output space without retraining." },
  { q: "Two experiments change LoRA targets and learning rate. What can you claim?", options: ["Exactly which target caused the gain", "The combined configuration performed differently", "Learning rate never matters", "The result is invalid and must be deleted"], answer: 1, why: "You can compare configurations, but causal attribution requires changing one factor at a time." },
  { q: "Why was an encoder classifier a better fit than generative Qwen here?", options: ["It has no parameters", "It directly scores the fixed 77 choices", "It does not need training data", "It always reaches 100%"], answer: 1, why: "The output is a closed taxonomy. Direct class logits avoid the extra difficulty of generating an exact allowed string." },
  { q: "DeBERTa forward losses were finite but weights became NaN after optimizer.step(). What did that indicate?", options: ["The test labels leaked", "The HTTP server was threaded", "The parameter-update precision was unstable", "The dataset had only one label"], answer: 2, why: "Instrumentation localized the first failure after the update. Casting float16 parameters to float32 fixed the numerical path." },
  { q: "Experiment 011 trained only layers 20–23. What happened to layers 0–19?", options: ["They were deleted", "They stayed frozen and still participated in the forward pass", "They became test data", "They were converted to labels"], answer: 1, why: "Frozen means used but not updated. This reduces trainable capacity while preserving lower-layer representations." },
  { q: "What is the correct interpretation of an out-of-fold disagreement?", options: ["The original label is certainly wrong", "It is a review signal, not proof", "The row must enter the test set", "The model should memorize it"], answer: 1, why: "A held-out model may reveal noise, but it may also misunderstand a genuinely difficult or ambiguous example." },
  { q: "What must happen before Exp015 replaces Exp011?", options: ["Its training loss must be lowest", "It must use the test answers", "It must exceed 92.987% on unchanged validation", "It must remove more than 1,078 rows"], answer: 2, why: "The promotion rule is fixed in advance. Test remains locked and training loss alone cannot establish generalization." },
];

const glossary = [
  ["Adapter", "A small set of trained weights attached to a frozen base model."],
  ["Baseline", "The untouched model result used as the control for later comparisons."],
  ["Batch size", "How many examples are processed together before one optimizer update."],
  ["BF16", "A 16-bit floating-point format that reduces memory while retaining useful numeric range."],
  ["Canonical label", "The exact official output string required by the task taxonomy."],
  ["Checkpoint", "A saved intermediate artifact or result that lets work resume safely."],
  ["Constrained decoding", "Masking invalid next tokens so generation can only follow approved output paths."],
  ["Data leakage", "Test or validation information improperly influencing training or model selection."],
  ["Epoch", "One pass over every example in the training set."],
  ["Gradient", "A signal showing how each trainable value should move to reduce loss."],
  ["Inference", "Using a trained model to make a prediction; no weight update occurs."],
  ["Learning rate", "The step size used when updating trainable values."],
  ["Logits", "Raw model scores for every possible next token before probabilities are calculated."],
  ["LoRA", "Low-Rank Adaptation: train small matrices instead of all original model weights."],
  ["Macro F1", "F1 calculated per class and averaged so every class has equal weight."],
  ["NaN", "Not a Number: a sign that numerical computation became invalid or unstable."],
  ["Overfitting", "Learning training examples too specifically and performing worse on unseen data."],
  ["Parameter", "A learned numeric value inside a model."],
  ["Seed", "A fixed number that makes randomized data selection or initialization reproducible."],
  ["SFT", "Supervised fine-tuning on examples containing an input and known desired output."],
  ["Token", "A text unit processed by the model; it may be a word, fragment, symbol, or special marker."],
  ["Validation set", "Held-out data used during development to monitor training and compare settings."],
  ["Classifier head", "The final layer that converts an encoder representation into one score for each allowed class."],
  ["Cross-entropy", "A loss that penalizes the model when probability is placed away from the correct class."],
  ["Encoder", "A model that turns an entire input sequence into contextual representations rather than generating an open-ended answer."],
  ["Float32", "A 32-bit floating-point format. It uses more memory than float16 but can make optimizer updates more numerically stable."],
  ["Frozen layer", "A layer used during prediction whose parameters are deliberately not updated during training."],
  ["Full fine-tuning", "Updating all or nearly all pretrained model parameters for a downstream task."],
  ["Gradient accumulation", "Combining gradients from several smaller batches before an optimizer update to imitate a larger effective batch."],
  ["Hard negative", "A wrong class that the model finds especially plausible and is explicitly taught to rank below the true class."],
  ["Label noise", "Training examples whose supplied labels may be incorrect, inconsistent, or ambiguous."],
  ["Label smoothing", "A training technique that gives a small amount of target probability to non-target classes to discourage overconfidence."],
  ["MPS", "Apple's Metal Performance Shaders backend used by PyTorch to run tensor work on the Mac GPU."],
  ["Noise pruning", "Removing strongly suspected noisy training rows under a predefined rule, without rewriting their labels."],
  ["Out-of-fold", "Predictions for each training row made by a model that did not train on that row."],
  ["Oversampling", "Repeating selected training examples or classes so they influence more optimizer updates."],
  ["Partial fine-tuning", "Updating selected model layers while freezing the rest, as in Experiment 011's upper-layer refinement."],
  ["Promotion rule", "A criterion fixed before evaluation that decides whether a child experiment replaces the current champion."],
];

const selector = document.querySelector("#modelSelector");
models.forEach((model, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.role = "tab";
  button.textContent = model.tab;
  button.dataset.model = model.id;
  button.setAttribute("aria-selected", index === 0 ? "true" : "false");
  button.addEventListener("click", () => selectModel(model.id));
  selector.append(button);
});

function selectModel(id) {
  const model = models.find((item) => item.id === id) || models[0];
  selector.querySelectorAll("button").forEach((button) => button.setAttribute("aria-selected", String(button.dataset.model === id)));
  document.querySelector("#modelKicker").textContent = model.kicker;
  document.querySelector("#modelName").textContent = model.name;
  document.querySelector("#modelStory").textContent = model.story;
  document.querySelector("#modelDecision").textContent = model.decision;
  document.querySelector("#modelScope").textContent = model.scope || "Full test · 3,080 questions";
  const pending = model.accuracy === null;
  document.querySelector("#accuracyValue").textContent = pending ? "Pending" : `${model.accuracy.toFixed(2)}%`;
  document.querySelector("#f1Value").textContent = pending ? "Pending" : (model.f1 / 100).toFixed(4);
  document.querySelector("#invalidValue").textContent = String(model.invalid);
  document.querySelector(".baseline-mark").style.left = `${model.benchmark ?? 47.8571}%`;
  requestAnimationFrame(() => {
    document.querySelector("#accuracyBar").style.width = pending ? "0%" : `${model.accuracy}%`;
    document.querySelector("#f1Bar").style.width = pending ? "0%" : `${model.f1}%`;
    document.querySelector("#invalidBar").style.width = `${(model.invalid / 176) * 100}%`;
  });
  document.querySelector("#accuracyNote").textContent = model.accuracyNote || (model.id === "base" ? "The control mark." : `${Math.round((model.accuracy - 47.8571) * 30.8)} net answers above baseline.`);
  document.querySelector("#invalidNote").textContent = model.invalidNote || (model.invalid === 0 ? "Closed output contract." : "Out of 3,080 test questions.");
  document.querySelector("#modelFacts").innerHTML = model.facts.map(([term, value]) => `<div><dt>${term}</dt><dd>${value}</dd></div>`).join("");
  localStorage.setItem("omtl-model", id);
}
selectModel(localStorage.getItem("omtl-model") || "base");

const timelineNav = document.querySelector("#timelineNav");
stages.forEach((stage, index) => {
  const item = document.createElement("li");
  item.innerHTML = `<button type="button" role="tab" data-stage="${index}" data-step="${String(index + 1).padStart(2, "0")}" aria-selected="${index === 0}">${stage.title}</button>`;
  item.querySelector("button").addEventListener("click", () => selectStage(index));
  timelineNav.append(item);
});
function selectStage(index) {
  const stage = stages[index];
  timelineNav.querySelectorAll("button").forEach((button) => button.setAttribute("aria-selected", String(Number(button.dataset.stage) === index)));
  document.querySelector("#timelineDetail").innerHTML = `
    <span class="stage-tag">${stage.tag}</span>
    <h3>${stage.title}</h3>
    <p>${stage.summary}</p>
    <div class="stage-columns"><div><span>What we did</span><p>${stage.did}</p></div><div><span>Why it mattered</span><p>${stage.why}</p></div></div>
    <code class="stage-command">${stage.command}</code>
    <div class="stage-lesson"><strong>What you should remember:</strong> ${stage.lesson}</div>`;
  localStorage.setItem("omtl-stage", String(index));
}
selectStage(Math.min(Number(localStorage.getItem("omtl-stage") || 0), stages.length - 1));

const failureFilters = ["all", "system", "data", "code", "numeric", "experiment"];
let activeFailureFilter = "all";
failureFilters.forEach((filter) => {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = filter;
  button.className = filter === "all" ? "active" : "";
  button.addEventListener("click", () => {
    activeFailureFilter = filter;
    failureFilters && document.querySelectorAll("#failureFilters button").forEach((item) => item.classList.toggle("active", item.textContent === filter));
    renderFailures();
  });
  document.querySelector("#failureFilters").append(button);
});
function renderFailures() {
  const visible = failures.filter((failure) => activeFailureFilter === "all" || failure.type === activeFailureFilter);
  document.querySelector("#failureCount").textContent = `${visible.length} incidents`;
  document.querySelector("#failureList").innerHTML = visible.map((failure, index) => `
    <article class="failure-item">
      <button type="button" aria-expanded="false" aria-controls="failure-${index}">
        <span class="failure-type ${failure.type}">${failure.type}</span><span class="failure-title">${failure.title}</span><span class="failure-toggle">+</span>
      </button>
      <div class="failure-body" id="failure-${index}" hidden>
        <div><span>Symptom</span><p>${failure.symptom}</p></div>
        <div><span>Cause</span><p>${failure.cause}</p></div>
        <div><span>Fix</span><p>${failure.fix}</p></div>
      </div>
    </article>`).join("");
  document.querySelectorAll(".failure-item button").forEach((button) => button.addEventListener("click", () => {
    const body = button.nextElementSibling;
    const open = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", String(!open));
    button.querySelector(".failure-toggle").textContent = open ? "+" : "−";
    body.hidden = open;
  }));
}
renderFailures();

function renderModelStack(mode = "qv") {
  const stack = document.querySelector("#modelStack");
  stack.innerHTML = Array.from({ length: 28 }, (_, index) => `<i class="${index >= 12 && ((mode === "qv" && index % 3 === 0) || (mode === "qkvo" && index % 2 === 0)) ? "trainable" : ""}" style="height:${75 + ((index * 17) % 115)}px"></i>`).join("");
  const qkvo = mode === "qkvo";
  document.querySelector("#loraValues").textContent = qkvo ? "1,835,008" : "917,504";
  document.querySelector("#loraShare").textContent = qkvo ? "0.107%" : "0.053%";
  document.querySelector("#loraSize").textContent = qkvo ? "7.35 MB" : "3.68 MB";
  document.querySelectorAll("#loraToggle button").forEach((button) => button.classList.toggle("active", button.dataset.lora === mode));
}
document.querySelectorAll("#loraToggle button").forEach((button) => button.addEventListener("click", () => renderModelStack(button.dataset.lora)));
renderModelStack();

let flashcardIndex = 0;
function renderFlashcard() {
  const [question, answer] = flashcards[flashcardIndex];
  document.querySelector("#flashcardNumber").textContent = `${String(flashcardIndex + 1).padStart(2, "0")} / ${String(flashcards.length).padStart(2, "0")}`;
  document.querySelector("#flashcardQuestion").textContent = question;
  document.querySelector("#flashcardAnswer").innerHTML = `<p>${answer}</p>`;
  document.querySelector("#flashcardAnswer").hidden = true;
  document.querySelector("#revealFlashcard").textContent = "Reveal answer";
}
document.querySelector("#revealFlashcard").addEventListener("click", (event) => {
  const answer = document.querySelector("#flashcardAnswer");
  answer.hidden = !answer.hidden;
  event.currentTarget.textContent = answer.hidden ? "Reveal answer" : "Hide answer";
});
document.querySelector("#nextFlashcard").addEventListener("click", () => { flashcardIndex = (flashcardIndex + 1) % flashcards.length; renderFlashcard(); });
renderFlashcard();

let quizIndex = 0;
let quizScore = 0;
let quizLocked = false;
function renderQuiz() {
  const card = document.querySelector("#quizCard");
  if (quizIndex >= quizQuestions.length) {
    const percent = Math.round((quizScore / quizQuestions.length) * 100);
    card.innerHTML = `<div class="quiz-score"><span class="quiz-count">Assessment complete</span><strong>${percent}%</strong><h3>${percent >= 80 ? "You can explain the lab." : "Review the flight recorder, then try again."}</h3><button class="primary-action small" id="restartQuiz" type="button">Restart quiz</button></div>`;
    document.querySelector("#quizProgress").style.width = "100%";
    localStorage.setItem("omtl-quiz-best", String(Math.max(percent, Number(localStorage.getItem("omtl-quiz-best") || 0))));
    document.querySelector("#restartQuiz").addEventListener("click", () => { quizIndex = 0; quizScore = 0; renderQuiz(); });
    return;
  }
  quizLocked = false;
  const item = quizQuestions[quizIndex];
  card.innerHTML = `<span class="quiz-count">Question ${quizIndex + 1} of ${quizQuestions.length}</span><h3>${item.q}</h3><div class="quiz-options">${item.options.map((option, index) => `<button type="button" data-option="${index}">${option}</button>`).join("")}</div><div class="quiz-feedback" hidden></div><button class="primary-action small quiz-next" type="button" hidden>Next question</button>`;
  document.querySelector("#quizProgress").style.width = `${(quizIndex / quizQuestions.length) * 100}%`;
  card.querySelectorAll(".quiz-options button").forEach((button) => button.addEventListener("click", () => answerQuiz(Number(button.dataset.option))));
}
function answerQuiz(choice) {
  if (quizLocked) return;
  quizLocked = true;
  const item = quizQuestions[quizIndex];
  if (choice === item.answer) quizScore += 1;
  const card = document.querySelector("#quizCard");
  card.querySelectorAll(".quiz-options button").forEach((button) => {
    const option = Number(button.dataset.option);
    if (option === item.answer) button.classList.add("correct");
    else if (option === choice) button.classList.add("wrong");
    button.disabled = true;
  });
  const feedback = card.querySelector(".quiz-feedback");
  feedback.hidden = false;
  feedback.innerHTML = `<strong>${choice === item.answer ? "Correct." : "Not quite."}</strong> ${item.why}`;
  const next = card.querySelector(".quiz-next");
  next.hidden = false;
  next.addEventListener("click", () => { quizIndex += 1; renderQuiz(); });
}
renderQuiz();

function renderGlossary(query = "") {
  const normalized = query.trim().toLowerCase();
  const terms = glossary.filter(([term, definition]) => `${term} ${definition}`.toLowerCase().includes(normalized));
  document.querySelector("#glossaryGrid").innerHTML = terms.length ? terms.map(([term, definition]) => `<article class="glossary-item"><h3>${term}</h3><p>${definition}</p></article>`).join("") : `<p>No matching term. Try a shorter search.</p>`;
}
document.querySelector("#glossarySearch").addEventListener("input", (event) => renderGlossary(event.target.value));
renderGlossary();

document.querySelectorAll("[data-reveal]").forEach((button) => button.addEventListener("click", () => {
  const target = document.getElementById(button.dataset.reveal);
  target.hidden = !target.hidden;
  button.textContent = target.hidden ? button.dataset.originalText || "Reveal answer" : "Hide answer";
}));
document.querySelectorAll("[data-reveal]").forEach((button) => { button.dataset.originalText = button.textContent; });

const depthToggle = document.querySelector("#depthToggle");
depthToggle.addEventListener("click", () => {
  const active = document.body.classList.toggle("deep-mode");
  depthToggle.setAttribute("aria-pressed", String(active));
  depthToggle.lastChild.textContent = active ? " Hide deeper notes" : " Show deeper notes";
  localStorage.setItem("omtl-depth", active ? "1" : "0");
});
if (localStorage.getItem("omtl-depth") === "1") depthToggle.click();

const tracked = [...document.querySelectorAll("[data-track]")];
const seen = new Set(JSON.parse(localStorage.getItem("omtl-seen") || "[]"));
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting && entry.intersectionRatio > .2) {
      seen.add(entry.target.dataset.track);
      localStorage.setItem("omtl-seen", JSON.stringify([...seen]));
      localStorage.setItem("omtl-last", `#${entry.target.id}`);
      updateReadingProgress();
    }
  });
}, { threshold: [.2] });
tracked.forEach((section) => observer.observe(section));
function updateReadingProgress() {
  const percent = Math.round((seen.size / tracked.length) * 100);
  document.querySelector("#readingProgress").textContent = `${percent}%`;
  document.querySelector("#readingProgressBar").style.width = `${percent}%`;
}
updateReadingProgress();
document.querySelector("#resumeButton").addEventListener("click", () => {
  const target = document.querySelector(localStorage.getItem("omtl-last") || "#map");
  target?.scrollIntoView({ behavior: "smooth" });
});
