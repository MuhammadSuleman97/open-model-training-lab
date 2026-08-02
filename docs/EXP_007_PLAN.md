# Experiment 007 — Refine the trained BERT model

## Why this comes before another architecture

Experiment 006 produced a real trained asset at 91.5584% validation accuracy.
The next learning objective is to improve that model through a controlled
change, not immediately replace it. DeBERTaV3 remains a later architecture
benchmark only after the BERT refinement ladder is complete.

## Evidence-based hypothesis

The validation split is balanced at ten examples per intent, while training
frequencies range widely. Several weak validation intents, including
`card_swallowed`, `card_acceptance` and `topping_up_by_card`, are among the
lower-frequency training classes. Experiment 007 continues from the selected
Experiment 006 weights using inverse-square-root class weighting.

This makes one primary change: the loss gives moderately more influence to
underrepresented classes. Label smoothing remains zero so its effect is not
mixed into the result.

## Precommitted rules

- Start from the roundtrip-verified Experiment 006 selected model.
- Use training frequencies only to calculate class weights.
- Keep the same training, validation and sealed test records.
- Train for at most two refinement epochs at learning rate `1e-6`.
- Select by validation accuracy; ties keep the earlier checkpoint.
- Promote the refinement only if it strictly exceeds 91.5584% validation
  accuracy.
- Keep 95% as the program target, not a guaranteed or manufactured result.

## Later ladder

If class balancing does not improve the model, the next isolated BERT
refinement may test confidence regularization or a contrastive objective. Only
after the current model's evidence-based refinement ladder is exhausted will a
new encoder architecture be compared.
