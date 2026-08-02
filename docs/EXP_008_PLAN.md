# Experiment 008 — BERT confidence regularization

## Evidence

Experiment 007's class-balanced child regressed from 91.5584% to 91.1688%.
It fixed three parent errors but harmed six correct predictions. It produced no
fixes within the upweighted classes and harmed three examples from those
classes, so class weighting is rejected.

Experiment 006's error analysis found thirteen incorrect validation predictions
with confidence above 90%. Several involve ambiguous neighboring intents.

## Hypothesis

Label smoothing of `0.05` may reduce brittle overconfidence and improve
generalization between neighboring intents. Experiment 008 starts again from
the canonical Experiment 006 model, uses uniform class weights and changes only
the loss regularization.

## Precommitted rules

- Parent: roundtrip-verified Experiment 006 selected model.
- Label smoothing: exactly `0.05`; no search over smoothing values.
- Class weights: uniform.
- Refinement: at most two epochs at learning rate `1e-6`.
- Promotion: validation accuracy must strictly exceed 91.5584%.
- Ties keep the parent.
- Test remains sealed.

## Probe result

The two-update backward probe passed with finite loss `0.462308`, finite
gradients and peak MPS driver memory of `9.541 GiB`. It loaded no validation or
test records and saved no weights. Full refinement is therefore authorized
under the precommitted rules above.

## Training result

The two refinement epochs reached validation accuracies of `0.911688` and
`0.912987`. The best child was two correct predictions behind the canonical
parent's `0.915584`, so the strict promotion rule rejected it. Experiment 006
remains canonical and the test split remains sealed.

## Transition analysis

The candidate changed thirteen predictions, fixing four parent errors while
harming six parent-correct records. It fixed zero of the thirteen
high-confidence parent errors that motivated the experiment. The proposed
mechanism therefore failed, and Experiment 008 is closed as rejected.
