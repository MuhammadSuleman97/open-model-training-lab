---
title: Open Model Training Lab
emoji: 🧪
colorFrom: blue
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
datasets:
  - PolyAI/banking77
models:
  - Qwen/Qwen3-1.7B-MLX-bf16
  - google-bert/bert-large-cased
  - microsoft/deberta-v3-large
tags:
  - apple-silicon
  - fine-tuning
  - mlx
  - pytorch
  - education
---

# Open Model Training Lab

An interactive, beginner-friendly account of training Qwen3, BERT-Large and
DeBERTa-v3-large for BANKING77 intent classification on an Apple M2 Max.

The guide covers fifteen controlled experiments, including numerical failures,
rejected refinements, data-leakage protections, a searchable glossary, quiz and
interview practice.

**Best recorded result:** 92.99% validation accuracy and 94.12%
reporting-only test accuracy with a DeBERTa upper-layer refinement.

Source and reproduction instructions:
[msulemans/open-model-training-lab](https://github.com/msulemans/open-model-training-lab)

This is educational software, not a production banking classifier.
