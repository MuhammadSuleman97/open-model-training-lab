# Models

The first model is `Qwen/Qwen3-1.7B-MLX-bf16`, an official MLX conversion of
the post-trained Qwen3 1.7B text model.

It was selected because it:

- is small enough for comfortable BF16 LoRA work on 32 GB unified memory;
- avoids adding quantisation as a confounding variable in the first experiment;
- is text-only and directly supported by MLX-LM;
- has an Apache 2.0 licence; and
- is capable enough to provide a meaningful untouched-model baseline.

Downloaded weights are ignored by Git. `model_manifest.json` records the exact
resolved Hugging Face revision and local weight metadata after download.
