# Public presence plan

The goal is not to open accounts everywhere. It is to create one strong public
artifact, then explain it repeatedly at different depths.

## The presence stack

### 1. GitHub — canonical evidence

GitHub owns the source, history, issues and reproducibility discussion.

Profile setup for `MuhammadSuleman97`:

- **Name:** Muhammad Suleman
- **Bio:** `Building reproducible open-model experiments on Apple Silicon. MLX, PyTorch MPS, evaluation and failures included.`
- **Website:** the live Flight Recorder URL
- **Pinned repository:** `open-model-training-lab`
- **Profile README:** add after the repository is live; lead with what you are
  learning and link only two or three strong projects.

Repository setup:

- public visibility;
- description and topics from [SHARING.md](SHARING.md);
- Issues and Discussions enabled;
- private vulnerability reporting enabled;
- GitHub Pages deployed from `learning/`;
- a social-preview image using the suggested result card; and
- Exp011 accuracy described as educational, never production-ready.

### 2. X — ongoing learning log

X is the distribution channel, not the source of truth.

Suggested profile:

- **Name:** Muhammad Suleman
- **Bio:** `Learning open models in public · Apple Silicon · MLX + PyTorch MPS · Reproducible experiments and failures included`
- **Website:** live guide
- **Pinned post:** the five-post launch thread

Do not create a “launch-only” account. Before and after launch, post compact
technical lessons:

- one graph or result plus the experimental question;
- one failure diagnosis and the evidence that localized it;
- one beginner concept explained in your own words;
- one rejected hypothesis and what you would test differently; and
- replies that answer other people's questions without linking your project.

### 3. Hugging Face — ML-native credibility

This is the best additional home for this project. Hugging Face supports model
repositories, model cards and static HTML Spaces.

Recommended sequence:

1. Publish `learning/` as a public static Space.
2. Link the Space to BANKING77 and the base models in its card metadata.
3. Publish an Exp011 model repository only if you intentionally choose to
   distribute the selected weights and have verified licences, size and model
   card limitations.
4. Never upload the entire ignored `artifacts/` directory.

A model card should state base model, dataset, intended use, limitations,
training configuration and evaluation results. The GitHub codebase remains the
full research trail; Hugging Face becomes the runnable ML-facing entry point.

### 4. DEV Community — durable long-form explanation

Publish one substantial article after GitHub and the live guide work. The
article should teach architecture fit and failure diagnosis rather than repeat
the README. Link to GitHub at the end. Disclose AI assistance according to the
platform's current guidelines.

### 5. Hacker News — optional launch spike

Use Show HN only when the guide is public and immediately usable. The useful
artifact is the interactive course, not an announcement about a repository.
Participate in the community and stay available for the discussion.

### 6. Reddit — community participation first

Do not drop the project into r/LocalLLaMA from a new or promotion-only account.
Current moderation emphasizes meaningful non-promotional participation and
substantive analysis. Join discussions first. Later, share one specific
technical lesson—such as float16 MPS optimizer failure—with disclosure that you
built the linked project.

## Channels not prioritized

- **LinkedIn:** excluded by preference.
- **Medium:** adds little if the canonical article already lives on DEV.
- **Substack:** premature for a single project; use after a repeatable cadence.
- **YouTube:** valuable only if you want to make a three-to-five-minute screen
  walkthrough of the interactive guide.
- **Discord communities:** participate where already relevant; do not join only
  to paste a launch link.

## Fourteen-day launch rhythm

| Day | Action |
|---:|---|
| 0 | Publish GitHub, verify logged-out access, CI and live guide |
| 1 | Post and pin the X launch thread; reply with the failure timeline |
| 2 | Publish the Hugging Face static Space |
| 3 | Post the DeBERTa float16 → float32 diagnosis as a standalone lesson |
| 5 | Publish the DEV article |
| 7 | Post what Exp015 taught about label-noise auditing |
| 9 | Ask for one reproducibility run on another Apple Silicon machine |
| 11 | Share a beginner glossary card or quiz question |
| 14 | Publish a short retrospective: feedback, reproductions and next experiment |

## Measure useful signals

Follower count is a lagging metric. Track:

- repository clones and stars;
- guide visits;
- issues or Discussions containing reproducibility evidence;
- Hugging Face Space likes/duplicates;
- technical replies and bookmarks on X; and
- whether another person can explain or reproduce one experiment.

The ideal public identity is **the person who documents the whole experiment,
including the parts that failed**.
