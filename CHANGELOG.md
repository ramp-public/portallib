# Changelog

All notable changes to `portallib` are documented here.

## 0.1.0 - 2026-07-15

Initial public release of `portallib` on PyPI.

- Canonical shared task-latent/core architecture with one exact-path alignment per base.
- Joint multi-source core training, frozen-core target refitting, and normalized evaluation APIs.
- Deterministic balanced task rounds, EMA loss normalization, per-base latent-gradient
  balancing, learning-rate warmup, checkpointing, and best-validation-epoch selection.
- Local and Hugging Face dataset loading, canonical JSON serialization, and explicit Hub upload.
- Standard `save_pretrained`, `from_pretrained`, and `push_to_hub` behavior through
  `ModelHubMixin`.
- Exact PEFT model materialization and reloadable task-specific PEFT adapter export.
- Published Qwen3-1.7B and Qwen3-4B source artifacts containing the jointly trained shared core and
  task latents with their respective base alignments.
- Published 1,000-example-per-task Qwen3-8B and Gemma 3 4B refit artifacts.
- Pinned source training, target refitting, and evaluation recipes.
- CPU correctness contracts plus Docker and Modal execution guidance.
