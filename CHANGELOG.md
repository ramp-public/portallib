# Changelog

All notable changes to `portallib` are documented here.

## Unreleased

- Place generated LoRA factors from each target projection's live output device and dtype so
  adapters remain active and differentiable when decoder layers are disk-offloaded.
- Support explicit sparse target layouts and shape-grouped base alignments for models whose
  projection dimensions vary across decoder layers.
- Add explicit multimodal-model loading for text-only execution through multimodal wrappers.

## 0.2.0 - 2026-07-16

- Add the `portallib train`, `refit`, `evaluate`, and `validate` commands with strict, versioned
  TOML recipes.
- Emit JSONL epoch events, structured errors, and final machine-readable results, with optional
  result-file persistence.
- Keep credentials outside recipe files, accept recipes from files or standard input, and resolve
  local paths relative to the recipe location or current working directory respectively.
- Share the same Hugging Face loading helpers across the CLI and editable Python examples.
- Add pinned CLI recipes for source training, target refitting, and evaluation.

## 0.1.2 - 2026-07-16

- Derive `portallib.__version__` from installed distribution metadata so it always matches the
  package version.

## 0.1.1 - 2026-07-16

- Preserve prompt/continuation token boundaries consistently during training and evaluation,
  including WinoGrande blanks at the beginning or within a sentence.
- Normalize prepared choices to one leading space with no trailing whitespace and pin the verified
  dataset revision used by the checked-in recipes.
- Allow `PortalEvaluator` to evaluate any requested task subset contained in an artifact.
- Expose optional dtype, device-map, and attention-implementation controls in the example base
  recipes without forcing a bulk device move after Hugging Face model loading.
- Validate the four published `v0.1.0` model artifacts against the normalized dataset; their model
  weights and artifact revisions remain unchanged.

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
