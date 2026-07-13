# Changelog

All notable changes to `portallib` are documented here.

## 0.1.0 - Unreleased

- Add the native safetensors PorTAL artifact format.
- Integrate standard Hub save, load, model-card, and push behavior through `ModelHubMixin`.
- Load local and Hugging Face PorTAL artifacts with `PortalModel.from_pretrained`.
- Generate task-specific LoRA weights and materialize ordinary PEFT models.
- Populate adapters through PEFT state-dict APIs with exact module-path validation.
- Export reloadable standard PEFT adapter directories without loading the base LLM.
- Add CPU contract tests and a library-based training example.
- Add normalized local/Hugging Face dataset loading with explicit optional Hub upload.
- Add the paper-faithful canonical core and thin per-base alignment artifact format.
- Add public core training, frozen-core adapter refitting, and exact evaluation APIs.
- Document and support exact source/refit layer paths in the training example, including a
  paper-scale Qwen-to-Gemma-3 refit invocation.
- Add paper, citation, architecture, compatibility, and public artifact-release guidance.
- Point package, citation, and model-card metadata at the public `ramp-public/portallib` repository.
- Add balanced task rounds, EMA loss normalization, per-base latent-gradient balancing, and
  best-epoch checkpoint selection.
- Separate source and refit data budgets: use all source examples by default and 500 examples per
  task for target alignment refitting.
- Decouple source pool size from optimization length with a fixed balanced-round budget, defaulting
  to 1,000 joint source steps per epoch.
- Add patience-based early stopping and task-level regression diagnostics relative to epoch zero.
