# PorTAL: Portable Task Adapters for LLMs

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ramp-public/portallib/main/docs/assets/portal_header_dark_v2.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ramp-public/portallib/main/docs/assets/portal_header_light_v3.png">
    <img src="https://raw.githubusercontent.com/ramp-public/portallib/main/docs/assets/portal_header_light_v3.png" width="560" alt="PorTAL wordmark passing through two portals">
  </picture>
</p>

> Alpha research release [announced by Ramp Labs](https://x.com/RampLabs/status/2072381992285647280).
> APIs and artifact formats may evolve before the first stable release.

PorTAL learns a base-agnostic task latent and a light per-base alignment that generates ordinary
per-layer LoRA weights. A task can be trained once, adapted to supported frozen base models, and
exported as a standard Hugging Face PEFT adapter.

`portallib` is an alpha Python library for loading, training, saving, publishing, and exporting
PorTAL artifacts with standard PyTorch and Hugging Face interfaces.

The included pinned recipes reproduce the PorTAL source-training, target-refitting, and evaluation
method described by Ramp Labs. Reported results should be generated from the released artifacts and
their recorded evaluation configuration rather than treated as fixed package guarantees.

![PorTAL source training and target-base refitting phases](https://raw.githubusercontent.com/ramp-public/portallib/main/docs/assets/portal_phases.gif)

During source training, PorTAL jointly learns the task-latent table, one shared canonical core, and
one alignment for each source base. To port the learned tasks, it freezes the latent table and core
and refits only a fresh alignment for the target base. The resulting task adapter is exportable as
an ordinary PEFT LoRA adapter.

## Install

Install the inference library from PyPI:

```bash
pip install portallib
```

Install the optional dataset dependency for complete training and evaluation workflows:

```bash
pip install 'portallib[training]'
```

Python 3.11 and 3.12 are supported. Install a CUDA-compatible PyTorch build for GPU training before
installing the training extra when your platform requires a specific CUDA wheel.

## Load and export

Load a native PorTAL artifact, select a trained task, and obtain a normal PEFT model:

```python
from transformers import AutoModelForCausalLM
from portallib import PortalModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    revision="1cfa9a7208912126459214e8b04321603b3df60c",
)
portal = PortalModel.from_pretrained(
    "RampPublic/portal-qwen3-4b",
    revision="v0.2.0",
)
model = portal.get_peft_model("rte", base)
model.save_pretrained("./portal-rte-qwen3-4b")
```

A task can also be exported without loading the base LLM:

```python
portal.export_peft("rte", "./portal-rte-qwen3-4b")
```

The exported directory is an ordinary PEFT adapter and reloads with
`PeftModel.from_pretrained`.

## Published artifacts

| Artifact | Role |
|---|---|
| [`RampPublic/portal-qwen3-1.7b`](https://huggingface.co/RampPublic/portal-qwen3-1.7b/tree/v0.2.0) | Jointly trained shared weights plus the 1.7B alignment |
| [`RampPublic/portal-qwen3-4b`](https://huggingface.co/RampPublic/portal-qwen3-4b/tree/v0.2.0) | Jointly trained shared weights plus the 4B alignment |
| [`RampPublic/portal-qwen3-8b`](https://huggingface.co/RampPublic/portal-qwen3-8b/tree/v0.2.0) | 1,000-example-per-task refit |
| [`RampPublic/portal-gemma-3-4b`](https://huggingface.co/RampPublic/portal-gemma-3-4b/tree/v0.2.0) | 1,000-example-per-task cross-family refit |
| [`RampPublic/portal-gemma-4-e2b`](https://huggingface.co/RampPublic/portal-gemma-4-e2b/tree/v0.2.0) | 1,000-example-per-task heterogeneous-attention refit |

The recipes load the `v0.2.0` artifact revisions. Each repository contains one base-specific native
PorTAL artifact; task-specific standard PEFT adapters can be generated from it as needed.

## Python workflows

[`examples/train_example.py`](https://github.com/ramp-public/portallib/blob/main/examples/train_example.py) is thin orchestration around the public
canonical trainer APIs. It freezes each base model, jointly learns shared task latents and a canonical
core with one thin alignment per source base, evaluates epoch zero and every training epoch, restores
the best held-out epoch, and writes one native artifact per source base. Its only model downloads are
the raw Hugging Face bases selected for source training.

The complete pinned recipe is a short, editable block near the top of the file. It selects the
dataset, exact model revisions, output directory, source bases, and `PortalTrainingConfig`:

```bash
python examples/train_example.py
```

[`examples/refit_example.py`](https://github.com/ramp-public/portallib/blob/main/examples/refit_example.py) loads either source artifact as a carrier
for the task vectors and canonical core learned jointly from Qwen3-1.7B and Qwen3-4B. It downloads
only the new raw target base, freezes the shared components, and trains a fresh target alignment:

```bash
python examples/refit_example.py
```

The checked-in recipe reads the shared weights from `RampPublic/portal-qwen3-4b`; this does not make
the refit 4B-only—the 1.7B and 4B source artifacts contain identical jointly trained task latents and
canonical core weights. The default target is Qwen3-8B with at most 1,000 training examples per task.
The adjacent Gemma 3 recipe uses the same shared components and its exact text-decoder layer path.

[`examples/evaluate_example.py`](https://github.com/ramp-public/portallib/blob/main/examples/evaluate_example.py) loads a trained PorTAL artifact and
its matching raw base, then reports the base floor, adapted per-task metrics, macro metrics, and
accuracy lift:

```bash
python examples/evaluate_example.py
```

The checked-in evaluation recipe uses `RampPublic/portal-qwen3-8b`; change the artifact and matching
base recipe together to evaluate one of the other published source or refit artifacts.

The examples are repository assets rather than installed console commands. Clone the repository to
run them, then install the released training package:

```bash
git clone https://github.com/ramp-public/portallib
cd portallib
pip install 'portallib[training]==0.2.0'
python examples/train_example.py
```

The trainer, refitter, and evaluator are regular Python APIs. The examples define their recipes as
editable Python objects and invoke `PortalCoreTrainer`, `PortalAdapterRefitter`, and
`PortalEvaluator` directly.

## Configuration-driven CLI

The CLI runs the same library workflows from strict TOML recipes, which is useful for containers,
scheduled jobs, and reproducible subprocess execution:

| Workflow | Python | CLI |
|---|---|---|
| Source training | `python examples/train_example.py` | `portallib train --config examples/configs/train.toml` |
| Target refitting | `python examples/refit_example.py` | `portallib refit --config examples/configs/refit.toml` |
| Evaluation | `python examples/evaluate_example.py` | `portallib evaluate --config examples/configs/evaluate.toml` |

Install the training dependencies and optionally validate a recipe without loading models:

```bash
pip install 'portallib[training]==0.2.0'
portallib validate --config examples/configs/train.toml
portallib train --config examples/configs/train.toml
```

Recipes can also be piped without creating a temporary file:

```bash
generate-recipe | portallib evaluate --config -
```

Relative paths in piped recipes resolve from the current working directory.

The CLI rejects unknown keys and command/recipe mismatches. It emits JSONL progress and final
results, uses exit code `2` for recipe errors and `1` for runtime failures, and reads Hugging Face
authentication from `HF_TOKEN` or the host's cached login. Credentials do not belong in recipe
files. See [`CLI.md`](https://github.com/ramp-public/portallib/blob/main/CLI.md) for the schema and
automation contract.

`portallib inspect` prints an artifact's configuration — base model and revision, tasks, canonical
dimensions, and projection layout — by reading only its `config.json`, without downloading weights
or loading the base model:

```bash
portallib inspect RampPublic/portal-qwen3-1.7b --revision v0.2.0
portallib inspect ./local-artifact --json
```

[`REPRODUCING.md`](https://github.com/ramp-public/portallib/blob/main/REPRODUCING.md) records pinned dataset and model revisions, the complete training
configuration, checkpoint selection, and source/Qwen/Gemma recipes.

[`COMPUTE.md`](https://github.com/ramp-public/portallib/blob/main/COMPUTE.md) shows how to run any example locally with Docker or remotely through
Modal. The compute wrapper provisions the runtime and persistent storage; the training and evaluation
behavior comes from the installed `portallib` release and selected recipe.

## Model compatibility

PorTAL supports Qwen3 and cross-family refitting to Gemma 3 and Gemma 4. Qwen3 exposes decoder
layers at `model.layers`; Gemma 3 and Gemma 4 expose their text decoder at
`model.language_model.layers`. Gemma 4 uses the multimodal auto-model loader and explicit sparse
projection targets because its projection dimensions and available projections vary across layers.

Every artifact uses the same explicit projection-target format. Other model families can use it
when their exact decoder-layer and projection paths are supplied through `BaseModelSpec`. Set
`allow_heterogeneous_targets=True` to opt into sparse per-layer targets or varying projection
widths. PorTAL records every resolved target and validates its exact path and dimensions before
training, refitting, evaluation, or PEFT materialization. It does not infer architecture mappings
from fuzzy module-name patterns.

The checked-in `modules=("q", "v")` setting generates LoRA for query/value projections. Set it to
`("q", "k", "v", "o", "gate", "up", "down")` to include the attention output and MLP
projections. In both cases, the base model parameters remain frozen.

## Artifact format

Native artifacts use the standard Hugging Face layout:

- `config.json` contains `format_version=1`, the base model and revision, task names, LoRA settings,
  and one explicit list of exact projection targets for both uniform and heterogeneous bases.
- `model.safetensors` contains `task_latents`, the canonical `core`, and one base-specific
  `alignment`, with `portallib` format metadata.
- `README.md` is the generated model card.

`PortalModel` is a `torch.nn.Module` and inherits `ModelHubMixin`. Its `state_dict()` uses the same
`task_latents`, `core.*`, and `alignment.*` names as the native safe artifact, while
`save_pretrained`, `from_pretrained`, and `push_to_hub` follow standard Hugging Face Hub behavior:

```python
portal.push_to_hub("your-namespace/portal-qwen3-4b", private=True)
```

Configured layers and projections are resolved deterministically. Missing modules, incompatible
dimensions, unknown format versions, and inconsistent target declarations fail explicitly.

## Public API

- `PortalConfig` validates artifacts and builds exact configurations from supported base models.
- `BaseModelSpec`, `load_base`, `load_dataset`, and `runtime_device` provide the shared loading
  surface used by the Python examples and CLI.
- `PortalCoreTrainer` jointly trains shared latents/core and one alignment per source base using
  balanced per-task updates, EMA loss normalization, and per-base latent-gradient balancing.
- `PortalAdapterRefitter` freezes a source artifact's latents/core and trains only a target alignment.
- `PortalTrainingConfig.from_portal_config` preserves an artifact's architecture while selecting a
  new optimization recipe for refitting.
- `PortalEvaluator` evaluates or compares raw and adapted bases while reporting character-normalized
  multiple-choice accuracy and token-mean gold NLL.
- `EvaluationResult.to_dict` returns the canonical JSON-ready evaluation representation.
- `PortalModel` is the PyTorch task-latent/core/alignment module and loads, saves, publishes,
  materializes, and exports trained artifacts.
- `ChoiceDataset` loads and saves the normalized local/Hub task schema and supports explicit Hub upload.
- `collate_gold_batch` provides the causal-LM batch format used by the training APIs.

## Development

```bash
uv run ruff check src tests examples scripts
uv run pytest -q
uv run python -m build
```

PorTAL is licensed under Apache-2.0.

For questions or feedback, reach Ben Geist on X at [@bgeist](https://x.com/bgeist).

## Citation

If you use PorTAL, cite the software metadata in
[`CITATION.cff`](https://github.com/ramp-public/portallib/blob/main/CITATION.cff).
