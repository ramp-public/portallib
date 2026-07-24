# PorTAL command-line interface

The `portallib` command runs source training, target-base refitting, and evaluation from strict,
versioned TOML recipes. It is intended for local processes, containers, schedulers, and sandboxed
workers that need a stable subprocess boundary around the Python APIs.

For programmatic use, the corresponding Python entry points are `PortalCoreTrainer.train()`,
`PortalAdapterRefitter.refit()`, and `PortalEvaluator.evaluate()`. Complete Python workflows are in
`examples/train_example.py`, `examples/refit_example.py`, and `examples/evaluate_example.py`.

## Commands

```bash
portallib validate --config recipe.toml
portallib train --config recipe.toml
portallib refit --config recipe.toml
portallib evaluate --config recipe.toml
```

Pass `-` to read the same TOML recipe from standard input:

```bash
generate-recipe | portallib evaluate --config -
```

Every recipe declares `recipe_version = 1` and a `kind` matching the command that executes it.
`validate` checks the complete recipe structure without loading a dataset, model, or artifact.
Unknown keys, duplicate tasks or source model IDs, invalid types, and command/recipe mismatches are
errors.

`recipe_version` versions this TOML automation contract. It is independent from the
`PortalConfig.format_version` stored in native artifacts, which versions the artifact format.

The checked-in starting points are:

- [`examples/configs/train.toml`](examples/configs/train.toml)
- [`examples/configs/refits/qwen3-8b.toml`](examples/configs/refits/qwen3-8b.toml)
- [`examples/configs/refits/gemma-3-4b.toml`](examples/configs/refits/gemma-3-4b.toml)
- [`examples/configs/refits/gemma-4-e2b.toml`](examples/configs/refits/gemma-4-e2b.toml)
- [`examples/configs/refits/mistral-7b.toml`](examples/configs/refits/mistral-7b.toml)
- [`examples/configs/evaluate.toml`](examples/configs/evaluate.toml)

## Automation contract

CLI progress is emitted as one JSON object per line. Training and refitting emit an `epoch` event
after epoch zero and every completed epoch, followed by one `result` event. Evaluation emits one
`result` event. When `result_path` is present, the final result is also written there as formatted
JSON.

Exit codes are:

| Code | Meaning |
|---:|---|
| `0` | Validation or execution succeeded |
| `1` | Model loading, training, refitting, evaluation, or persistence failed |
| `2` | The TOML file or recipe schema is invalid |

Recipe files never contain authentication tokens. Hugging Face libraries use `HF_TOKEN` or the
host's cached login. Model and dataset revisions can be pinned to immutable commits or tags.
Local model and artifact locations begin with `./`, `../`, `~/`, or an absolute path. Relative
paths resolve from the TOML file's directory for `--config recipe.toml` and from the current working
directory for `--config -`.

## Common fields

All recipes support:

| Field | Meaning |
|---|---|
| `recipe_version` | Must be `1` |
| `kind` | `train`, `refit`, or `evaluate` |
| `result_path` | Optional final JSON path, resolved relative to the recipe |
| `tasks` | Optional ordered task subset for training or evaluation |
| `[dataset]` | Exactly one Hub `repo_id` or local JSON `path`, plus an optional Hub `revision` |
| `[runtime]` | Host `device` and `dtype`; both default to `auto` |

A local dataset is explicit and unambiguous:

```toml
[dataset]
path = "data/tasks.json"
```

A Hub dataset can be pinned independently:

```toml
[dataset]
repo_id = "RampPublic/portallib-tasks"
revision = "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
```

The runtime accepts any valid PyTorch device string. `dtype` is `auto`, `bfloat16`, `float16`, or
`float32`. Automatic selection uses CUDA with bf16 when available and otherwise CPU with fp32.

## Base model tables

Source training uses one or more `[[bases]]` tables. Refitting and evaluation use one `[base]`
table. Each table accepts:

| Field | Required | Meaning |
|---|---:|---|
| `model_id` | yes | Hugging Face model ID or local model path |
| `revision` | no | Exact model revision |
| `layer_path` | no | Exact decoder-layer path; default `model.layers` |
| `module_paths` | no | Explicit short-name to projection-path mapping |
| `dtype` | no | Per-base dtype override |
| `device_map` | no | Hugging Face device-map string or inline table |
| `attn_implementation` | no | Hugging Face attention implementation |
| `loader` | no | `causal_lm` (default) or `multimodal_lm` |
| `allow_heterogeneous_targets` | no | Opt into exact sparse targets and per-layer projection widths |

When `device_map` is present, the loader preserves Hugging Face placement and does not apply a bulk
device move. Exact layer and projection paths are passed to `PortalBase` and validated before
training.

## Training fields

`[training]` maps directly to `PortalTrainingConfig`. Source recipes may set architecture and
optimizer fields. Refit recipes inherit architecture fields from the source artifact and reject
attempts to override them. Use the string `"all"` for `source_max_examples`, `eval_max_examples`,
or evaluation `max_examples` when no cap is desired.

Refit recipes may set `refit_gradient_strategy = "norm_equalized"` to equalize the global
alignment-gradient norm contributed by every task before each optimizer step. Set
`refit_choice_loss_weight` above zero to add character-normalized multiple-choice cross-entropy to
gold-token NLL; this uses the same continuation boundaries and `acc_norm` scores as
`PortalEvaluator`. Their defaults, `"sum"` and `0.0`, preserve the standard gold-NLL refit.

Source training requires `output_dir` and one or more `[[bases]]` tables. It writes one complete
artifact per source base beneath the output directory. Refitting requires `output_dir`,
`source_artifact`, and one `[base]` target table. Checkpoints are written beneath
`output_dir/checkpoints`.

Evaluation requires `artifact` and the matching `[base]` table. It reports the raw-base floor, the
adapted result, and macro accuracy lift. `max_examples`, `max_prompt`, and `batch_size` control its
evaluation workload.

All relative filesystem paths use the same anchor: the TOML file's directory for file-backed
recipes, or the current working directory for recipes read from standard input.
