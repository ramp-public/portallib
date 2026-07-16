# Training the release artifacts

This guide defines the release recipe for the four native PorTAL artifacts: two jointly trained
source artifacts and two target-base refits. The [Ramp Labs
announcement](https://x.com/RampLabs/status/2072381992285647280) introduces PorTAL. The pinned
configuration below reproduces the paper's source-training, target-refitting, and evaluation method;
it does not make a fixed numerical-performance guarantee across hardware or dependency versions.

## Install the released package and recipes

The library is distributed through PyPI. The runnable recipes remain in the repository so their
model IDs, revisions, data budgets, and optimization settings are visible and editable.

```bash
git clone https://github.com/ramp-public/portallib
cd portallib
python -m pip install 'portallib[training]==0.1.1'
```

Install the CUDA-compatible PyTorch build required by your GPU platform before the command above
when the default PyPI wheel is not appropriate.

## Pinned inputs

| Input | Revision |
|---|---|
| [`RampPublic/portallib-tasks`](https://huggingface.co/datasets/RampPublic/portallib-tasks) | `ffc3c0e44f529bf64a5ae62ed5db090952db97ea` |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| `google/gemma-3-4b-pt` | `cc012e0a6d0787b4adcc0fa2c4da74402494554d` |

## Rebuild the canonical task data

[`scripts/prepare_dataset.py`](scripts/prepare_dataset.py) downloads pinned revisions of the 14
upstream benchmarks and reproduces the prompt and choice normalization used by the release recipes:

```bash
python scripts/prepare_dataset.py --output portal_tasks.json
```

The generated JSON contains 129,212 training rows and 19,548 validation rows. Its SHA-256 is
`97ae9193a02b96daec13f7e21f56fbe7ed5102fd900e6c2093d9bbfc009f74cd`, allowing a locally rebuilt
copy to be checked against the pinned Hub input. To train from the local file, set
`DATASET = "portal_tasks.json"` and `DATASET_REVISION = None` in the example recipe.

Pass `--tasks rte,boolq` to prepare a smaller subset. The script writes locally by default. Dataset
publication is an explicit operation:

```bash
python scripts/prepare_dataset.py --output portal_tasks.json \
  --push-to-hub your-namespace/portal-tasks --private
```

The upload includes [`scripts/portal_tasks_dataset_card.md`](scripts/portal_tasks_dataset_card.md)
in the same Hub commit as the normalized splits. Review the terms of every selected upstream
dataset before redistributing rows: portallib's Apache-2.0 license covers the software, not the
mixed-license benchmark collection.

The input is either a local JSON file or a Hugging Face dataset repository with `train` and
`validation` splits. Each row uses this schema:

```json
{
  "task": "rte",
  "prompt": "Premise: ...\nHypothesis: ...\nEntailment?",
  "choices": [" yes", " no"],
  "gold_idx": 0
}
```

A local JSON file wraps these rows in `train` and `validation` arrays.

## Shared architecture and evaluation

| Setting | Value |
|---|---:|
| Target modules | query and value projections (`q`, `v`) |
| Rank / alpha | 8 / 16 |
| Task latent width | 256 |
| Layer embedding width | 32 |
| Hidden width | 512 |
| Canonical width | 1024 |
| Batch size | 4 |
| Core/alignment learning rate | `1e-3` |
| Task-latent learning rate | `2e-3` during source training |
| Schedule / warmup | linear decay / 10% of optimizer steps |
| Optimizer / weight decay | AdamW / 0 |
| Gradient clipping | 1.0 |
| Loss EMA decay / floor | 0.9 / `1e-3` |
| Maximum prompt tokens | 768 |
| Validation examples per task | up to 1,000 |
| Gradient checkpointing | enabled, non-reentrant |

Evaluation runs before optimization and after every epoch. Choices are ranked by continuation
log-probability divided by character length (`acc_norm`); token-mean gold continuation NLL is
reported separately. Candidate continuations are batched during evaluation; `eval_batch_size`
controls the memory/throughput tradeoff without changing the metric definition. The saved artifact
is the epoch with maximum macro validation `acc_norm`, with
lower NLL as the tie-breaker. The independently minimum-NLL epoch is also reported.

## Phase 1: jointly train the source artifacts

[`examples/train_example.py`](examples/train_example.py) loads Qwen3-1.7B and Qwen3-4B together. It
learns one task-latent table, one canonical core, and one alignment for each source base.

| Source setting | Value |
|---|---:|
| Training examples per task | up to 2,000 |
| Subset policy | deterministic leading subset |
| Epochs | 12 |
| Balanced optimizer rounds per epoch | 500 |

Every optimizer round draws one batch for each `(base, task)` unit. Shorter task pools recycle with a
deterministic seeded order. Per-unit losses are EMA-normalized, and each source base's task-latent
gradient is norm-equalized before the shared latent update. Each base uses its own tokenizer.

```bash
python examples/train_example.py
```

The run writes one complete artifact per source base:

- [`RampPublic/portal-qwen3-1.7b`](https://huggingface.co/RampPublic/portal-qwen3-1.7b)
- [`RampPublic/portal-qwen3-4b`](https://huggingface.co/RampPublic/portal-qwen3-4b)

The two artifacts contain identical task latents and canonical core weights. Each contains only the
alignment and exact model metadata for its own base.

## Phase 2: refit Qwen3-8B and Gemma 3 4B

[`examples/refit_example.py`](examples/refit_example.py) loads either source artifact as a carrier
for the task latents and canonical core learned jointly from Qwen3-1.7B and Qwen3-4B. It freezes
those shared weights and trains only a new alignment for the selected target base.

| Refit setting | Value |
|---|---:|
| Training examples per task | up to 1,000 |
| Epochs | 5 |
| Optimizer steps per epoch | maximum number of task batches |

The checked-in recipe reads the identical shared weights from the 4B-specific copy at
`RampPublic/portal-qwen3-4b` and targets Qwen3-8B:

```bash
python examples/refit_example.py
```

To create the cross-family artifact, select the adjacent Gemma 3 target recipe and its explicit
`model.language_model.layers` path. Both refits use the same 1,000-example budget and best-epoch
selection rule. The original Qwen source models are not loaded during refitting.

The resulting release artifacts are:

- [`RampPublic/portal-qwen3-8b`](https://huggingface.co/RampPublic/portal-qwen3-8b)
- [`RampPublic/portal-gemma-3-4b`](https://huggingface.co/RampPublic/portal-gemma-3-4b)

## Phase 3: reload and evaluate

[`examples/evaluate_example.py`](examples/evaluate_example.py) loads a published native artifact and
its matching frozen base, evaluates both, and reports per-task results, macro results, and accuracy
lift. Its default is the Qwen3-8B refit.

```bash
python examples/evaluate_example.py
```

Change the artifact and matching `BaseRecipe` together to evaluate the 1.7B, 4B, or Gemma artifact.
Release metrics should be taken from this clean Hub reload, not from an in-memory training object.

[`COMPUTE.md`](COMPUTE.md) describes equivalent Docker and Modal launch patterns. Changing the
compute platform does not change the package-level training recipe.
