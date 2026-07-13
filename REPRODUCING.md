# Reproducing PorTAL experiments

The original PorTAL research results were [announced by Ramp
Labs](https://x.com/RampLabs/status/2072381992285647280). This document separates the benchmark
recipe from the earlier expanded-suite experiment whose source artifacts are already public.

## Published expanded-suite source artifacts

This document records the exact data, model snapshots, training configuration, selection rule, and
results for these native artifacts:

- [`RampPublic/portallib-qwen3-1.7b`](https://huggingface.co/RampPublic/portallib-qwen3-1.7b)
- [`RampPublic/portallib-qwen3-4b`](https://huggingface.co/RampPublic/portallib-qwen3-4b)

### Data provenance

The training job read a local normalized `dataset.json`. That file is exactly reproduced by
`examples/prepare_dataset.py`: regenerating it produced the same 129,212 training rows, 19,548
validation rows, task order, and SHA-256
`c5aec929f1800a3f1f4b3150aa1c9e464356fdb0cd11645c29df5b78efcdec00`.

The same rows are public at
[`RampPublic/portallib-tasks`](https://huggingface.co/datasets/RampPublic/portallib-tasks), revision
`3f0a5e6e028a56cf6029bb4761df97d0ff36731d`. Hugging Face stores the public copy as Parquet, so the
JSON checksum applies to output from the preparation script, not to the Hub's Parquet files.

Source training used every available training row as the sampling pool. It did not make one
sequential pass through that pool. Each epoch contained 1,000 balanced optimizer rounds; every
round drew one batch for each `(base, task)` pair, and shorter tasks recycled deterministically.

### Pinned inputs

| Input | Revision |
|---|---|
| Public portallib recipe | `d14bd83f4b7dc00d26aea622f139ca9e29fb13ab` |
| Normalized dataset | `3f0a5e6e028a56cf6029bb4761df97d0ff36731d` |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Original refit target, `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |

The model revisions above were verified against the snapshot directories used by the training job.
The canonical trainer, evaluator, and decoder in the pinned public revision are the implementations
used to produce the artifacts.

### Complete published-artifact configuration

| Setting | Value |
|---|---:|
| Source bases | Qwen3-1.7B and Qwen3-4B, jointly trained |
| Target modules | query and value projections (`qv`) |
| Rank / alpha | 8 / 16 |
| Task latent width | 256 |
| Layer embedding width | 32 |
| Hidden width | 512 |
| Canonical width | 1024 |
| Source sampling pool | all available examples per task |
| Balanced source rounds per epoch | 1,000 |
| Maximum training epochs | 5 |
| Batch size | 4 per `(base, task)` unit |
| Core/alignment learning rate | `1e-3` |
| Task-latent learning rate | `2e-3` |
| Optimizer | AdamW |
| Weight decay | 0 |
| Gradient clipping | 1.0 |
| Loss EMA decay / floor | 0.9 / `1e-3` |
| Maximum prompt tokens | 768 |
| Gradient checkpointing | enabled, non-reentrant |
| Seed | 0 |
| Early-stopping patience | 1 non-improving validation epoch |
| Hardware / model dtype | NVIDIA H200 / bfloat16 |

Per-unit losses were divided by their EMA. After accumulating every task for one base, its latent
gradient was captured; per-base latent gradients were norm-equalized before the optimizer step. Each
base used its own tokenizer. Evaluation ran before training and after every epoch using
character-normalized continuation log-probability (`acc_norm`), with token-mean gold continuation
NLL reported separately.

The saved shared checkpoint maximized mean validation `acc_norm` across both source bases, breaking
ties with lower NLL. One non-improving epoch stopped training, and the trainer restored epoch 3.

### Published source-artifact command

Run from the pinned public portallib checkout after installing the `training` extra:

```bash
python examples/train_example.py \
  --dataset RampPublic/portallib-tasks \
  --dataset-revision 3f0a5e6e028a56cf6029bb4761df97d0ff36731d \
  --output portal-qwen3-14task-s0 \
  --base-model Qwen/Qwen3-1.7B \
  --base-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --base-model Qwen/Qwen3-4B \
  --base-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --modules qv \
  --source-max-train 0 \
  --source-steps-per-epoch 1000 \
  --refit-max-train 500 \
  --epochs 5 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --latent-learning-rate 0.002 \
  --seed 0
```

This source-only form writes one artifact per source base. The validation job that produced the
published source artifacts also performed a reduced-data Qwen3-8B refit with at most 500 examples
per task. That downstream check did not produce the source artifacts and is not the paper-scale
refitting recipe. Source artifacts were saved before refitting; refit optimization cannot modify
their frozen shared core or task latents.

The remaining architecture and optimization values in the table are listed explicitly so this
historical recipe does not depend on current defaults.

## Paper reproduction recipe

The paper uses the complete 14-task suite. Both joint source training and target-base refitting cap
each task at 2,000 training examples. With batch size 4, a full epoch has as many balanced optimizer
rounds as the longest capped task has batches (at most 500); shorter tasks recycle. Evaluation caps
each task at 1,000 examples and reports character-normalized choice accuracy (`acc_norm`) plus
token-mean gold continuation NLL.

The source bases are Qwen3-1.7B and Qwen3-4B. Qwen3-8B and Gemma-3-4B are unseen refit targets. Each
phase evaluates epoch zero and every one of five training epochs. The saved checkpoint maximizes
macro validation `acc_norm`, breaking ties with lower NLL; the minimum-NLL epoch is also reported.
No early stopping is used in the paper recipe.

The complete source-training plus Qwen3-8B paper-scale command is:

```bash
python examples/train_example.py \
  --dataset RampPublic/portallib-tasks \
  --dataset-revision 3f0a5e6e028a56cf6029bb4761df97d0ff36731d \
  --output portal-qwen3-14task-paper-s0 \
  --base-model Qwen/Qwen3-1.7B \
  --base-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --base-model Qwen/Qwen3-4B \
  --base-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --refit-base-model Qwen/Qwen3-8B \
  --refit-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --modules qv \
  --source-max-train 2000 \
  --static-source-subset \
  --source-steps-per-epoch 0 \
  --refit-max-train 2000 \
  --eval-max-examples 1000 \
  --epochs 5 \
  --early-stopping-patience 0 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --latent-learning-rate 0.002 \
  --seed 0
```

`--static-source-subset` preserves the paper's fixed source subset. Without it, portallib's expanded
default draws a new deterministic capped subset each epoch so the shared core can see more of the
available source data at the same number of balanced rounds. The zero values for
`--source-steps-per-epoch` and `--early-stopping-patience` mean “derive a full
epoch” and “run every epoch,” respectively. Seeds 0, 1, and 2 reproduce the paper's three-seed
protocol. For Gemma, replace the refit model and revision and add
`--refit-layer-path model.language_model.layers`.

The currently published source artifacts cannot seed a paper reproduction because their shared core
was trained with the expanded-suite settings documented above. A paper run must retrain the shared
core and task latents before refitting an unseen base. New paper artifacts will only be published
after the validation run is complete; stopped or failed runs are not validation evidence.

### Recorded expanded-suite source results

| Base | Epoch-zero `acc_norm` | Selected epoch-3 `acc_norm` | Change |
|---|---:|---:|---:|
| Qwen3-1.7B | 0.5820 | 0.6841 | +0.1021 |
| Qwen3-4B | 0.6389 | 0.7368 | +0.0979 |
| Mean across source bases | 0.6104 | 0.7105 | +0.1000 |

These are full-validation, one-seed research results. Exact floating-point reproduction can still
vary across PyTorch, CUDA, and GPU versions even when data, code, model snapshots, and seeds are
pinned.
