# Reproducing the published Qwen3 source artifacts

This document records the exact data, model snapshots, training configuration, selection rule, and
results for these native artifacts:

- [`RampPublic/portallib-qwen3-1.7b`](https://huggingface.co/RampPublic/portallib-qwen3-1.7b)
- [`RampPublic/portallib-qwen3-4b`](https://huggingface.co/RampPublic/portallib-qwen3-4b)

## Data provenance

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

## Pinned inputs

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

## Complete configuration

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

## Public reproduction command

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

This source-only form writes one artifact per source base. The original combined job then loaded
Qwen3-8B and refit a fresh alignment with at most 500 examples per task. Source artifacts were saved
before refitting; refit optimization cannot modify their frozen shared core or task latents.

The remaining architecture and optimization values in the table are the canonical v0.1 defaults in
`PortalTrainingConfig`. They are listed explicitly here so the recipe does not depend on readers
discovering implicit defaults.

## Recorded source results

| Base | Epoch-zero `acc_norm` | Selected epoch-3 `acc_norm` | Change |
|---|---:|---:|---:|
| Qwen3-1.7B | 0.5820 | 0.6841 | +0.1021 |
| Qwen3-4B | 0.6389 | 0.7368 | +0.0979 |
| Mean across source bases | 0.6104 | 0.7105 | +0.1000 |

These are full-validation, one-seed research results. Exact floating-point reproduction can still
vary across PyTorch, CUDA, and GPU versions even when data, code, model snapshots, and seeds are
pinned.
