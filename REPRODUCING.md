# Training PorTAL artifacts

The [Ramp Labs announcement](https://x.com/RampLabs/status/2072381992285647280) introduces PorTAL.
This guide pins the public data and model inputs and provides complete commands for source training
and target-base refitting.

## Pinned inputs

| Input | Revision |
|---|---|
| Normalized task dataset | `3f0a5e6e028a56cf6029bb4761df97d0ff36731d` |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| `google/gemma-3-4b-pt` | `cc012e0a6d0787b4adcc0fa2c4da74402494554d` |

The normalized rows are available from
[`RampPublic/portallib-tasks`](https://huggingface.co/datasets/RampPublic/portallib-tasks).
[`examples/prepare_dataset.py`](examples/prepare_dataset.py) rebuilds the local JSON representation
from pinned upstream datasets. Its canonical JSON serialization contains 129,212 training rows and
19,548 validation rows and has SHA-256
`c5aec929f1800a3f1f4b3150aa1c9e464356fdb0cd11645c29df5b78efcdec00`.

## Configuration

| Setting | Value |
|---|---:|
| Source bases | Qwen3-1.7B and Qwen3-4B |
| Target modules | query and value projections (`qv`) |
| Rank / alpha | 8 / 16 |
| Task latent width | 256 |
| Layer embedding width | 32 |
| Hidden width | 512 |
| Canonical width | 1024 |
| Source examples per task and epoch | up to 2,000 |
| Source subset | deterministic resampling each epoch |
| Refit examples per task | command-specific |
| Validation examples per task | up to 1,000 |
| Epochs / batch size | 5 / 4 |
| Core/alignment learning rate | `1e-3` |
| Task-latent learning rate | `2e-3` |
| Optimizer / weight decay | AdamW / 0 |
| Gradient clipping | 1.0 |
| Loss EMA decay / floor | 0.9 / `1e-3` |
| Maximum prompt tokens | 768 |
| Gradient checkpointing | enabled, non-reentrant |

Each balanced source round draws one batch for every `(base, task)` unit. Per-unit losses are
normalized by their EMA, and per-base task-latent gradients are norm-equalized before the optimizer
step. Each base uses its own tokenizer.

Evaluation runs before training and after every epoch. Choices are ranked by continuation
log-probability normalized by character length (`acc_norm`), with token-mean gold continuation NLL
reported separately. The saved artifact maximizes macro validation `acc_norm`, breaking ties with
lower NLL. The independently minimum-NLL epoch is also reported.

## 1. Train the source artifacts

This command jointly trains the shared task-latent table and canonical core, then saves one
base-specific artifact for Qwen3-1.7B and Qwen3-4B.

```bash
python examples/train_example.py \
  --dataset RampPublic/portallib-tasks \
  --dataset-revision 3f0a5e6e028a56cf6029bb4761df97d0ff36731d \
  --output portal-qwen3-sources \
  --base-model Qwen/Qwen3-1.7B \
  --base-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --base-model Qwen/Qwen3-4B \
  --base-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --modules qv \
  --source-max-train 2000 \
  --source-steps-per-epoch 0 \
  --eval-max-examples 1000 \
  --epochs 5 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --latent-learning-rate 0.002 \
  --seed 0
```

## 2. Train and refit Qwen3-8B

This end-to-end command saves both source artifacts before loading Qwen3-8B and fitting its target
alignment. The shared core and task latents remain frozen during refitting.

```bash
python examples/train_example.py \
  --dataset RampPublic/portallib-tasks \
  --dataset-revision 3f0a5e6e028a56cf6029bb4761df97d0ff36731d \
  --output portal-qwen3-8b \
  --base-model Qwen/Qwen3-1.7B \
  --base-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --base-model Qwen/Qwen3-4B \
  --base-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --refit-base-model Qwen/Qwen3-8B \
  --refit-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --refit-max-train 1000 \
  --eval-max-examples 1000 \
  --epochs 5 \
  --seed 0
```

## 3. Train and refit Gemma-3-4B

Gemma uses an explicit decoder-layer path. The source configuration and task order remain identical
to the Qwen recipe.

```bash
python examples/train_example.py \
  --dataset RampPublic/portallib-tasks \
  --dataset-revision 3f0a5e6e028a56cf6029bb4761df97d0ff36731d \
  --output portal-gemma-3-4b \
  --base-model Qwen/Qwen3-1.7B \
  --base-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --base-model Qwen/Qwen3-4B \
  --base-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --refit-base-model google/gemma-3-4b-pt \
  --refit-revision cc012e0a6d0787b4adcc0fa2c4da74402494554d \
  --refit-layer-path model.language_model.layers \
  --refit-max-train 1000 \
  --eval-max-examples 1000 \
  --epochs 5 \
  --seed 0
```

Run seeds 0, 1, and 2 for aggregate results. Save every source artifact before unloading its base
model; refitting only updates the target alignment.
