#!/usr/bin/env python3
"""Train PorTAL source artifacts from raw Hugging Face base models.

Edit the recipe block, then run:

    python examples/train_example.py

This stage learns the shared task latents, canonical core, and one alignment per source base. It
saves a complete base-specific PorTAL artifact for every source base.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from portallib import PortalCoreTrainer, PortalTrainingConfig
from utils import BaseRecipe, load_base, load_dataset, model_slug, runtime_device


# ---------------------------------------------------------------------------
# Recipe: raw Hugging Face bases -> trained PorTAL source artifacts.
# ---------------------------------------------------------------------------

DATASET = "RampPublic/portallib-tasks"
DATASET_REVISION = "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
OUTPUT_DIR = Path("artifacts/portal-qwen-sources")
TASKS: tuple[str, ...] | None = None

SOURCE_BASES = (
    BaseRecipe("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    BaseRecipe("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
)

TRAINING_CONFIG = PortalTrainingConfig(
    modules=("q", "v"),
    rank=8,
    alpha=16,
    d_z=256,
    d_layer=32,
    hidden=512,
    d_core=1024,
    source_max_examples=2000,
    source_resample_each_epoch=False,
    source_steps_per_epoch=500,
    eval_max_examples=1000,
    epochs=12,
    batch_size=4,
    learning_rate=1e-3,
    latent_learning_rate=2e-3,
    lr_scheduler="linear",
    warmup_ratio=0.1,
    seed=0,
)

# ---------------------------------------------------------------------------
# End recipe.
# ---------------------------------------------------------------------------


def print_epoch(epoch) -> None:
    print(
        json.dumps(
            {
                "phase": "source",
                "epoch": epoch.epoch,
                "acc_norm": epoch.macro_accuracy,
                "gold_nll": epoch.macro_gold_nll,
                "bases": {
                    name: {"acc_norm": result.macro_accuracy, "gold_nll": result.macro_gold_nll}
                    for name, result in epoch.evaluations.items()
                },
            }
        ),
        flush=True,
    )


def main() -> None:
    if not SOURCE_BASES:
        raise ValueError("SOURCE_BASES must contain at least one base")
    if len({base.model_id for base in SOURCE_BASES}) != len(SOURCE_BASES):
        raise ValueError("SOURCE_BASES model IDs must be unique")

    torch.manual_seed(TRAINING_CONFIG.seed)
    dataset = load_dataset(DATASET, revision=DATASET_REVISION)
    tasks = TASKS or dataset.tasks
    device, dtype = runtime_device()
    config = replace(TRAINING_CONFIG, checkpoint_dir=OUTPUT_DIR / "checkpoints")
    bases = [load_base(base, device=device, dtype=dtype) for base in SOURCE_BASES]
    result = PortalCoreTrainer(bases, dataset, tasks=tasks, config=config).train(on_epoch=print_epoch)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for base in SOURCE_BASES:
        destination = OUTPUT_DIR / f"source-{model_slug(base.model_id)}"
        result.artifacts[base.model_id].save_pretrained(destination)
        outputs[base.model_id] = str(destination)

    print(
        json.dumps(
            {
                "best_epoch": result.best_epoch,
                "best_loss_epoch": result.best_loss_epoch,
                "outputs": outputs,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
