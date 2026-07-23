#!/usr/bin/env python3
"""Refit a trained Hugging Face PorTAL artifact onto a new raw base model.

Edit the recipe block, then run:

    python examples/refit_example.py

Either jointly trained source artifact supplies the same frozen task latents and canonical core.
The checked-in recipe uses the 4B-specific copy; only a fresh target-base alignment is trained, and
the original 1.7B and 4B source models are not downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from portallib import (
    BaseModelSpec,
    PortalAdapterRefitter,
    PortalModel,
    PortalTrainingConfig,
    load_base,
    load_dataset,
    runtime_device,
)

# ---------------------------------------------------------------------------
# Recipe: trained PorTAL artifact + raw Hugging Face target -> refitted artifact.
# ---------------------------------------------------------------------------

# Both source artifacts contain the same jointly trained task latents and canonical core.
SOURCE_ARTIFACT = "RampPublic/portal-qwen3-4b"
SOURCE_ARTIFACT_REVISION: str | None = "v0.2.0"
TARGET_BASE = BaseModelSpec(
    "Qwen/Qwen3-8B",
    "b968826d9c46dd6066d109eabc6255188de91218",
)

# Cross-family target alternative:
# TARGET_BASE = BaseModelSpec(
#     "google/gemma-3-4b-pt",
#     "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
#     layer_path="model.language_model.layers",
# )

DATASET = "RampPublic/portallib-tasks"
DATASET_REVISION = "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
OUTPUT_DIR = Path("artifacts/portal-qwen3-8b-1000")
REFIT_MAX_EXAMPLES = 1000
EVAL_MAX_EXAMPLES = 1000
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
LR_SCHEDULER = "linear"
WARMUP_RATIO = 0.1
SEED = 0

# ---------------------------------------------------------------------------
# End recipe.
# ---------------------------------------------------------------------------


def refit_config(source: PortalModel) -> PortalTrainingConfig:
    return PortalTrainingConfig.from_portal_config(
        source.config,
        refit_max_examples=REFIT_MAX_EXAMPLES,
        eval_max_examples=EVAL_MAX_EXAMPLES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        seed=SEED,
        checkpoint_dir=OUTPUT_DIR / "checkpoints",
    )


def print_epoch(epoch) -> None:
    print(json.dumps({"phase": "refit", **epoch.to_dict()}), flush=True)


def main() -> None:
    torch.manual_seed(SEED)
    dataset = load_dataset(DATASET, revision=DATASET_REVISION)
    source = PortalModel.from_pretrained(SOURCE_ARTIFACT, revision=SOURCE_ARTIFACT_REVISION)
    device, dtype = runtime_device()
    target = load_base(TARGET_BASE, device=device, dtype=dtype)
    result = PortalAdapterRefitter(source, target, dataset, config=refit_config(source)).refit(on_epoch=print_epoch)

    result.artifact.save_pretrained(OUTPUT_DIR)
    print(
        json.dumps(
            {
                "source_artifact": SOURCE_ARTIFACT,
                "target_base": TARGET_BASE.model_id,
                "best_epoch": result.best_epoch,
                "best_loss_epoch": result.best_loss_epoch,
                "output": str(OUTPUT_DIR),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
