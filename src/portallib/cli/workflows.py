"""Execution wrappers connecting parsed recipes to the public PorTAL API."""

from __future__ import annotations

import json
import sys
from typing import Any

import torch

from ..data import ChoiceDataset
from ..evaluation import PortalEvaluator
from ..model import PortalModel
from ..runtime import load_base, load_dataset, runtime_device
from ..training import EpochMetrics, PortalAdapterRefitter, PortalCoreTrainer, PortalTrainingConfig
from .recipes import CommonRecipe, EvaluateRecipe, RefitRecipe, TrainRecipe


def _emit(value: dict[str, Any], *, stream: Any | None = None) -> None:
    print(json.dumps(value, sort_keys=True), file=stream or sys.stdout, flush=True)


def _write_result(recipe: CommonRecipe, value: dict[str, Any]) -> None:
    _emit(value)
    if recipe.result_path is not None:
        recipe.result_path.parent.mkdir(parents=True, exist_ok=True)
        recipe.result_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _epoch_event(phase: str, epoch: EpochMetrics) -> dict[str, Any]:
    return {"event": "epoch", "phase": phase, **epoch.to_dict()}


def _load_dataset(recipe: CommonRecipe) -> ChoiceDataset:
    return load_dataset(recipe.dataset.source, revision=recipe.dataset.revision)


def _load_runtime(recipe: CommonRecipe) -> tuple[ChoiceDataset, torch.device, torch.dtype]:
    dataset = _load_dataset(recipe)
    device, dtype = runtime_device(recipe.runtime.device, recipe.runtime.dtype)
    return dataset, device, dtype


def _training_overrides(recipe: TrainRecipe | RefitRecipe) -> dict[str, Any]:
    training = recipe.training.overrides()
    torch.manual_seed(training.get("seed", PortalTrainingConfig.seed))
    return training


def _run_train(recipe: TrainRecipe) -> None:
    training = _training_overrides(recipe)
    dataset, device, dtype = _load_runtime(recipe)
    tasks = recipe.tasks or dataset.tasks
    config = PortalTrainingConfig(**training, checkpoint_dir=recipe.output_dir / "checkpoints")
    bases = [load_base(base.to_runtime(), device=device, dtype=dtype) for base in recipe.bases]
    result = PortalCoreTrainer(bases, dataset, tasks=tasks, config=config).train(
        on_epoch=lambda epoch: _emit(_epoch_event("train", epoch))
    )
    outputs = {
        model_id: str(destination) for model_id, destination in result.save_pretrained(recipe.output_dir).items()
    }
    _write_result(
        recipe,
        {
            "event": "result",
            "kind": "train",
            "best_epoch": result.best_epoch,
            "best_loss_epoch": result.best_loss_epoch,
            "outputs": outputs,
        },
    )


def _run_refit(recipe: RefitRecipe) -> None:
    training = _training_overrides(recipe)
    dataset, device, dtype = _load_runtime(recipe)
    source = PortalModel.from_pretrained(recipe.source_artifact, revision=recipe.source_artifact_revision)
    target = load_base(recipe.base.to_runtime(), device=device, dtype=dtype)
    config = PortalTrainingConfig.from_portal_config(
        source.config,
        **training,
        checkpoint_dir=recipe.output_dir / "checkpoints",
    )
    result = PortalAdapterRefitter(source, target, dataset, config=config).refit(
        on_epoch=lambda epoch: _emit(_epoch_event("refit", epoch))
    )
    result.artifact.save_pretrained(recipe.output_dir)
    _write_result(
        recipe,
        {
            "event": "result",
            "kind": "refit",
            "source_artifact": recipe.source_artifact,
            "target_base": recipe.base.model_id,
            "best_epoch": result.best_epoch,
            "best_loss_epoch": result.best_loss_epoch,
            "output": str(recipe.output_dir),
        },
    )


def _run_evaluate(recipe: EvaluateRecipe) -> None:
    dataset, device, dtype = _load_runtime(recipe)
    portal = PortalModel.from_pretrained(recipe.artifact, revision=recipe.artifact_revision)
    portal.validate_base_model(recipe.base.model_id, recipe.base.revision)
    tasks = recipe.tasks or tuple(portal.config.tasks)
    base = load_base(recipe.base.to_runtime(), device=device, dtype=dtype)
    evaluator = PortalEvaluator(max_prompt=recipe.max_prompt, batch_size=recipe.batch_size)
    base_result, portal_result = evaluator.compare(
        base,
        dataset,
        portal,
        tasks=tasks,
        max_examples=recipe.max_examples,
    )
    _write_result(
        recipe,
        {
            "event": "result",
            "kind": "evaluate",
            "artifact": recipe.artifact,
            "base": base_result.to_dict(),
            "portal": portal_result.to_dict(),
            "macro_accuracy_lift": portal_result.macro_accuracy - base_result.macro_accuracy,
        },
    )
