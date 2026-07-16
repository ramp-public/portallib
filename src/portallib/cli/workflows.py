"""Execution wrappers connecting parsed recipes to the public PorTAL API."""

from __future__ import annotations

import json
import sys
from typing import Any

import torch

from ..data import ChoiceDataset
from ..evaluation import PortalEvaluator
from ..model import PortalModel
from .recipes import CommonRecipe, EvaluateRecipe, RefitRecipe, TrainRecipe
from ..runtime import load_base, load_dataset, model_slug, runtime_device
from ..training import EpochMetrics, PortalAdapterRefitter, PortalCoreTrainer, PortalTrainingConfig


def _emit(value: dict[str, Any], *, stream: Any | None = None) -> None:
    print(json.dumps(value, sort_keys=True), file=stream or sys.stdout, flush=True)


def _write_result(recipe: CommonRecipe, value: dict[str, Any]) -> None:
    _emit(value)
    if recipe.result_path is not None:
        recipe.result_path.parent.mkdir(parents=True, exist_ok=True)
        recipe.result_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _epoch_event(phase: str, epoch: EpochMetrics) -> dict[str, Any]:
    return {
        "event": "epoch",
        "phase": phase,
        "epoch": epoch.epoch,
        "acc_norm": epoch.macro_accuracy,
        "gold_nll": epoch.macro_gold_nll,
        "bases": {
            name: {"acc_norm": result.macro_accuracy, "gold_nll": result.macro_gold_nll}
            for name, result in epoch.evaluations.items()
        },
    }


def _load_dataset(recipe: CommonRecipe) -> ChoiceDataset:
    return load_dataset(recipe.dataset.source, revision=recipe.dataset.revision)


def _run_train(recipe: TrainRecipe) -> None:
    training = recipe.training.overrides()
    torch.manual_seed(training.get("seed", PortalTrainingConfig.seed))
    dataset = _load_dataset(recipe)
    tasks = recipe.tasks or dataset.tasks
    device, dtype = runtime_device(recipe.runtime.device, recipe.runtime.dtype)
    config = PortalTrainingConfig(**training, checkpoint_dir=recipe.output_dir / "checkpoints")
    bases = [load_base(base.to_runtime(), device=device, dtype=dtype) for base in recipe.bases]
    result = PortalCoreTrainer(bases, dataset, tasks=tasks, config=config).train(
        on_epoch=lambda epoch: _emit(_epoch_event("train", epoch))
    )
    recipe.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for base in recipe.bases:
        destination = recipe.output_dir / f"source-{model_slug(base.model_id)}"
        result.artifacts[base.model_id].save_pretrained(destination)
        outputs[base.model_id] = str(destination)
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
    training = recipe.training.overrides()
    seed = training.get("seed", PortalTrainingConfig.seed)
    torch.manual_seed(seed)
    dataset = _load_dataset(recipe)
    source = PortalModel.from_pretrained(recipe.source_artifact, revision=recipe.source_artifact_revision)
    device, dtype = runtime_device(recipe.runtime.device, recipe.runtime.dtype)
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
    dataset = _load_dataset(recipe)
    portal = PortalModel.from_pretrained(recipe.artifact, revision=recipe.artifact_revision)
    if portal.config.base_model_name_or_path != recipe.base.model_id:
        raise ValueError(
            f"artifact expects {portal.config.base_model_name_or_path!r}, but base.model_id is {recipe.base.model_id!r}"
        )
    tasks = recipe.tasks or tuple(portal.config.tasks)
    device, dtype = runtime_device(recipe.runtime.device, recipe.runtime.dtype)
    base = load_base(recipe.base.to_runtime(), device=device, dtype=dtype)
    evaluator = PortalEvaluator(max_prompt=recipe.max_prompt, batch_size=recipe.batch_size)
    base_result = evaluator.evaluate(base, dataset, tasks=tasks, max_examples=recipe.max_examples)
    portal_result = evaluator.evaluate(
        base,
        dataset,
        tasks=tasks,
        portal=portal,
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
