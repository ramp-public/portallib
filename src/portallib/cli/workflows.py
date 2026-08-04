"""Execution wrappers connecting parsed recipes to the public PorTAL API."""

from __future__ import annotations

import json
import sys
from typing import Any

import torch

from ..config import PortalConfig
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


def _load_runtime(recipe: CommonRecipe) -> tuple[ChoiceDataset, torch.device, torch.dtype]:
    dataset = load_dataset(recipe.dataset.source, revision=recipe.dataset.revision)
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


def _inspect_payload(config: PortalConfig, *, source: str, revision: str | None) -> dict[str, Any]:
    """Summarize a config-only artifact view for the ``inspect`` command."""
    modules = list(config.modules)
    targets_by_module: dict[str, dict[str, Any]] = {}
    for name in modules:
        module_targets = [target for target in config.projection_targets if target.module_name == name]
        dimensions = sorted({target.dimensions for target in module_targets})
        targets_by_module[name] = {
            "targets": len(module_targets),
            "layers": len({target.layer_index for target in module_targets}),
            "dimensions": [list(dimension) for dimension in dimensions],
        }
    input_groups, output_groups = config.alignment_groups
    total_targets = len(config.projection_targets)
    heterogeneous = any(len(module["dimensions"]) != 1 for module in targets_by_module.values())
    return {
        "event": "inspect",
        "source": source,
        "revision": revision,
        "format_version": config.format_version,
        "base_model": config.base_model_name_or_path,
        "base_model_revision": config.base_model_revision,
        "architecture": config.architecture,
        "task_type": config.task_type,
        "n_layers": config.n_layers,
        "tasks": list(config.tasks),
        "modules": modules,
        "rank": config.rank,
        "alpha": config.alpha,
        "scaling": config.scaling,
        "d_z": config.d_z,
        "d_layer": config.d_layer,
        "hidden": config.hidden,
        "d_core": config.d_core,
        "layer_path": config.layer_path,
        "projection_targets": total_targets,
        "heterogeneous": heterogeneous,
        "sparse": total_targets != config.n_layers * len(modules),
        "targets_by_module": targets_by_module,
        "alignment_groups": {"input": input_groups, "output": output_groups},
    }


def _render_inspect(payload: dict[str, Any]) -> str:
    """Render a human-readable summary of an inspect payload."""
    revision = payload["revision"] or "(unspecified)"
    base_revision = payload["base_model_revision"] or "(unpinned)"
    layout = "heterogeneous" if payload["heterogeneous"] else "uniform"
    if payload["sparse"]:
        layout += ", sparse"
    lines = [
        f"artifact:        {payload['source']} @ {revision}",
        f"format_version:  {payload['format_version']} ({payload['architecture']}, {payload['task_type']})",
        f"base_model:      {payload['base_model']} @ {base_revision}",
        f"layers:          {payload['n_layers']}  (layer_path={payload['layer_path']})",
        f"tasks ({len(payload['tasks'])}):       {', '.join(payload['tasks'])}",
        (
            f"dims:            rank={payload['rank']} alpha={payload['alpha']} "
            f"scaling={payload['scaling']:.4g} d_z={payload['d_z']} "
            f"d_layer={payload['d_layer']} hidden={payload['hidden']} d_core={payload['d_core']}"
        ),
        f"projection ({payload['projection_targets']} targets, {layout}):",
    ]
    for name, module in payload["targets_by_module"].items():
        dims = "; ".join(f"{in_features}->{out_features}" for in_features, out_features in module["dimensions"])
        lines.append(f"  {name:<5} {module['targets']} targets over {module['layers']} layers  [{dims}]")
    input_groups, output_groups = payload["alignment_groups"]["input"], payload["alignment_groups"]["output"]
    lines.append(f"alignment in:    {input_groups}")
    lines.append(f"alignment out:   {output_groups}")
    return "\n".join(lines)


def _run_inspect(source: str, *, revision: str | None = None, as_json: bool = False) -> None:
    """Print an artifact's configuration without downloading weights or the base model."""
    config = PortalConfig.from_pretrained(source, revision=revision)
    payload = _inspect_payload(config, source=source, revision=revision)
    if as_json:
        _emit(payload)
    else:
        print(_render_inspect(payload), flush=True)
