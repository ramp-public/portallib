"""Configuration-driven command line interface for PorTAL workflows."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from . import __version__
from .config import SUPPORTED_MODULES
from .data import ChoiceDataset
from .evaluation import PortalEvaluator
from .model import PortalModel
from .runtime import BaseRecipe, load_base, load_dataset, model_slug, runtime_device
from .training import EpochMetrics, PortalAdapterRefitter, PortalCoreTrainer, PortalTrainingConfig


class RecipeError(ValueError):
    """A structurally invalid CLI recipe."""


@dataclass(frozen=True)
class DatasetRecipe:
    source: str
    revision: str | None
    local: bool


@dataclass(frozen=True)
class RuntimeRecipe:
    device: str = "auto"
    dtype: str = "auto"


@dataclass(frozen=True)
class CommonRecipe:
    kind: Literal["train", "refit", "evaluate"]
    dataset: DatasetRecipe
    runtime: RuntimeRecipe
    tasks: tuple[str, ...] | None
    result_path: Path | None
    config_path: Path


@dataclass(frozen=True)
class TrainRecipe(CommonRecipe):
    output_dir: Path
    bases: tuple[BaseRecipe, ...]
    training: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefitRecipe(CommonRecipe):
    output_dir: Path
    source_artifact: str
    source_artifact_revision: str | None
    base: BaseRecipe
    training: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluateRecipe(CommonRecipe):
    artifact: str
    artifact_revision: str | None
    base: BaseRecipe
    max_examples: int | None = 1000
    max_prompt: int = 768
    batch_size: int = 8


CliRecipe = TrainRecipe | RefitRecipe | EvaluateRecipe

_COMMON_KEYS = {"schema_version", "kind", "dataset", "runtime", "tasks", "result_path"}
_TRAIN_KEYS = _COMMON_KEYS | {"output_dir", "bases", "training"}
_REFIT_KEYS = (_COMMON_KEYS - {"tasks"}) | {
    "output_dir",
    "source_artifact",
    "source_artifact_revision",
    "base",
    "training",
}
_EVALUATE_KEYS = _COMMON_KEYS | {
    "artifact",
    "artifact_revision",
    "base",
    "max_examples",
    "max_prompt",
    "batch_size",
}
_BASE_KEYS = {
    "model_id",
    "revision",
    "layer_path",
    "module_paths",
    "dtype",
    "device_map",
    "attn_implementation",
}
_TRAINING_KEYS = {
    "modules",
    "rank",
    "alpha",
    "d_z",
    "d_layer",
    "hidden",
    "d_core",
    "source_max_examples",
    "source_resample_each_epoch",
    "source_steps_per_epoch",
    "refit_max_examples",
    "eval_max_examples",
    "eval_batch_size",
    "epochs",
    "batch_size",
    "learning_rate",
    "latent_learning_rate",
    "lr_scheduler",
    "warmup_ratio",
    "weight_decay",
    "grad_clip",
    "ema_decay",
    "ema_floor",
    "max_prompt",
    "seed",
    "gradient_checkpointing",
    "early_stopping_patience",
    "task_regression_threshold",
}
_REFIT_TRAINING_KEYS = _TRAINING_KEYS - {
    "modules",
    "rank",
    "alpha",
    "d_z",
    "d_layer",
    "hidden",
    "d_core",
    "source_max_examples",
    "source_resample_each_epoch",
    "source_steps_per_epoch",
    "latent_learning_rate",
}
_INTEGER_TRAINING_KEYS = {
    "rank",
    "alpha",
    "d_z",
    "d_layer",
    "hidden",
    "d_core",
    "source_steps_per_epoch",
    "refit_max_examples",
    "eval_batch_size",
    "epochs",
    "batch_size",
    "max_prompt",
    "seed",
    "early_stopping_patience",
}
_FLOAT_TRAINING_KEYS = {
    "learning_rate",
    "latent_learning_rate",
    "warmup_ratio",
    "weight_decay",
    "grad_clip",
    "ema_decay",
    "ema_floor",
    "task_regression_threshold",
}
_BOOLEAN_TRAINING_KEYS = {"source_resample_each_epoch", "gradient_checkpointing"}
_ALL_OR_INTEGER_TRAINING_KEYS = {"source_max_examples", "eval_max_examples"}


def _check_keys(
    value: dict[str, Any],
    *,
    allowed: set[str],
    required: set[str] = frozenset(),
    context: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RecipeError(f"{context} has unknown keys: {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise RecipeError(f"{context} is missing required keys: {', '.join(missing)}")


def _table(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecipeError(f"{context} must be a TOML table")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: Any, context: str) -> str | None:
    return None if value is None else _string(value, context)


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecipeError(f"{context} must be a positive integer")
    return value


def _path(value: Any, *, parent: Path, context: str) -> Path:
    path = Path(_string(value, context)).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _location(value: Any, *, parent: Path, context: str) -> str:
    location = _string(value, context)
    if location.startswith(("./", "../", "~/")) or Path(location).is_absolute():
        return str(_path(location, parent=parent, context=context))
    return location


def _tasks(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or any(not isinstance(task, str) or not task.strip() for task in value):
        raise RecipeError("tasks must be a non-empty array of task names")
    if len(value) != len(set(value)):
        raise RecipeError("tasks must not contain duplicates")
    return tuple(value)


def _parse_dataset(value: Any, *, parent: Path) -> DatasetRecipe:
    table = _table(value, "dataset")
    _check_keys(table, allowed={"repo_id", "path", "revision"}, context="dataset")
    has_repo = "repo_id" in table
    has_path = "path" in table
    if has_repo == has_path:
        raise RecipeError("dataset must define exactly one of repo_id or path")
    revision = _optional_string(table.get("revision"), "dataset.revision")
    if has_path:
        if revision is not None:
            raise RecipeError("dataset.revision is not allowed with dataset.path")
        source = str(_path(table["path"], parent=parent, context="dataset.path"))
    else:
        source = _string(table["repo_id"], "dataset.repo_id")
    return DatasetRecipe(source, revision, has_path)


def _parse_runtime(value: Any) -> RuntimeRecipe:
    if value is None:
        return RuntimeRecipe()
    table = _table(value, "runtime")
    _check_keys(table, allowed={"device", "dtype"}, context="runtime")
    device = _string(table.get("device", "auto"), "runtime.device")
    dtype = _string(table.get("dtype", "auto"), "runtime.dtype")
    if dtype not in {"auto", "bfloat16", "float16", "float32"}:
        raise RecipeError("runtime.dtype must be 'auto', 'bfloat16', 'float16', or 'float32'")
    try:
        if device != "auto":
            torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise RecipeError(f"runtime.device is invalid: {device!r}") from exc
    return RuntimeRecipe(device, dtype)


def _parse_base(value: Any, context: str, *, parent: Path) -> BaseRecipe:
    table = _table(value, context)
    _check_keys(table, allowed=_BASE_KEYS, required={"model_id"}, context=context)
    module_paths = table.get("module_paths")
    if module_paths is not None:
        module_paths = _table(module_paths, f"{context}.module_paths")
        if not module_paths:
            raise RecipeError(f"{context}.module_paths must not be empty")
        if any(not isinstance(name, str) or not isinstance(path, str) for name, path in module_paths.items()):
            raise RecipeError(f"{context}.module_paths must map strings to strings")
    dtype = _optional_string(table.get("dtype"), f"{context}.dtype")
    if dtype == "auto":
        dtype = None
    if dtype is not None and dtype not in {"bfloat16", "float16", "float32"}:
        raise RecipeError(f"{context}.dtype must be 'auto', 'bfloat16', 'float16', or 'float32'")
    device_map = table.get("device_map")
    if device_map is not None and not isinstance(device_map, (str, dict)):
        raise RecipeError(f"{context}.device_map must be a string or TOML table")
    if isinstance(device_map, dict) and any(
        not isinstance(name, str) or isinstance(target, bool) or not isinstance(target, (str, int))
        for name, target in device_map.items()
    ):
        raise RecipeError(f"{context}.device_map values must be device strings or integer device indexes")
    try:
        return BaseRecipe(
            model_id=_location(table["model_id"], parent=parent, context=f"{context}.model_id"),
            revision=_optional_string(table.get("revision"), f"{context}.revision"),
            layer_path=_string(table.get("layer_path", "model.layers"), f"{context}.layer_path"),
            module_paths=module_paths,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=_optional_string(
                table.get("attn_implementation"),
                f"{context}.attn_implementation",
            ),
        )
    except ValueError as exc:
        raise RecipeError(f"{context} is invalid: {exc}") from exc


def _parse_training(value: Any, *, refit: bool) -> dict[str, Any]:
    if value is None:
        return {}
    table = _table(value, "training")
    allowed = _REFIT_TRAINING_KEYS if refit else _TRAINING_KEYS
    _check_keys(table, allowed=allowed, context="training")
    parsed: dict[str, Any] = {}
    for name, raw in table.items():
        context = f"training.{name}"
        if name == "modules":
            if not isinstance(raw, list) or not raw or any(not isinstance(item, str) or not item for item in raw):
                raise RecipeError(f"{context} must be a non-empty array of module names")
            if len(raw) != len(set(raw)):
                raise RecipeError(f"{context} must not contain duplicates")
            unknown = sorted(set(raw) - SUPPORTED_MODULES)
            if unknown:
                raise RecipeError(f"{context} has unsupported module names: {', '.join(unknown)}")
            parsed[name] = tuple(raw)
        elif name in _INTEGER_TRAINING_KEYS:
            parsed[name] = _positive_int(raw, context) if name != "seed" else _integer(raw, context)
        elif name in _FLOAT_TRAINING_KEYS:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise RecipeError(f"{context} must be a number")
            parsed[name] = float(raw)
        elif name in _BOOLEAN_TRAINING_KEYS:
            if not isinstance(raw, bool):
                raise RecipeError(f"{context} must be true or false")
            parsed[name] = raw
        elif name in _ALL_OR_INTEGER_TRAINING_KEYS:
            if raw == "all":
                parsed[name] = None
            else:
                parsed[name] = _positive_int(raw, context)
        elif name == "lr_scheduler":
            parsed[name] = _string(raw, context)
        else:
            raise AssertionError(f"unhandled training field: {name}")
    try:
        PortalTrainingConfig(**parsed)
    except (TypeError, ValueError) as exc:
        raise RecipeError(f"training is invalid: {exc}") from exc
    return parsed


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeError(f"{context} must be an integer")
    return value


def load_recipe(path: str | Path) -> CliRecipe:
    """Parse and strictly validate one versioned TOML recipe without loading models."""
    config_path = Path(path).expanduser().resolve()
    try:
        value = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RecipeError(f"invalid TOML in {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecipeError("recipe root must be a TOML table")
    if value.get("schema_version") != 1:
        raise RecipeError("schema_version must be 1")
    kind = value.get("kind")
    if kind not in {"train", "refit", "evaluate"}:
        raise RecipeError("kind must be 'train', 'refit', or 'evaluate'")
    allowed = {"train": _TRAIN_KEYS, "refit": _REFIT_KEYS, "evaluate": _EVALUATE_KEYS}[kind]
    required = {
        "train": {"schema_version", "kind", "dataset", "output_dir", "bases"},
        "refit": {"schema_version", "kind", "dataset", "output_dir", "source_artifact", "base"},
        "evaluate": {"schema_version", "kind", "dataset", "artifact", "base"},
    }[kind]
    _check_keys(value, allowed=allowed, required=required, context="recipe")
    parent = config_path.parent
    common = {
        "kind": kind,
        "dataset": _parse_dataset(value["dataset"], parent=parent),
        "runtime": _parse_runtime(value.get("runtime")),
        "tasks": _tasks(value.get("tasks")),
        "result_path": (
            _path(value["result_path"], parent=parent, context="result_path") if "result_path" in value else None
        ),
        "config_path": config_path,
    }
    if kind == "train":
        bases = value["bases"]
        if not isinstance(bases, list) or not bases:
            raise RecipeError("bases must contain at least one [[bases]] table")
        parsed_bases = tuple(_parse_base(base, f"bases[{index}]", parent=parent) for index, base in enumerate(bases))
        model_ids = [base.model_id for base in parsed_bases]
        if len(model_ids) != len(set(model_ids)):
            raise RecipeError("bases model_id values must be unique")
        return TrainRecipe(
            **common,
            output_dir=_path(value["output_dir"], parent=parent, context="output_dir"),
            bases=parsed_bases,
            training=_parse_training(value.get("training"), refit=False),
        )
    if kind == "refit":
        return RefitRecipe(
            **common,
            output_dir=_path(value["output_dir"], parent=parent, context="output_dir"),
            source_artifact=_location(value["source_artifact"], parent=parent, context="source_artifact"),
            source_artifact_revision=_optional_string(
                value.get("source_artifact_revision"),
                "source_artifact_revision",
            ),
            base=_parse_base(value["base"], "base", parent=parent),
            training=_parse_training(value.get("training"), refit=True),
        )
    max_examples_raw = value.get("max_examples", 1000)
    max_examples = None if max_examples_raw == "all" else _positive_int(max_examples_raw, "max_examples")
    return EvaluateRecipe(
        **common,
        artifact=_location(value["artifact"], parent=parent, context="artifact"),
        artifact_revision=_optional_string(value.get("artifact_revision"), "artifact_revision"),
        base=_parse_base(value["base"], "base", parent=parent),
        max_examples=max_examples,
        max_prompt=_positive_int(value.get("max_prompt", 768), "max_prompt"),
        batch_size=_positive_int(value.get("batch_size", 8), "batch_size"),
    )


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


def _load_recipe_dataset(recipe: DatasetRecipe) -> ChoiceDataset:
    if recipe.local:
        return ChoiceDataset.from_json(recipe.source)
    return load_dataset(recipe.source, revision=recipe.revision)


def _run_train(recipe: TrainRecipe) -> None:
    torch.manual_seed(recipe.training.get("seed", PortalTrainingConfig.seed))
    dataset = _load_recipe_dataset(recipe.dataset)
    tasks = recipe.tasks or dataset.tasks
    device, dtype = runtime_device(recipe.runtime.device, recipe.runtime.dtype)
    config = PortalTrainingConfig(**recipe.training, checkpoint_dir=recipe.output_dir / "checkpoints")
    bases = [load_base(base, device=device, dtype=dtype) for base in recipe.bases]
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
    seed = recipe.training.get("seed", PortalTrainingConfig.seed)
    torch.manual_seed(seed)
    dataset = _load_recipe_dataset(recipe.dataset)
    source = PortalModel.from_pretrained(recipe.source_artifact, revision=recipe.source_artifact_revision)
    device, dtype = runtime_device(recipe.runtime.device, recipe.runtime.dtype)
    target = load_base(recipe.base, device=device, dtype=dtype)
    config = PortalTrainingConfig.from_portal_config(
        source.config,
        **recipe.training,
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
    dataset = _load_recipe_dataset(recipe.dataset)
    portal = PortalModel.from_pretrained(recipe.artifact, revision=recipe.artifact_revision)
    if portal.config.base_model_name_or_path != recipe.base.model_id:
        raise ValueError(
            f"artifact expects {portal.config.base_model_name_or_path!r}, but base.model_id is {recipe.base.model_id!r}"
        )
    tasks = recipe.tasks or tuple(portal.config.tasks)
    device, dtype = runtime_device(recipe.runtime.device, recipe.runtime.dtype)
    base = load_base(recipe.base, device=device, dtype=dtype)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portallib", description="Run PorTAL workflows from strict TOML recipes.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "refit", "evaluate", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path, help="Path to a versioned TOML recipe.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        recipe = load_recipe(args.config)
    except (OSError, RecipeError) as exc:
        _emit({"event": "error", "stage": "config", "message": str(exc)}, stream=sys.stderr)
        return 2
    if args.command == "validate":
        _emit({"event": "validated", "kind": recipe.kind, "schema_version": 1})
        return 0
    if recipe.kind != args.command:
        _emit(
            {
                "event": "error",
                "stage": "config",
                "message": f"recipe kind {recipe.kind!r} cannot run with command {args.command!r}",
            },
            stream=sys.stderr,
        )
        return 2
    try:
        if isinstance(recipe, TrainRecipe):
            _run_train(recipe)
        elif isinstance(recipe, RefitRecipe):
            _run_refit(recipe)
        else:
            _run_evaluate(recipe)
    except Exception as exc:
        _emit(
            {
                "event": "error",
                "stage": "runtime",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            stream=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
